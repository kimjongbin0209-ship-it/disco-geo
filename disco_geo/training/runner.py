import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
import dm_env
from disco_geo.agents.disco import GeoAgent
from disco_geo.data.bundle import RGBBundle
from disco_geo.envs.ring import RingEnvironment
from disco_geo.io import load_checkpoint, save_checkpoint, sha256, write_json
from disco_geo.training.snapshots import save_encoder_snapshot
from disco_geo.upstream import DISCO_COMMIT, RULE_HASH, ensure_disco

ensure_disco()
from disco_rl import types  # noqa: E402


def make_env(config, split="train", batch_size=None):
    bundle = RGBBundle(config.dataset, split)
    images, transitions = bundle.training_arrays()
    env = RingEnvironment(
        images,
        transitions,
        batch_size or config.batch_size,
        config.task,
        config.max_moves,
        config.move_cost,
    )
    return bundle, env


def make_collector(agent, env, length):
    def collect(params, actor_state, env_state, ts, key):
        def step(carry, rng):
            state, timestep, net_state = carry
            actor, next_net_state = agent.actor_step(params, rng, timestep, net_state)
            valid = ~state.done
            next_state, next_ts = env.step(state, actor.actions)
            return (next_state, next_ts, next_net_state), (actor, valid)

        (env_state, ts, actor_state), (actors, valid) = jax.lax.scan(
            step, (env_state, ts, actor_state), jax.random.split(key, length)
        )
        # Bootstrap observation is recorded without stepping the environment.
        bootstrap, _ = agent.actor_step(params, jax.random.fold_in(key, length), ts, actor_state)
        stacked = jax.tree.map(
            lambda a, b: jnp.concatenate((a, b[None]), axis=0), actors, bootstrap
        )
        return types.ActorRollout.from_timestep(stacked), actor_state, env_state, ts, valid.sum()

    return jax.jit(collect)


def evaluate_policy(agent, params, config, split="rl_val", episodes=128, seed=123, random=False):
    start_time = time.monotonic()
    _, env = make_env(config, split, batch_size=min(episodes, 32))
    state, ts = env.reset(jax.random.PRNGKey(seed))
    apply = jax.jit(agent._network.one_step)
    step = jax.jit(env.step)
    net_state = agent.initial_actor_state(jax.random.PRNGKey(seed))
    rng = np.random.default_rng(seed)
    returns = np.zeros(env.batch_size)
    successes, completed_returns, completed_moves = [], [], []
    for _ in range((config.max_moves + 2) * (episodes // env.batch_size + 2)):
        if random:
            actions = rng.integers(env.num_actions, size=env.batch_size, dtype=np.int32)
        else:
            out, net_state = apply(params, net_state, ts.observation, state.done)
            actions = jnp.argmax(out["logits"], axis=-1).astype(jnp.int32)
        state, ts = step(state, actions)
        reward, terminal = np.asarray(ts.reward), np.asarray(ts.step_type) == dm_env.StepType.LAST
        returns += reward
        for i in np.flatnonzero(terminal):
            successes.append(float(reward[i] > 0))
            completed_returns.append(float(returns[i]))
            completed_moves.append(int(np.asarray(state.moves)[i]))
            returns[i] = 0
        if len(successes) >= episodes:
            break
    if len(successes) < episodes:
        raise RuntimeError("Policy evaluation failed to complete the requested episodes")
    s = np.asarray(successes[:episodes])
    moves = np.asarray(completed_moves[:episodes])
    return {
        "evaluation_wall_seconds": time.monotonic() - start_time,
        "episodes": episodes,
        "success_rate": float(s.mean()),
        "mean_return": float(np.mean(completed_returns[:episodes])),
        "mean_moves": float(moves.mean()),
        "moves_on_success": float(moves[s > 0].mean()) if s.any() else None,
    }


def train(config, resume=False):
    output = Path(config.output)
    if (output / "config.json").exists() and not resume:
        raise FileExistsError(f"Run exists; choose a new output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    bundle, env = make_env(config)
    agent = GeoAgent(config, env)
    key = jax.random.PRNGKey(config.seed)
    # Identical encoder module names and initialization key across methods.
    learner = agent.initial_learner_state(key)
    initial_encoder = {k: v for k, v in learner.params.items() if k.startswith("visual_encoder/")}
    actor_state = agent.initial_actor_state(key)
    env_state, ts = env.reset(jax.random.fold_in(key, 1))
    rule = agent.rule_params()
    dataset_hash = sha256(bundle.root / "manifest.json")
    transitions, updates, best, previous_wall = 0, 0, -float("inf"), 0.0
    if resume:
        saved = load_checkpoint(output / "latest.pkl")
        old_config = dict(saved["config"])
        new_config = config.to_dict()
        # Increasing the target budget and relocating a copied run are allowed.
        for key in ("steps", "output", "dataset"):
            old_config.pop(key, None)
            new_config.pop(key, None)
        old_config.setdefault("snapshot_every", 0)
        if old_config != new_config or saved["dataset_manifest_sha256"] != dataset_hash:
            raise ValueError("Resume requires the same model, training settings, and dataset")
        if config.steps < saved["config"]["steps"]:
            raise ValueError("Resume cannot decrease the original target budget")
        learner, actor_state = saved["learner"], saved["actor_state"]
        env_state, ts, key = saved["env_state"], saved["timestep"], saved["rng"]
        initial_encoder = saved["initial_encoder"]
        transitions, updates = saved["transitions"], saved["updates"]
        previous_wall = saved.get("elapsed_wall_seconds", 0.0)
        best = saved.get("best_validation_return", -float("inf"))
        if (
            config.method != "untrained"
            and not np.isfinite(best)
            and (output / "best.pkl").exists()
        ):
            best = evaluate_policy(
                agent,
                load_checkpoint(output / "best.pkl")["learner"].params,
                config,
                episodes=config.eval_episodes,
            )["mean_return"]
        # A crash can leave a logged update newer than the last atomic checkpoint.
        if (output / "metrics.jsonl").exists():
            from disco_geo.evaluation.report import read_jsonl

            records = [r for r in read_jsonl(output / "metrics.jsonl") if r["updates"] <= updates]
            (output / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    write_json(output / "config.json", config.to_dict())
    if not resume:
        write_json(
            output / "provenance.json",
            {
                "disco_commit": DISCO_COMMIT,
                "rule_sha256": RULE_HASH,
                "dataset_manifest_sha256": sha256(bundle.root / "manifest.json"),
                "jax_version": jax.__version__,
                "devices": [str(d) for d in jax.devices()],
                "geometry_supervision": False,
                "fresh_rollouts": True,
                "warning": "Loss ablations perturb the frozen rule's training distribution.",
            },
        )
    collect = make_collector(agent, env, config.rollout_length)
    learner_step = jax.jit(agent.learner_step, static_argnames=("is_meta_training",))
    start = time.monotonic()

    def checkpoint(path):
        save_checkpoint(
            path,
            jax.device_get(
                {
                    "config": config.to_dict(),
                    "learner": learner,
                    "actor_state": actor_state,
                    "env_state": env_state,
                    "timestep": ts,
                    "rng": key,
                    "transitions": transitions,
                    "updates": updates,
                    "best_validation_return": best,
                    "elapsed_wall_seconds": previous_wall + time.monotonic() - start,
                    "initial_encoder": initial_encoder,
                    "dataset_manifest_sha256": dataset_hash,
                }
            ),
        )

    if config.snapshot_every and not resume:
        save_encoder_snapshot(output, config, learner.params, dataset_hash, 0, 0)
    if (
        config.snapshot_every
        and resume
        and (updates % config.snapshot_every == 0 or transitions >= saved["config"]["steps"])
    ):
        # Stop may occur after latest.pkl was committed but before its snapshot/manifest.
        point = output / "snapshots" / f"iter{updates:07d}"
        policy = None
        if config.method != "untrained" and not (point / "iteration.json").exists():
            policy = evaluate_policy(agent, learner.params, config, episodes=config.eval_episodes)
        save_encoder_snapshot(
            output, config, learner.params, dataset_hash, updates, transitions, policy
        )

    if config.method == "untrained":
        if not resume:
            checkpoint(output / "best.pkl")
            checkpoint(output / "latest.pkl")
        write_json(
            output / "completed.json",
            {
                "transitions": 0,
                "updates": 0,
                "elapsed_wall_seconds": previous_wall + time.monotonic() - start,
            },
        )
        return output
    with (output / "metrics.jsonl").open("a" if resume else "w") as log:
        while transitions < config.steps:
            key, actor_key, learner_key = jax.random.split(key, 3)
            rollout, actor_state, env_state, ts, valid = collect(
                learner.params, actor_state, env_state, ts, actor_key
            )
            learner, _, metrics = learner_step(
                learner_key, rollout, learner, actor_state, rule, False
            )
            transitions += int(valid)
            updates += 1
            loss = float(metrics["total_loss"])
            if not np.isfinite(loss):
                raise FloatingPointError(f"Nonfinite learner loss at update {updates}")
            snapshot_due = config.snapshot_every and (
                updates % config.snapshot_every == 0 or transitions >= config.steps
            )
            if (
                updates == 1
                or updates % config.eval_every == 0
                or transitions >= config.steps
                or snapshot_due
            ):
                policy = evaluate_policy(
                    agent, learner.params, config, episodes=config.eval_episodes
                )
                record = {
                    "transitions": transitions,
                    "updates": updates,
                    "wall_seconds": previous_wall + time.monotonic() - start,
                    "loss": loss,
                    "gradient_norm": float(metrics["global_gradient_norm"]),
                    **policy,
                }
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                # Selection uses reward on RL validation, never geometric probes.
                if policy["mean_return"] > best:
                    best = policy["mean_return"]
                    checkpoint(output / "best.pkl")
                checkpoint(output / "latest.pkl")
                if snapshot_due:
                    save_encoder_snapshot(
                        output, config, learner.params, dataset_hash, updates, transitions, policy
                    )
    write_json(
        output / "completed.json",
        {
            "transitions": transitions,
            "updates": updates,
            "elapsed_wall_seconds": previous_wall + time.monotonic() - start,
        },
    )
    return output
