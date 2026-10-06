"""Measure the actual host after compilation; never present CPU timings as GPU timings."""

import time
import jax
from disco_geo.agents.disco import GeoAgent
from disco_geo.training.runner import make_collector, make_env, evaluate_policy
from disco_geo.io import write_json


def benchmark_rl(config, output, updates=20, warmup=3):
    if updates < 1 or warmup < 1 or config.method == "untrained":
        raise ValueError("Use a trained method and positive warmup/measured update counts")
    _, env = make_env(config)
    agent = GeoAgent(config, env)
    key = jax.random.PRNGKey(config.seed)
    learner, actor = agent.initial_learner_state(key), agent.initial_actor_state(key)
    state, ts = env.reset(key)
    rule = agent.rule_params()
    collect = make_collector(agent, env, config.rollout_length)
    step = jax.jit(agent.learner_step, static_argnames=("is_meta_training",))

    def update():
        nonlocal key, learner, actor, state, ts
        key, ak, lk = jax.random.split(key, 3)
        rollout, actor, state, ts, count = collect(learner.params, actor, state, ts, ak)
        learner, _, metrics = step(lk, rollout, learner, actor, rule, False)
        jax.block_until_ready(metrics["total_loss"])
        return int(count)

    start = time.monotonic()
    for _ in range(warmup):
        update()
    warmup_seconds = time.monotonic() - start
    start = time.monotonic()
    transitions = sum(update() for _ in range(updates))
    training_seconds = time.monotonic() - start
    start = time.monotonic()
    evaluate_policy(agent, learner.params, config, episodes=config.eval_episodes)
    evaluation_seconds = time.monotonic() - start
    rate = transitions / training_seconds
    estimate_updates = config.steps / (transitions / updates)
    # Upper bound on scheduled evaluations (overlap between cadences is counted twice).
    eval_count = 2 + estimate_updates / config.eval_every
    if config.snapshot_every:
        eval_count += estimate_updates / config.snapshot_every
    result = {
        "config": config.to_dict(),
        "devices": [str(d) for d in jax.devices()],
        "platform": jax.default_backend(),
        "warmup_including_compilation_seconds": warmup_seconds,
        "measured_updates": updates,
        "measured_transitions": transitions,
        "training_seconds": training_seconds,
        "transitions_per_second": rate,
        "evaluation_seconds_including_first_eval_compile": evaluation_seconds,
        "target_rl_compute_hours": config.steps / rate / 3600,
        "target_rl_plus_eval_hours": (config.steps / rate + eval_count * evaluation_seconds) / 3600,
        "limitations": "Short warmed-up sample; excludes disk checkpoints, probes, adapter fitting, generation and install/download. Policy changes can change transition throughput.",
    }
    write_json(output, result)
    return result
