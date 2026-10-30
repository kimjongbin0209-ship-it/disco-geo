"""Lazy CLI for protocol inspection, real benchmark pilots, and frozen readouts."""

import argparse
from dataclasses import replace
from importlib import metadata
import json
from pathlib import Path
import platform
import sys


def parser():
    p = argparse.ArgumentParser(prog="disco-spatial", description="Limited-reward spatial abilities")
    commands = p.add_subparsers(dest="command", required=True)
    q = commands.add_parser("protocol", help="Validate/export the research declaration; no training")
    q.add_argument("--config", default="configs/development.json")
    q.add_argument("--output")
    q = commands.add_parser("doctor", help="Inspect dependencies; optionally reset/step the real task")
    q.add_argument("--environment", action="store_true")
    q.add_argument("--maze-size", type=int, default=9)
    q.add_argument("--resolution", type=int, default=64)
    q.add_argument("--steps", type=int, default=4)
    q.add_argument("--output")
    q = commands.add_parser("train", help="Recurrent pilot: fresh/replay rule loss plus optional "
                                          "self-imitation or feedback self-teacher")
    q.add_argument("--config", default="configs/development.json")
    q.add_argument("--output")
    q.add_argument("--method", choices=("disco", "actor_critic", "actor_critic_aux"))
    q.add_argument("--recipe", choices=("fresh", "recent_replay"))
    for name in ("steps", "num-envs", "rollout-length", "seed"):
        q.add_argument(f"--{name}", type=int)
    q.add_argument("--teacher", choices=("none", "outcome", "outcome_memory"))
    q.add_argument("--teacher-feedback", choices=("actual", "shuffled"))
    q.add_argument("--teacher-query", choices=("actual_success", "counterfactual_success"))
    q.add_argument("--teacher-gate", choices=("evidence", "always", "never"))
    for name in ("distillation-weight", "self-imitation-weight"):
        q.add_argument(f"--{name}", type=float)
    q.add_argument("--no-evaluation", action="store_true",
                   help="Development wiring check only; no best-checkpoint claim")
    q = commands.add_parser("evaluate", help="Whole fixed-seed episodes, with an optional memory intervention")
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--seeds", nargs="+", type=int)
    q.add_argument("--reset-memory", action="store_true")
    q.add_argument("--teacher", action="store_true",
                   help="Let the causal success-query teacher act (teacher validation)")
    q.add_argument("--output", required=True)
    q = commands.add_parser("navigate", help="Frozen-policy test episodes: return, matched revisits, "
                                             "memory interventions (primary 3)")
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--seeds", nargs="+", type=int, required=True)
    q.add_argument("--interventions", nargs="+", default=["none", "reset_every_step"],
                   choices=("none", "reset_every_step", "truncate_16", "truncate_64"))
    q.add_argument("--sample", action="store_true", help="Sample actions instead of greedy")
    q.add_argument("--teacher", action="store_true")
    q.add_argument("--max-steps", type=int)
    q.add_argument("--output", required=True)
    q = commands.add_parser("scripted", help="Fixed reactive reference policy (uniform or an "
                                             "action bias): return that needs no learning")
    q.add_argument("--policy", default="uniform", help="'uniform' or '<action>:<probability>'")
    q.add_argument("--seeds", nargs="+", type=int, required=True)
    q.add_argument("--maze-size", type=int, default=9)
    q.add_argument("--resolution", type=int, default=64)
    q.add_argument("--max-steps", type=int)
    q.add_argument("--output", required=True)
    q = commands.add_parser("collect", help="Reference-policy common histories with render telemetry")
    q.add_argument("--output", required=True)
    q.add_argument("--partition", choices=("probe_train", "probe_validation", "probe_final_test"),
                   default="probe_train")
    q.add_argument("--seed", type=int, default=3000000)
    q.add_argument("--episodes", type=int, default=1)
    q.add_argument("--max-steps", type=int, default=1000)
    q.add_argument("--maze-size", type=int, default=9)
    q.add_argument("--resolution", type=int, default=64)
    q.add_argument("--policy", choices=("explore", "uniform"), default="explore")
    q = commands.add_parser("extract", help="Causal frozen features on an identical common-history dataset")
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--collection", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--chunk-size", type=int, default=128)
    q = commands.add_parser("probe", help="Generic layout-disjoint ridge diagnostic, not the full ability suite")
    q.add_argument("--train", required=True, dest="train_directory")
    q.add_argument("--validation", required=True, dest="validation_directory")
    q.add_argument("--test", required=True, dest="test_directory")
    q.add_argument("--output", required=True)
    q.add_argument("--representation", choices=("belief", "image_features"), default="belief")
    q.add_argument("--target", choices=("origin_relative_position", "origin_relative_heading",
                                        "targets_vec", "free_space_rays", "depth_coarse"),
                   default="origin_relative_position")
    q.add_argument("--stride", type=int, default=1)
    q = commands.add_parser("abilities", help="Frozen-representation ability suite over common histories")
    q.add_argument("--train", required=True, dest="train_directory")
    q.add_argument("--validation", required=True, dest="validation_directory")
    q.add_argument("--test", required=True, dest="test_directory")
    q.add_argument("--stride", type=int, default=5)
    q.add_argument("--output", required=True)
    q = commands.add_parser("report", help="Aggregate ability/navigation results over training seeds")
    q.add_argument("--runs", nargs="+", required=True,
                   help="label:seed:abilities.json[:navigation.json] entries")
    q.add_argument("--reference", required=True, help="Label used for paired seed differences")
    q.add_argument("--output", required=True)
    return p


def load_policy(checkpoint):
    """Rebuild an agent and its saved parameters from a trusted local checkpoint."""
    import numpy as np
    from dm_env import specs
    from disco_spatial.agents import RecurrentAgent
    from disco_spatial.config import TrainConfig
    from disco_spatial.io import load_checkpoint

    saved = load_checkpoint(checkpoint)
    config = TrainConfig(**dict(saved["config"], channels=tuple(saved["config"]["channels"]),
                                validation_seeds=tuple(saved["config"]["validation_seeds"])))

    class Specification:
        def single_observation_spec(self):
            return {"rgb": specs.Array((config.camera_resolution,) * 2 + (3,), np.uint8),
                    "previous_action": specs.Array((), np.int32),
                    "previous_reward": specs.Array((), np.float32)}

        def single_action_spec(self):
            return specs.BoundedArray((), np.int32, 0, 5)

    return RecurrentAgent(config, Specification()), saved, config


def doctor(environment=False, maze_size=9, resolution=64, steps=4):
    packages = {}
    for name in ("numpy", "jax", "jaxlib", "dm-haiku", "optax", "rlax", "dm-env",
                 "memory-maze", "dm-control", "mujoco", "labmaze", "gym"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": sys.version, "platform": platform.platform(), "packages": packages,
              "renderer_tested": False}
    if environment:
        import numpy as np
        from disco_spatial.envs.memory_maze import make_memory_maze

        if not 1 <= steps <= 100:
            raise ValueError("Doctor is a bounded reset/step test: choose 1..100 steps")
        env = make_memory_maze(maze_size, 1000000, "validation", resolution, evaluation=True,
                               telemetry=True)
        try:
            timestep = env.reset()
            before = timestep.observation["rgb"].copy()
            frames = [before]
            labels = env.last_sidecar
            floor = in_walls = 0
            for action in ([1, 2, 3, 0] * 25)[:steps]:
                timestep = env.step(action)
                frames.append(timestep.observation["rgb"].copy())
                floor += int(env.last_sidecar["floor_pixels"])
                in_walls += int(env.last_sidecar["floor_pixels_in_walls"])
            if set(timestep.observation) != {"rgb"}:
                raise RuntimeError("Privileged observation crossed the boundary")
            spatial_range = int(np.max(np.ptp(timestep.observation["rgb"], axis=(0, 1))))
            if spatial_range == 0:
                raise RuntimeError("Uniform RGB observation: inspect target-border/rendering configuration")
            result.update(renderer_tested=True, environment=env.metadata,
                          rgb_shape=list(timestep.observation["rgb"].shape),
                          learner_keys=list(timestep.observation),
                          evaluation_sidecar_keys=sorted(env.last_sidecar),
                          changed_rgb_fraction=float(np.mean(before != timestep.observation["rgb"])),
                          distinct_rgb_frames=len({frame.tobytes() for frame in frames}),
                          spatial_rgb_range=spatial_range,
                          position_delta=float(np.linalg.norm(env.last_sidecar["agent_pos"] - labels["agent_pos"])),
                          reward=float(timestep.reward or 0),
                          telemetry_floor_pixels=floor,
                          telemetry_floor_pixels_in_walls_fraction=in_walls / floor if floor else None)
        finally:
            env.close()
    return result


def main(argv=None):
    options = vars(parser().parse_args(argv))
    command = options.pop("command")
    if command == "protocol":
        from disco_spatial.config import TrainConfig
        from disco_spatial.evaluation.protocol import build_manifest
        from disco_spatial.io import write_json

        output = options.pop("output")
        result = build_manifest(TrainConfig.load(options["config"]).to_dict())
        if output:
            write_json(output, result)
    elif command == "doctor":
        from disco_spatial.io import write_json

        output = options.pop("output")
        result = doctor(**options)
        if output:
            write_json(output, result)
    elif command == "train":
        from disco_spatial.config import TrainConfig
        from disco_spatial.training import train

        path = options.pop("config")
        no_evaluation = options.pop("no_evaluation")
        config = TrainConfig.load(path, **options)
        if no_evaluation:
            if config.purpose != "development":
                raise ValueError("Only development wiring runs may omit evaluation")
            config = replace(config, validation_seeds=())
        result = train(config)
    elif command == "evaluate":
        from disco_spatial.io import write_json
        from disco_spatial.training import evaluate

        agent, saved, config = load_policy(options["checkpoint"])
        if options["seeds"] and any(not 1000000 <= seed < 3000000 for seed in options["seeds"]):
            raise ValueError("Policy evaluation must use reserved validation/test seed namespaces")
        if Path(options["output"]).exists():
            raise FileExistsError("Use a new evaluation output")
        head = saved.get("teacher_ema") if options["teacher"] else None
        if options["teacher"] and head is None:
            raise ValueError("This checkpoint has no trained teacher head")
        result = evaluate(agent, saved["learner"].params, config, options["seeds"],
                          options["reset_memory"], teacher_head=head)
        write_json(options["output"], result)
    elif command == "navigate":
        from disco_spatial.evaluation.navigation import evaluate_navigation
        from disco_spatial.io import write_json
        from disco_spatial.training import teacher_actor

        agent, saved, config = load_policy(options["checkpoint"])
        if Path(options["output"]).exists():
            raise FileExistsError("Use a new navigation output")
        params = saved["learner"].params
        factory = None
        if options["teacher"]:
            if saved.get("teacher_ema") is None:
                raise ValueError("This checkpoint has no trained teacher head")
            factory = lambda: teacher_actor(agent, params, saved["teacher_ema"])  # noqa: E731
        result = evaluate_navigation(
            agent, params, config.maze_size, config.camera_resolution, options["seeds"],
            interventions=tuple(options["interventions"]), greedy=not options["sample"],
            actor_factory=factory, max_steps=options["max_steps"])
        result.update(checkpoint=options["checkpoint"], training_seed=config.seed,
                      method=config.method, teacher_acting=options["teacher"])
        write_json(options["output"], result)
    elif command == "scripted":
        from disco_spatial.envs.memory_maze import make_memory_maze
        from disco_spatial.evaluation.scripted import scripted_returns
        from disco_spatial.io import write_json

        if any(not 2000000 <= seed < 3000000 for seed in options["seeds"]):
            raise ValueError("Reference policies use the policy-final-test seed namespace")
        if Path(options["output"]).exists():
            raise FileExistsError("Use a new reference-policy output")
        result = scripted_returns(make_memory_maze, options["seeds"], options["policy"],
                                  maze_size=options["maze_size"], resolution=options["resolution"],
                                  max_steps=options["max_steps"])
        write_json(options["output"], result)
        result = {key: result[key] for key in ("policy", "reward_per_1000_steps", "mean_return")}
    elif command == "collect":
        from disco_spatial.artifacts import collect_common_history

        result = collect_common_history(**options)
    elif command == "extract":
        from disco_spatial.artifacts import extract_common_features

        result = extract_common_features(**options)
    elif command == "probe":
        from disco_spatial.artifacts import probe_features

        result = probe_features(**options)
    elif command == "abilities":
        from disco_spatial.evaluation.suite import run_suite
        from disco_spatial.io import write_json

        output = Path(options.pop("output"))
        if output.exists():
            raise FileExistsError("Use a new ability output")
        result = run_suite(**options)
        write_json(output, result)
        result = {"output": str(output), "primary": result["primary"]}
    elif command == "report":
        from disco_spatial.evaluation.report import build_report
        from disco_spatial.io import write_json

        output = Path(options["output"])
        if output.exists():
            raise FileExistsError("Use a new report output")
        result = build_report(options["runs"], options["reference"])
        write_json(output, result)
        output.with_suffix(".md").write_text(result.pop("markdown"))
    else:
        raise AssertionError(command)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
