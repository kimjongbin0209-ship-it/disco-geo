"""Explicit development budgets and supported recipes for the new study."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    output: str = "runs/development/disco-seed0"
    method: str = "disco"
    recipe: str = "fresh"
    purpose: str = "development"
    # "cue_recall" is the synthetic positive control (development only), never a study task.
    environment: str = "memory_maze"
    control_delay: int = 0
    control_episode_steps: int = 240
    control_fixed_cue: int = -1  # 0..5: a one-answer bandit; -1: random cues
    maze_size: int = 9
    camera_resolution: int = 64
    seed: int = 0
    steps: int = 4096
    num_envs: int = 2
    rollout_length: int = 32
    channels: tuple[int, ...] = (32, 64, 64)
    feature_dim: int = 128
    hidden_size: int = 128
    learning_rate: float = 3e-4
    entropy_cost: float = 0.01
    eval_every: int = 50
    validation_seeds: tuple[int, ...] = (1000000, 1000001)
    # Greedy argmax of a near-uniform policy is a fixed-action policy; "sampled" scores the policy itself.
    validation_policy: str = "greedy"
    # Log whether each update raises the taken action's log-probability more on steps
    # followed by a reward within this many steps than on steps that are not (0 = off).
    credit_window: int = 0
    replay_weight: float = 0.2
    replay_capacity_episodes: int = 16
    replay_max_age: int = 20
    replay_max_abs_log_ratio: float = 5.0
    save_trajectories: bool = True
    # Step each training environment in its own process (identical data, parallel stepping).
    env_processes: bool = False
    # Positive-advantage n-step self-imitation on replayed reward-containing chunks.
    self_imitation_weight: float = 0.0
    self_imitation_max_age: int = 100
    # SDPO-style feedback self-teacher. "none" disables every teacher component.
    teacher: str = "none"
    teacher_feedback: str = "actual"
    teacher_query: str = "actual_success"
    teacher_gate: str = "evidence"
    teacher_horizon: int = 64
    teacher_hidden: int = 128
    teacher_l2: float = 0.0
    teacher_learning_weight: float = 1.0
    distillation_weight: float = 0.0
    teacher_ema_rate: float = 0.01
    teacher_warmup_updates: int = 20
    teacher_min_success_steps: int = 256
    teacher_min_information_gain: float = 0.01
    teacher_max_copy_rate: float = 0.5

    @property
    def batch_size(self):
        return self.num_envs

    def __post_init__(self):
        if self.method not in {"disco", "actor_critic", "actor_critic_aux"}:
            raise ValueError("Choose disco, actor_critic, or actor_critic_aux")
        if self.recipe not in {"fresh", "recent_replay"}:
            raise ValueError("recipe selects the rule-loss data: fresh or recent_replay; "
                             "SDPO is configured through the teacher fields")
        if not isinstance(self.env_processes, bool) or not isinstance(self.save_trajectories, bool):
            raise ValueError("env_processes and save_trajectories must be booleans")
        if self.teacher not in {"none", "outcome", "outcome_memory"}:
            raise ValueError("teacher must be none, outcome, or outcome_memory")
        if self.teacher_feedback not in {"actual", "shuffled"}:
            raise ValueError("teacher_feedback must be actual or shuffled (control)")
        if self.teacher_query not in {"actual_success", "counterfactual_success"}:
            raise ValueError("teacher_query must be actual_success or counterfactual_success")
        if self.teacher_gate not in {"evidence", "always", "never"}:
            raise ValueError("teacher_gate must be evidence, always, or never")
        if self.teacher == "none" and (self.distillation_weight or self.teacher_gate != "evidence"
                                       or self.teacher_feedback != "actual"):
            raise ValueError("Teacher settings require a teacher variant")
        for key in ("teacher_horizon", "self_imitation_max_age"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.teacher_hidden, int) or self.teacher_hidden < 0:
            raise ValueError("teacher_hidden must be a nonnegative integer (0 = linear head)")
        if not math.isfinite(self.teacher_l2) or self.teacher_l2 < 0:
            raise ValueError("teacher_l2 must be finite and nonnegative")
        for key in ("teacher_warmup_updates", "teacher_min_success_steps"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        for key in ("self_imitation_weight", "teacher_learning_weight", "distillation_weight"):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 <= value <= 10:
                raise ValueError(f"{key} must be finite in [0, 10]")
        if not 0 < self.teacher_ema_rate <= 1 or not 0 < self.teacher_max_copy_rate <= 1:
            raise ValueError("teacher EMA rate and copy-rate limit must be in (0, 1]")
        if not math.isfinite(self.teacher_min_information_gain):
            raise ValueError("teacher_min_information_gain must be finite")
        if self.purpose not in {"development", "pilot", "confirmatory"}:
            raise ValueError("Unknown study purpose")
        if self.purpose == "confirmatory":
            raise ValueError("Confirmatory training requires a locked validated protocol; not enabled")
        if self.maze_size not in {9, 11, 13, 15}:
            raise ValueError("Use an official Memory Maze size")
        if self.environment not in {"memory_maze", "cue_recall"}:
            raise ValueError("environment must be memory_maze or cue_recall (positive control)")
        if self.environment == "cue_recall" and self.purpose != "development":
            raise ValueError("The cue_recall positive control is a development diagnostic only")
        for key in ("control_delay", "credit_window"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if (not isinstance(self.control_episode_steps, int)
                or self.control_episode_steps <= self.control_delay):
            raise ValueError("control_episode_steps must exceed control_delay")
        if (not isinstance(self.control_fixed_cue, int) or isinstance(self.control_fixed_cue, bool)
                or not -1 <= self.control_fixed_cue <= 5):
            raise ValueError("control_fixed_cue must be -1 (random) or an action index")
        if self.validation_policy not in {"greedy", "sampled"}:
            raise ValueError("validation_policy must be greedy or sampled")
        for key in ("steps", "num_envs", "rollout_length", "feature_dim", "hidden_size",
                    "eval_every", "replay_capacity_episodes"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 <= self.seed < 1000 or self.num_envs > 64:
            raise ValueError("Training seeds must stay in the reserved namespace")
        if self.steps % self.num_envs:
            raise ValueError("steps must be divisible by num_envs for an exact interaction budget")
        if not isinstance(self.camera_resolution, int) or self.camera_resolution < 64:
            raise ValueError("Memory Maze requires camera_resolution >= 64 to preserve its RGB border")
        if not self.channels or any(not isinstance(c, int) or c <= 0 for c in self.channels):
            raise ValueError("channels must contain positive integers")
        if (not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not 0 <= self.entropy_cost <= 1):
            raise ValueError("Invalid optimizer settings")
        if not 0 <= self.replay_weight < 1 or self.replay_max_age < 0:
            raise ValueError("Replay weight must be in [0,1) and age nonnegative")
        if (not math.isfinite(self.replay_max_abs_log_ratio)
                or not 0 < self.replay_max_abs_log_ratio <= 20):
            raise ValueError("Replay log-ratio eligibility limit must be finite in (0,20]")
        if len(self.validation_seeds) != len(set(self.validation_seeds)):
            raise ValueError("Validation seeds must be unique")
        if any(not 1000000 <= s < 2000000 for s in self.validation_seeds):
            raise ValueError("Validation seeds must be in [1000000,2000000)")
        if self.purpose != "development" and not self.validation_seeds:
            raise ValueError("Pilot runs require a fixed validation bank")
        if not self.output:
            raise ValueError("Choose a dedicated output directory")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path, **overrides):
        data = json.loads(Path(path).read_text())
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        data.update({k: v for k, v in overrides.items() if v is not None})
        for key in ("channels", "validation_seeds"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)
