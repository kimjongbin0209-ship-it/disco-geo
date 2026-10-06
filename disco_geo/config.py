"""Small validated configuration, intentionally without geometry-loss weights."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    dataset: str = "datasets/objects"
    output: str = "runs/disco/seed0"
    method: str = "disco"
    task: str = "recognition"
    seed: int = 0
    steps: int = 500_000
    batch_size: int = 16
    rollout_length: int = 16
    learning_rate: float = 0.0003
    feature_dim: int = 256
    channels: tuple[int, ...] = (32, 64, 64)
    max_moves: int = 12
    move_cost: float = 0.02
    eval_every: int = 20
    eval_episodes: int = 128
    ablation: str = "none"
    snapshot_every: int = 0

    def __post_init__(self):
        if self.method not in {"disco", "actor_critic", "actor_critic_aux", "untrained"}:
            raise ValueError(f"Unknown method: {self.method}")
        if self.task not in {"navigation", "recognition"}:
            raise ValueError(f"Unknown task: {self.task}")
        if self.ablation not in {"none", "no_yz_loss", "no_aux_policy"}:
            raise ValueError(f"Unknown ablation: {self.ablation}")
        if self.method != "disco" and self.ablation != "none":
            raise ValueError("Disco loss ablations require method=disco")
        for name in (
            "steps",
            "batch_size",
            "rollout_length",
            "feature_dim",
            "max_moves",
            "eval_every",
            "eval_episodes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.channels or min(self.channels) <= 0:
            raise ValueError("channels must be positive")
        if not 0 <= self.move_cost < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid reward cost or learning rate")
        if self.snapshot_every < 0:
            raise ValueError("snapshot_every must be zero (disabled) or positive learner updates")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path, **overrides):
        data = json.loads(Path(path).read_text())
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        data.update({k: v for k, v in overrides.items() if v is not None})
        if "channels" in data:
            data["channels"] = tuple(data["channels"])
        return cls(**data)
