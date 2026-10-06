"""A low-rank predictor of native conditioning tokens from frozen RL features."""

import torch
from torch import nn


class ConditioningAdapter(nn.Module):
    def __init__(
        self,
        feature_dim=256,
        token_count=4101,
        channels1=1280,
        channels2=128,
        rank=8,
        hidden_dim=256,
    ):
        super().__init__()
        self.settings = dict(
            feature_dim=feature_dim,
            token_count=token_count,
            channels1=channels1,
            channels2=channels2,
            rank=rank,
            hidden_dim=hidden_dim,
        )
        total = channels1 + channels2
        self.tokens = nn.Parameter(torch.randn(token_count, rank) / rank**0.5)
        self.predict = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, rank * total)
        )
        self.bias = nn.Parameter(torch.zeros(token_count, total))

    def forward(self, features, indices=None):
        # Detach even if the caller accidentally supplies a live encoder tensor.
        code = self.predict(features.detach()).reshape(
            -1, self.settings["rank"], self.settings["channels1"] + self.settings["channels2"]
        )
        basis = self.tokens if indices is None else self.tokens[indices]
        bias = self.bias if indices is None else self.bias[indices]
        predicted = torch.einsum("lr,brc->blc", basis, code) + bias
        a = self.settings["channels1"]
        return {"feature1": predicted[..., :a], "feature2": predicted[..., a:]}


def conditioning_loss(predicted, teacher):
    """One ordinary MSE on concatenated native image-conditioning features."""
    actual = torch.cat([predicted[k] for k in ("feature1", "feature2")], -1)
    target = torch.cat([teacher[k].detach() for k in ("feature1", "feature2")], -1)
    return torch.nn.functional.mse_loss(actual, target)
