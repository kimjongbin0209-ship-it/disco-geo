import csv
import json
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest

from disco_geo.config import TrainConfig
from disco_geo.data.bundle import RGBBundle
from disco_geo.evaluation.features import extract
from disco_geo.evaluation.probes import run_probes
from disco_geo.evaluation.report import create_report
from disco_geo.io import load_checkpoint, sha256, write_json
from disco_geo.training import runner
from disco_geo.training.benchmark import benchmark_rl
from scripts.run_snapshots import build_plan, execute, parser
from scripts.estimate_runtime import estimate


def config(dataset, output):
    return TrainConfig(
        dataset=str(dataset),
        output=str(output),
        method="actor_critic",
        channels=(8, 16),
        feature_dim=16,
        batch_size=2,
        rollout_length=3,
        steps=24,
        max_moves=4,
        eval_every=2,
        eval_episodes=4,
        snapshot_every=2,
    )


@pytest.mark.parametrize("crash_update", [1, 2])
def test_resume_restores_optimizer_rng_env_and_repairs_crash_log(
    dataset, tmp_path, monkeypatch, crash_update
):
    uninterrupted = config(dataset, tmp_path / "full")
    interrupted = replace(uninterrupted, output=str(tmp_path / "resume"))
    runner.train(uninterrupted)
    original = runner.save_checkpoint

    def crash_after_first_latest(path, data):
        original(path, data)
        if Path(path).name == "latest.pkl" and data["updates"] == crash_update:
            raise RuntimeError("simulated host stop after atomic checkpoint")

    monkeypatch.setattr(runner, "save_checkpoint", crash_after_first_latest)
    with pytest.raises(RuntimeError, match="simulated"):
        runner.train(interrupted)
    log = Path(interrupted.output) / "metrics.jsonl"
    with log.open("a") as f:
        f.write('{"updates":999,"loss":1}\n{"updates":')
    monkeypatch.setattr(runner, "save_checkpoint", original)
    runner.train(interrupted, resume=True)
    a = load_checkpoint(Path(uninterrupted.output) / "latest.pkl")
    b = load_checkpoint(Path(interrupted.output) / "latest.pkl")
    assert (a["updates"], a["transitions"]) == (b["updates"], b["transitions"])
    for key in ("learner", "actor_state", "env_state", "timestep", "rng"):
        for x, y in zip(jax.tree.leaves(a[key]), jax.tree.leaves(b[key])):
            np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len({row["updates"] for row in rows}) == len(rows)
    assert max(r["updates"] for r in rows) == b["updates"]
    with pytest.raises(ValueError, match="same model"):
        runner.train(replace(interrupted, learning_rate=0.001), resume=True)

    root = Path(interrupted.output)
    points = json.loads((root / "snapshots/manifest.json").read_text())["items"]
    assert points[0]["updates"] == 0 and points[-1]["updates"] == b["updates"]
    assert {p["updates"] for p in points} >= set(range(2, b["updates"] + 1, 2))
    last = root / "snapshots" / points[-1]["checkpoint"]
    small = load_checkpoint(last)
    assert small["checkpoint_kind"] == "frozen_encoder_snapshot"
    assert all(k.startswith("visual_encoder/") for k in small["learner"].params)
    assert last.stat().st_size < (root / "latest.pkl").stat().st_size
    extract(last, dataset, root / "snapshot.npy")
    extract(root / "latest.pkl", dataset, root / "latest.npy")
    np.testing.assert_allclose(np.load(root / "snapshot.npy"), np.load(root / "latest.npy"))
    probes = run_probes(
        dataset,
        root / "snapshot.npy",
        root / "val_probes.json",
        label_budget=12,
        evaluation_split="probe_val",
    )
    bundle = RGBBundle(dataset)
    assert set(probes["evaluation_scene_ids"]) == set(
        bundle.scene_ids[bundle.manifest["splits"]["probe_val"]]
    )
    assert not set(probes["evaluation_scene_ids"]) & set(
        bundle.scene_ids[bundle.manifest["splits"]["test"]]
    )
    assert (root / "val_probes.predictions.npz").exists()


def test_report_preserves_raw_values_and_exports_seed_stats_and_diagnostics(tmp_path):
    for seed, score in enumerate((0.2, 0.6)):
        root = tmp_path / "disco-none" / f"seed{seed}"
        write_json(root / "config.json", dict(method="disco", ablation="none", seed=seed))
        write_json(root / "policy.json", dict(success_rate=score, mean_return=score, mean_moves=2))
        write_json(
            root / "probes.json",
            dict(
                pose_mean_degrees=20,
                relative_mean_degrees=30,
                depth_rmse=0.2,
                depth_abs_rel=0.1,
                depth_delta_125=0.8,
            ),
        )
        (root / "metrics.jsonl").write_text(
            json.dumps(
                dict(
                    transitions=10,
                    updates=1,
                    success_rate=score,
                    mean_return=score,
                    mean_moves=2,
                    loss=0.5,
                    gradient_norm=0.3,
                    wall_seconds=1,
                )
            )
            + "\n"
        )
        point = root / "snapshots/iter0000001"
        write_json(point / "iteration.json", dict(updates=1, transitions=10))
        write_json(point / "probes.json", dict(evaluation_split="probe_val", depth_abs_rel=0.1))
        write_json(
            point / "generation/scene1/learned/evaluation/metrics.json",
            dict(
                scene=1,
                evaluation_split="probe_val",
                interpretation="diagnostic unregistered",
                mean_psnr=12,
                mean_foreground_psnr=10,
                mean_silhouette_iou=0.3,
            ),
        )
        np.savez(
            root / "probes.predictions.npz",
            depth_prediction=np.ones((2, 4, 4)),
            depth_target=np.ones((2, 4, 4)),
        )
    raw = tmp_path / "disco-none/seed0/metrics.jsonl"
    digest = sha256(raw)
    report = create_report(tmp_path)
    summary = json.loads((report / "summary.json").read_text())
    stats = summary["disco"]["metrics"]["success_rate"]
    assert stats["n"] == 2 and stats["mean"] == pytest.approx(0.4)
    assert stats["std_across_seeds"] == pytest.approx(np.std([0.2, 0.6], ddof=1))
    assert sha256(raw) == digest
    with (report / "learning_curves.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {"loss", "gradient_norm", "wall_seconds"} <= {r["metric"] for r in rows}
    with (report / "generation_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {r["interpretation"] for r in rows} == {"diagnostic unregistered"}
    for name in ("benchmark_comparison", "learning_curves", "snapshot_probes", "generation_curves"):
        assert (report / f"{name}.png").stat().st_size > 1000
        assert (report / f"{name}.pdf").stat().st_size > 1000


def test_snapshot_plan_uses_fixed_validation_inputs_and_counts_jobs(dataset, tmp_path):
    bundle = RGBBundle(dataset)
    run = tmp_path / "run"
    cache = tmp_path / "cache"
    write_json(run / "config.json", dict(dataset=str(dataset)))
    write_json(
        run / "snapshots/manifest.json",
        {
            "items": [
                dict(updates=i, checkpoint=f"iter{i:07d}/encoder.pkl") for i in (0, 1000, 2000)
            ]
        },
    )
    items = [
        dict(scene=s, scene_id=str(bundle.scene_ids[s]), view=0, split=split, image=f"{s}.png")
        for split in ("probe_val", "test")
        for s in bundle.manifest["splits"][split]
    ]
    write_json(
        cache / "manifest.json",
        dict(dataset_manifest_sha256=sha256(dataset / "manifest.json"), items=items),
    )
    args = parser().parse_args(
        [
            "--run",
            str(run),
            "--cache",
            str(cache),
            "--weights",
            "not-downloaded",
            "--iterations",
            "0",
            "1000",
            "--plan-only",
            "--modes",
            "learned",
            "zero",
        ]
    )
    plan = build_plan(args)
    assert plan["generation_jobs"] == 8 and plan["adapter_jobs"] == 2
    assert {i["split"] for i in plan["monitor_inputs"]} == {"probe_val"}
    assert not plan["test_identities_used"]
    args.iterations = [99]
    with pytest.raises(ValueError, match="must exist"):
        build_plan(args)


def test_runtime_measurement_reports_actual_platform_and_conditional_budget(dataset, tmp_path):
    cfg = config(dataset, tmp_path / "unused")
    result = benchmark_rl(cfg, tmp_path / "timing.json", updates=2, warmup=1)
    assert result["platform"] == jax.default_backend()
    assert result["transitions_per_second"] > 0
    assert result["target_rl_plus_eval_hours"] >= result["target_rl_compute_hours"]
    from argparse import Namespace

    profiles = []
    for method in ("actor_critic", "actor_critic_aux", "disco"):
        path = tmp_path / f"{method}.json"
        write_json(
            path,
            {
                "platform": "cpu",
                "devices": ["cpu"],
                "config": {"method": method},
                "target_rl_plus_eval_hours": 1,
            },
        )
        profiles.append(str(path))
    write_json(tmp_path / "adapter.json", {"wall_seconds": 120})
    write_json(tmp_path / "generation.json", {"wall_seconds": 60})
    args = Namespace(
        profiles=profiles,
        allow_cpu=False,
        seeds=3,
        ablations=True,
        snapshots_per_trained_run=5,
        items=2,
        modes=1,
        adapter_result=str(tmp_path / "adapter.json"),
        generation_metadata=str(tmp_path / "generation.json"),
        output=str(tmp_path / "estimate.json"),
    )
    with pytest.raises(ValueError, match="GPU measurements"):
        estimate(args)
    args.allow_cpu = True
    result = estimate(args)
    assert result["adapter_jobs"] == 78 and result["generation_jobs"] == 156
    assert result["measured_stage_subtotal_hours"] == pytest.approx(20.2)


def test_snapshot_jobs_resume_completed_adapter_and_keep_paired_validation_scene(
    dataset, tmp_path, monkeypatch
):
    # Exercise orchestration/transactional outputs, not pretrained GPU quality.
    import subprocess
    from disco_geo.generative.external import WEIGHTS

    run, cache, weights = (tmp_path / name for name in ("run", "cache", "weights"))
    bundle = RGBBundle(dataset)
    scene = bundle.manifest["splits"]["probe_val"][0]
    write_json(run / "config.json", dict(dataset=str(dataset)))
    write_json(
        run / "snapshots/manifest.json",
        {"items": [dict(updates=0, checkpoint="iter0000000/encoder.pkl")]},
    )
    point = run / "snapshots/iter0000000"
    point.mkdir()
    (point / "encoder.pkl").write_bytes(b"trusted test snapshot")
    write_json(
        cache / "manifest.json",
        dict(
            dataset_manifest_sha256=sha256(dataset / "manifest.json"),
            items=[
                dict(
                    scene=scene,
                    scene_id=str(bundle.scene_ids[scene]),
                    view=0,
                    split="probe_val",
                    image="source.png",
                )
            ],
        ),
    )
    for file in WEIGHTS.values():
        path = weights / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test weight provenance")
    calls = []
    fail = True

    def mock_call(command, check):
        nonlocal fail
        action = command[3]
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        if action == "extract":
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, np.zeros((20, 6, 16)))
            write_json(output.with_suffix(".json"), {"frozen": True})
        elif action == "probe":
            write_json(output, {"evaluation_split": "probe_val"})
        elif action == "train-adapter":
            write_json(output / "result.json", {"wall_seconds": 1})
            (output / "best.pt").write_bytes(b"adapter")
        elif action == "generate":
            assert command[command.index("--scene") + 1] == str(scene)
            assert command[command.index("--view") + 1] == "0"
            assert command[command.index("--exports") + 1 : command.index("--exports") + 3] == [
                "ply",
                "npz",
            ]
            if fail:
                fail = False
                raise subprocess.CalledProcessError(1, command)
            write_json(output / "generation.json", {"mode": "learned"})
        elif action == "evaluate-generation":
            assert command[command.index("--evaluation-split") + 1] == "probe_val"
            write_json(output / "metrics.json", {"scene": scene})
        else:
            raise AssertionError(action)

    monkeypatch.setattr("scripts.run_snapshots.subprocess.run", mock_call)
    args = parser().parse_args(
        ["--run", str(run), "--cache", str(cache), "--weights", str(weights), "--items", "1"]
    )
    with pytest.raises(subprocess.CalledProcessError):
        execute(args)
    assert (point / "adapter/result.json").exists()
    assert not list(point.rglob(".stage-*"))
    args.resume = True
    execute(args)
    assert sum(c[3] == "train-adapter" for c in calls) == 1
    assert sum(c[3] == "generate" for c in calls) == 2
    assert (point / "monitor_completed.json").exists()
    count = len(calls)
    execute(args)
    assert len(calls) == count
