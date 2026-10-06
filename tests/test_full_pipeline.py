import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from disco_geo.data.bundle import RGBBundle
from disco_geo.evaluation.transfer_report import create_transfer_report
from disco_geo.generative.external import WEIGHTS
from disco_geo.io import sha256, write_json
from scripts.run_transfer import execute, parser


def test_final_paired_transfer_resumes_and_compares_seed_means(dataset, tmp_path, monkeypatch):
    # Tests the pipeline's artifact/identity/statistics contract; no pretrained GPU claim.
    bundle = RGBBundle(dataset)
    cache, weights, output = (tmp_path / name for name in ("cache", "weights", "transfer"))
    cache.mkdir()
    items = []
    for scene in bundle.manifest["splits"]["test"]:
        Image.fromarray(bundle.images[scene, 0]).save(cache / f"{scene}.png")
        items.append(
            dict(
                scene=scene,
                scene_id=str(bundle.scene_ids[scene]),
                view=0,
                image=f"{scene}.png",
                split="test",
            )
        )
    write_json(
        cache / "manifest.json",
        dict(dataset_manifest_sha256=sha256(dataset / "manifest.json"), items=items),
    )
    for file in WEIGHTS.values():
        path = weights / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test provenance")
    specs = []
    for seed in (0, 1):
        file = tmp_path / f"disco{seed}.npy"
        np.save(file, np.full((20, 6, 8), seed, np.float32))
        write_json(
            file.with_suffix(".json"),
            dict(
                frozen=True,
                method="disco",
                ablation="none",
                seed=seed,
                dataset_manifest_sha256=sha256(dataset / "manifest.json"),
                scene_ids=bundle.scene_ids.tolist(),
            ),
        )
        specs.append(f"disco_seed{seed}={file}")
    args = parser().parse_args(
        [
            "--cache",
            str(cache),
            "--weights",
            str(weights),
            "--features",
            *specs,
            "--output",
            str(output),
            "--test-items",
            "2",
            "--diagnostic-dataset",
            str(dataset),
        ]
    )
    calls = []
    fail = True

    def value(cmd, option):
        return cmd[cmd.index(option) + 1]

    def mock_call(cmd, check):
        nonlocal fail
        action = cmd[3]
        calls.append(cmd)
        target = Path(value(cmd, "--output"))
        if action == "train-adapter":
            write_json(target / "result.json", {"wall_seconds": 1})
            (target / "best.pt").write_bytes(b"adapter")
        elif action == "generate":
            mode = value(cmd, "--mode")
            assert value(cmd, "--seed") == "42"
            if mode == "zero" and fail:
                fail = False
                raise subprocess.CalledProcessError(1, cmd)
            scene = (
                int(Path(value(cmd, "--image")).stem)
                if mode == "native"
                else int(value(cmd, "--scene"))
            )
            seed = None if mode == "native" else int(Path(value(cmd, "--features")).stem[-1])
            write_json(
                target / "generation.json", dict(mode=mode, scene=scene, seed=seed, wall_seconds=1)
            )
            (target / "object.ply").write_bytes(b"ply\n")
        elif action == "evaluate-generation":
            generated = json.loads(
                (Path(value(cmd, "--generation")) / "generation.json").read_text()
            )
            index = [i["scene"] for i in items].index(generated["scene"])
            score = (
                22 + 2 * index
                if generated["mode"] == "native"
                else 10 + 10 * generated["seed"] + 20 * index
            )
            score -= {"native": 0, "learned": 0, "zero": 2, "shuffled": 4}[generated["mode"]]
            views = [
                dict(view=v, psnr=score, foreground_psnr=score - 1, silhouette_iou=0.3)
                for v in range(1, 6)
            ]
            write_json(
                target / "metrics.json",
                dict(
                    scene=generated["scene"],
                    evaluation_split="test",
                    source_view_excluded=0,
                    interpretation="diagnostic unregistered",
                    views=views,
                    mean_psnr=score,
                    mean_foreground_psnr=score - 1,
                    mean_silhouette_iou=0.3,
                ),
            )
            for view in views:
                Image.new("RGB", (16, 16), (100, int(score), 150)).save(
                    target / f"view_{view['view']:03d}.png"
                )
        else:
            raise AssertionError(action)

    monkeypatch.setattr("scripts.run_transfer.subprocess.run", mock_call)
    with pytest.raises(subprocess.CalledProcessError):
        execute(args)
    assert not list(output.rglob(".stage-*"))
    args.resume = True
    execute(args)
    completed = json.loads((output / "completed.json").read_text())
    assert completed["generation_count"] == 14
    assert sum(c[3] == "train-adapter" for c in calls) == 2
    assert sum(c[3] == "generate" for c in calls) == 15  # one failed attempt
    assert sum(c[3] == "generate" and value(c, "--mode") == "native" for c in calls) == 2
    count = len(calls)
    execute(args)
    assert len(calls) == count
    monkeypatch.undo()
    report = create_transfer_report(output)
    summary = json.loads((report / "summary.json").read_text())
    learned = next(r for r in summary["groups"] if r["mode"] == "learned")
    assert learned["metrics"]["mean_psnr"]["mean"] == 25
    assert learned["metrics"]["mean_psnr"]["std_across_seeds"] == pytest.approx(
        np.std([20, 30], ddof=1)
    )
    native = next(r for r in summary["groups"] if r["mode"] == "native")
    assert native["n_seeds"] == 0 and native["metrics"]["mean_psnr"]["std_across_seeds"] is None
    assert summary["interpretation"] == "UNREGISTERED diagnostics"
    assert len(json.loads((report / "per_image.json").read_text())) == 14
    for file in (
        "per_image.csv",
        "per_view.csv",
        "per_seed.csv",
        "generation_comparison.png",
        "generation_comparison.pdf",
    ):
        assert (report / file).stat().st_size > 100
    assert len(list(report.glob("qualitative_*.png"))) == 4
    args.adapter_epochs += 1
    with pytest.raises(ValueError, match="identical settings"):
        execute(args)


def test_shell_dry_run_connects_all_variants_final_controls_and_reports_without_writes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/run_full_experiment.sh"
    output = tmp_path / "all results"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--python",
            sys.executable,
            "--output",
            str(output),
            "--seeds",
            "0",
            "2",
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert not output.exists()
    assert "scripts/run_study.py" in result.stdout and "--ablations" in result.stdout
    assert "--snapshot-cache" in result.stdout
    assert "scripts/run_transfer.py" in result.stdout
    for seed in (0, 2):
        for variant in (
            "untrained-none",
            "actor_critic-none",
            "actor_critic_aux-none",
            "disco-none",
            "disco-no_yz_loss",
            "disco-no_aux_policy",
        ):
            assert f"{variant}_seed{seed}=" in result.stdout
    assert "report-transfer" in result.stdout and "--adapter-device cuda" in result.stdout
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--python",
            sys.executable,
            "--output",
            str(output),
            "--dry-run",
            "--resume",
            "--snapshot-every",
            "0",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--snapshot-cache" not in result.stdout
    assert result.stdout.count("--resume") == 2
    result = subprocess.run(
        ["bash", str(script), "--seeds", "0", "0", "--dry-run"], capture_output=True, text=True
    )
    assert result.returncode == 2 and "Duplicate seed" in result.stderr
