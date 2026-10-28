"""Validate the recorded tiny integration run; never call it a research result.

Run pytest with --junitxml=reports/development/tests.xml first, then execute
this script from the repository root after the README's integration pipeline.
No model training, rendering or external network request is performed here.
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from disco_spatial.io import sha256, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    tests_path = root / "reports/development/tests.xml"
    suites = ET.parse(tests_path).getroot()
    cases = suites.findall(".//testcase")
    if not cases or suites.findall(".//failure") or suites.findall(".//error"):
        raise ValueError("The recorded unit suite did not pass")
    doctor_path = root / "runs/environment/local_cpu.json"
    doctor = json.loads(doctor_path.read_text())
    if (not doctor["renderer_tested"] or doctor["rgb_shape"] != [64, 64, 3]
            or doctor["spatial_rgb_range"] <= 0 or doctor["distinct_rgb_frames"] < 2):
        raise ValueError("Real benchmark RGB/reset/step verification is incomplete")
    outputs = {}
    sources = {str(doctor_path.relative_to(root)): sha256(doctor_path),
               str(tests_path.relative_to(root)): sha256(tests_path)}
    for recipe in ("fresh", "replay"):
        directory = root / f"runs/integration/real-disco-{recipe}"
        path = directory / "completed.json"
        completion = json.loads(path.read_text())
        rows = [json.loads(row) for row in (directory / "metrics.jsonl").read_text().splitlines()]
        if (completion["transitions"] != 8 or completion["updates"] != 2
                or completion["scientific_result"] or completion["sdpo_teacher_trained"]):
            raise ValueError("Expected only the explicitly tiny integration run")
        if sha256(directory / "latest.pkl") != completion["latest_checkpoint_sha256"]:
            raise ValueError("Integration checkpoint checksum changed")
        if recipe == "replay" and not any(row["replay_used"] for row in rows):
            raise ValueError("Recent replay was never used")
        outputs[recipe] = {"completion": completion,
                           "replay_updates": sum(row["replay_used"] for row in rows),
                           "all_losses_finite": bool(np.isfinite([row["loss"] for row in rows]).all())}
        sources[str(path.relative_to(root))] = sha256(path)
    common = []
    fingerprints = []
    for partition in ("probe_train", "probe_validation", "probe_final_test"):
        directory = root / "datasets/integration" / partition
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["entries"][0]
        with np.load(directory / entry["trajectory"], allow_pickle=False) as data:
            frames = data["rgb"]
            distinct = len({frame.tobytes() for frame in frames})
            if frames.shape[1:3] != (64, 64) or distinct < 2:
                raise ValueError("Common-history RGB is constant or incorrectly sized")
        fingerprints.append(entry["layout_id"])
        common.append({"partition": partition, "steps": entry["steps"],
                       "distinct_rgb_frames": distinct, "layout_id": entry["layout_id"]})
        sources[str(manifest_path.relative_to(root))] = sha256(manifest_path)
    if len(set(fingerprints)) != 3:
        raise ValueError("Integration probe partitions overlap in geometry")
    probe_path = root / "runs/integration/probe_location.json"
    probe = json.loads(probe_path.read_text())
    if probe["primary_endpoint"] or probe["confidence_interval"] is not None:
        raise ValueError("Generic tiny readout must not be presented as primary evidence")
    sources[str(probe_path.relative_to(root))] = sha256(probe_path)
    report = {
        "date": "2026-10-28", "status": "development_integration_passed",
        "scientific_result": False, "unit_tests_passed": len(cases),
        "runtime": doctor, "tiny_official_environment_runs": outputs,
        "common_history_checks": common,
        "generic_readout_pipeline_completed": True,
        "primary_ability_suite_completed": False,
        "teacher_implemented_or_validated": False,
        "gpu_benchmark_performed": False,
        "development_consumed_seeds_not_a_locked_confirmatory_bank": [0, 1000000, 3000000, 4000000, 5000000],
        "invalid_32px_runs_quarantined": "runs/debug-invalid-32px",
        "sources_sha256": sources,
        "source_tree_at_verification_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in sorted((root / "disco_spatial").rglob("*.py"))
        },
    }
    target = root / "reports/development/verification.json"
    write_json(target, report)
    print(json.dumps({"report": str(target), "unit_tests_passed": len(cases),
                      "actual_64px_pipeline": "passed", "scientific_result": False}))


if __name__ == "__main__":
    main()
