"""Summarize the local smoke pipeline and offline teacher check into a tracked report.

Inputs live under runs/ and datasets/ (gitignored). The output records what was
executed on the real benchmark and what it showed, explicitly not as research results.
"""

import json
from pathlib import Path

from disco_spatial.io import sha256, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    smoke = root / "runs/smoke"
    runs = {}
    for completed in sorted(smoke.glob("*/completed.json")):
        result = json.loads(completed.read_text())
        rows = [json.loads(line) for line in (completed.parent / "metrics.jsonl").read_text().splitlines()]
        runs[completed.parent.name] = {
            "updates": result["updates"], "transitions": result["transitions"],
            "wall_seconds": round(result["wall_seconds"], 1),
            "learner_network_steps": result.get("learner_network_steps"),
            "train_reward_events": sum(row["reward_events_this_chunk"] for row in rows),
            "final_validation_return": rows[-1].get("validation", {}).get("mean_return"),
            "teacher_validation_return": rows[-1].get("teacher_validation", {}).get("mean_return"),
            "replay_updates": sum(row["replay_used"] for row in rows),
            "self_imitation_updates": sum("self_imitation_loss" in row for row in rows),
            "teacher_updates": sum("teacher_nll" in row for row in rows),
            "distillation_updates": result.get("distillation_updates", 0),
            "teacher_gate": result.get("teacher_gate"),
        }
    abilities = {}
    for path in sorted((smoke / "abilities").glob("*.json")):
        result = json.loads(path.read_text())
        families = result["families"]
        abilities[path.stem] = {
            representation: {
                "seen_hidden_rmse": families["spatial_memory"][representation][
                    "seen_hidden_target_location_rmse"],
                "seen_hidden_train_mean_rmse": families["spatial_memory"][representation][
                    "seen_hidden_train_mean_rmse"],
                "known_wall_balanced_iou": families["map_topology"][representation][
                    "known_wall_balanced_iou"],
                "origin_position_rmse": families["self_localization"][representation][
                    "origin_position_rmse"],
                **({"place_recall_at_1": families["loop_closure"][representation][
                        "place_retrieval_recall_at_1"],
                    "place_chance": families["loop_closure"][representation]["chance_recall_at_1"],
                    "traversable_ap": families["local_geometry"][representation][
                        "traversable_forward_average_precision"]}
                   if representation in families["loop_closure"] else {}),
            } for representation in families["spatial_memory"]}
    navigation = {path.stem: {name: {"mean_return": value["mean_return"],
                                     "n_matched_pairs": value["n_matched_pairs"]}
                              for name, value in json.loads(path.read_text())["interventions"].items()}
                  for path in sorted((smoke / "navigation").glob("*.json"))}
    collections = {}
    for manifest in sorted((root / "datasets/smoke").glob("*/manifest.json")):
        data = json.loads(manifest.read_text())
        collections[data["partition"]] = {
            "episodes": len(data["entries"]),
            "layouts": sorted({entry["layout_id"][:12] for entry in data["entries"]}),
            "oracle_floor_pixels_in_walls_fraction": [
                entry["oracle_floor_pixels_in_walls_fraction"] for entry in data["entries"]],
        }
    teacher = {}
    for name in ("offline.json", "variants.json"):
        path = root / "runs/teacher_validation" / name
        if path.exists():
            teacher[name] = json.loads(path.read_text())
    report = {
        "date": "2026-10-28", "status": "smoke_pipeline_on_real_benchmark_not_research_results",
        "scientific_result": False, "training_runs": runs, "collections": collections,
        "abilities": abilities, "navigation": navigation, "offline_teacher_validation": teacher,
        "observations": [
            "2,048-step policies earned no training reward; self-imitation never found a reward chunk "
            "and the evidence gate stayed closed, so success paths are covered by unit tests only.",
            "disco-sil equals disco-fresh exactly because no auxiliary batch was formed: shared "
            "initialization and determinism hold.",
            "Untrained init features already exceed chance on place retrieval and traversability; "
            "those families must be compared against the init control, not chance.",
            "Offline, on a fixed observation-independent explore policy with sparse reward, no "
            "teacher variant reached positive held-out information gain; the gate would stay closed.",
        ],
        "provenance_note": "Each smoke run's provenance.json holds the exact source hashes it ran "
                           "with; the hashes below describe the tree when this summary was written.",
        "source_tree_at_summary_sha256": {str(path.relative_to(root)): sha256(path)
                                          for path in sorted((root / "disco_spatial").rglob("*.py"))},
    }
    target = root / "reports/development/smoke_summary.json"
    write_json(target, report)
    print(json.dumps({"report": str(target), "runs": len(runs), "abilities": len(abilities)}))


if __name__ == "__main__":
    main()
