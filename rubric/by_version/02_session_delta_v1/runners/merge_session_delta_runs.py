from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUBRIC_DIR = ROOT / "rubric"
OUTPUTS_DIR = RUBRIC_DIR / "outputs"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for model_result in results:
        for session in model_result.get("sessions", []) or []:
            row = {
                "add_model": model_result["add_model"],
                "workspace_dir": model_result["workspace_dir"],
                "conversation_idx": session["conversation_idx"],
                "session_index": session["session_index"],
                "delta_count": session.get("delta_count"),
                "session_score": session["session_score"],
            }
            for rubric in session.get("rubrics", []):
                row[f"rubric_{rubric['id']}"] = rubric["score"]
            rows.append(row)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def merge_model_results(results_list: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, Any]] = {}
    for results in results_list:
        for model_result in results:
            model_name = str(model_result.get("add_model") or "")
            if not model_name:
                continue
            current = by_model.get(model_name)
            if current is None:
                current = {
                    "add_model": model_result["add_model"],
                    "workspace_dir": model_result["workspace_dir"],
                    "session_count": 0,
                    "model_score": 0.0,
                    "sessions": [],
                }
                by_model[model_name] = current
            session_map = {
                (int(item["conversation_idx"]), int(item["session_index"])): item
                for item in current.get("sessions", [])
            }
            for item in model_result.get("sessions", []) or []:
                session_map[(int(item["conversation_idx"]), int(item["session_index"]))] = item
            merged_sessions = sorted(
                session_map.values(),
                key=lambda item: (int(item["conversation_idx"]), int(item["session_index"])),
            )
            current["sessions"] = merged_sessions
            current["session_count"] = len(merged_sessions)
            current["workspace_dir"] = model_result.get("workspace_dir") or current["workspace_dir"]
            current["model_score"] = (
                round(sum(float(item["session_score"]) for item in merged_sessions) / len(merged_sessions), 4)
                if merged_sessions
                else 0.0
            )
    return list(by_model.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge session-delta rubric runs")
    parser.add_argument("--inputs", type=str, required=True, help="Comma-separated run directories or rubric_scores.json files")
    parser.add_argument("--output-dir", type=str, required=True, help="Merged output directory")
    args = parser.parse_args()

    input_paths = [item.strip() for item in args.inputs.split(",") if item.strip()]
    runs: list[dict[str, Any]] = []
    for item in input_paths:
        path = Path(item).expanduser()
        if not path.is_absolute():
            candidate = (OUTPUTS_DIR / path)
            path = candidate if candidate.exists() else path.resolve()
        json_path = path if path.name.endswith(".json") else (path / "rubric_scores.json")
        payload = load_json(json_path)
        runs.append(payload)

    merged_results = merge_model_results([run.get("results") or [] for run in runs])
    evaluator_model = next((run.get("evaluator_model") for run in runs if run.get("evaluator_model")), "")
    missing: list[dict[str, Any]] = []
    for run in runs:
        missing.extend(run.get("missing") or [])
    final_scores = [float(item["session_score"]) for result in merged_results for item in result.get("sessions", [])]
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        candidate = OUTPUTS_DIR / output_dir
        output_dir = candidate if str(output_dir).startswith("session_") else output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_payload = {
        "run_name": output_dir.name,
        "evaluator_model": evaluator_model,
        "results": merged_results,
        "missing": missing,
        "final_score": round(sum(final_scores) / len(final_scores), 4) if final_scores else 0.0,
        "source_runs": input_paths,
    }
    write_json(output_dir / "rubric_scores.json", merged_payload)
    write_csv(output_dir / "rubric_scores.csv", merged_results)
    write_json(
        output_dir / "summary.json",
        {
            "evaluator_model": evaluator_model,
            "final_score": merged_payload["final_score"],
            "models": [
                {
                    "add_model": result["add_model"],
                    "workspace_dir": result["workspace_dir"],
                    "session_count": result["session_count"],
                    "model_score": result["model_score"],
                }
                for result in merged_results
            ],
            "missing": missing,
            "source_runs": input_paths,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "final_score": merged_payload["final_score"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
