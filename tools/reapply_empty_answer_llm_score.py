"""批量重算 eval：predicted_answer/response 为空时 llm_score 一律为 0。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from core.infra.flat_export import write_flattened_eval_records_from_file
from core.infra.scoring import load_and_summarize, reapply_empty_answer_llm_scores

WORKSPACES = PIPELINE_DIR / "workspaces"


def _score_summary_path(eval_path: Path) -> Path | None:
    """eval 文件同目录下的 score_summary_<stem>.json。"""
    stem = eval_path.stem  # evaluation_metrics_answerhistory
    if not stem.startswith("evaluation_metrics_"):
        return None
    mode = stem.removeprefix("evaluation_metrics_")
    candidate = eval_path.with_name(f"score_summary_{mode}.json")
    return candidate if candidate.exists() else None


def reapply_eval_file(eval_path: Path) -> dict[str, float | int]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {eval_path}")
    updated = reapply_empty_answer_llm_scores(payload)
    eval_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    write_flattened_eval_records_from_file(input_path=eval_path)
    summary_path = _score_summary_path(eval_path)
    if summary_path is not None:
        summary = load_and_summarize(eval_path)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    empty = sum(1 for r in updated if not str(r.get("predicted_answer") or r.get("response") or "").strip())
    overridden = sum(
        1
        for r in updated
        if not str(r.get("predicted_answer") or r.get("response") or "").strip()
        and r.get("llm_score_judge") == 1
    )
    llm_mean = load_and_summarize(eval_path)["overall"].get("llm_score", 0.0)
    return {
        "records": len(updated),
        "empty_answers": empty,
        "empty_overridden_from_1": overridden,
        "llm_score_mean": float(llm_mean),
    }


def reapply_llm_all_report(report_path: Path, eval_path: Path) -> None:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    eval_rows = {
        (int(x["conversation_idx"]), int(x["qa_index"])): x
        for x in json.loads(eval_path.read_text(encoding="utf-8"))
    }
    records = payload.get("records") or []
    for record in records:
        key = (int(record["conversation_idx"]), int(record["qa_index"]))
        ev = eval_rows.get(key, {})
        llm_score = ev.get("llm_score")
        record["llm_score"] = llm_score
        record["is_correct"] = bool(float(llm_score or 0) >= 1.0) if llm_score is not None else None
        if ev.get("llm_score_judge") is not None:
            record["llm_score_judge"] = ev["llm_score_judge"]
    meta = payload.setdefault("_meta", {})
    meta["scoring_rule"] = "predicted_answer 为空时 llm_score 一律为 0（不再采用 Judge 对空答案的 CORRECT 判定）。"
    meta["correct_count"] = sum(1 for r in records if r.get("is_correct") is True)
    meta["wrong_count"] = sum(1 for r in records if r.get("is_correct") is False)
    meta["accuracy_llm_judge"] = round(meta["correct_count"] / len(records), 6) if records else 0
    field_desc = meta.setdefault("field_descriptions", {})
    field_desc["is_correct"] = "是否正确：llm_score=1 为正确，llm_score=0 为错误；空答案一律为 0。"
    field_desc["llm_score"] = "LLM Judge 分数（已应用空答案规则）：1.0=CORRECT，0.0=WRONG；predicted_answer 为空时强制为 0。"
    field_desc["llm_score_judge"] = "Judge 原始分数：应用空答案规则前 LLM Judge 的 llm_score（仅空答案记录有此字段）。"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    eval_files = sorted(
        path
        for path in WORKSPACES.rglob("evaluation_metrics_*.json")
        if not path.name.endswith("_flattened.json")
    )
    print(f"processing {len(eval_files)} eval file(s) ...", flush=True)
    for eval_path in eval_files:
        stats = reapply_eval_file(eval_path)
        print(
            f"  {eval_path.relative_to(PIPELINE_DIR)} "
            f"n={stats['records']} empty={stats['empty_answers']} "
            f"override_1->0={stats['empty_overridden_from_1']} "
            f"llm_mean={stats['llm_score_mean']:.4f}",
            flush=True,
        )

    report = WORKSPACES / "v3" / "minimax" / "gemini" / "add_run01" / "reports" / "gemini_run01_qa_search_answer_llm_all.json"
    eval_llm = WORKSPACES / "v3" / "minimax" / "gemini" / "add_run01" / "llm" / "evaluation_metrics_answerhistory.json"
    if report.exists() and eval_llm.exists():
        reapply_llm_all_report(report, eval_llm)
        print(f"  updated report: {report.relative_to(PIPELINE_DIR)}", flush=True)


if __name__ == "__main__":
    main()
