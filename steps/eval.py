"""eval 步骤：对 answer 输出打分。

每条记录可能对多个参考答案候选（answer / reference_answer）取 max：
  - f1 / bleu：词级重叠
  - llm：异步调用 EVALUATOR_* 配置的裁判模型（prompts/llm_judge_v5.txt，CORRECT/WRONG → 1/0）

输出 evaluation_metrics_answer{mode}.json，并生成 flattened 便于表格分析。
"""

from __future__ import annotations

import asyncio
import ast
import json
import re
from pathlib import Path
from typing import Any, Sequence

from lib.metrics.bleu_f1 import compute_bleu1, compute_token_f1
from lib.metrics.llm_judge import configure_evaluator, evaluate_llm_judge, shutdown_evaluator
from lib.progress import ProgressBar

SUPPORTED_METRICS = frozenset({"llm", "f1", "bleu"})


def _normalize_prediction_text(text: str) -> str:
    """清洗模型输出；若形如 Python list 字符串则展平为逗号分隔。"""
    cleaned = str(text or "").strip()
    if not cleaned:
        return cleaned
    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            parsed = ast.literal_eval(cleaned)
            if isinstance(parsed, list):
                return ", ".join(str(item).strip() for item in parsed if str(item).strip())
        except (SyntaxError, ValueError):
            pass
    return cleaned


def _normalize_answer_candidates(answer: Any, reference_answer: Any) -> list[str]:
    """合并 answer 与 reference_answer 为去重后的参考答案候选列表。"""
    candidates: list[str] = []
    for raw in (answer, reference_answer):
        if isinstance(raw, list):
            candidates.extend(str(item).strip() for item in raw if str(item).strip())
        else:
            text = str(raw or "").strip()
            if text:
                candidates.append(text)
    if not candidates:
        return [""]
    seen: set[str] = set()
    unique: list[str] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


async def evaluate_records(
    *,
    input_path: str | Path,
    output_path: str | Path | None = None,
    concurrency: int = 2,
    metrics: Sequence[str] | None = None,
    evaluator_model: str | None = None,
    evaluator_base_url: str | None = None,
    evaluator_api_key: str | None = None,
) -> list[dict[str, Any]]:
    """并发评测 JSON 列表中每条记录，可选写入 output_path。"""
    enabled = {str(m).strip().lower() for m in (metrics or ["llm", "f1", "bleu"])} & SUPPORTED_METRICS
    if not enabled:
        enabled = set(SUPPORTED_METRICS)

    input_file = Path(input_path)
    payload = json.loads(input_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("eval step expects a JSON list")

    if "llm" in enabled:
        configure_evaluator(
            model=evaluator_model,
            base_url=evaluator_base_url,
            api_key=evaluator_api_key,
        )

    print(
        f"[eval] records={len(payload)} metrics={sorted(enabled)} concurrency={concurrency}",
        flush=True,
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))
    progress = ProgressBar("eval", total=len(payload) or None, unit="qa")
    progress_lock = asyncio.Lock()
    results: list[dict[str, Any] | None] = [None] * len(payload)

    async def _evaluate_record(index: int, record: dict[str, Any]) -> None:
        """信号量包装的单条评测任务。"""
        async with semaphore:
            async with progress_lock:
                progress.set_description(
                    f"eval conv{record.get('conversation_idx')} qa{record.get('qa_index')}"
                )
            results[index] = await _evaluate_one(record, enabled_metrics=enabled)
            async with progress_lock:
                progress.update(1)

    try:
        await asyncio.gather(
            *[_evaluate_record(index, record) for index, record in enumerate(payload)]
        )
    finally:
        progress.close()
        if "llm" in enabled:
            shutdown_evaluator()

    evaluated = [item for item in results if item is not None]

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(evaluated, ensure_ascii=False, indent=2), encoding="utf-8")
    return evaluated


async def _evaluate_one(record: dict[str, Any], *, enabled_metrics: set[str]) -> dict[str, Any]:
    """对一条 QA 在所有参考答案候选上取各指标最大值。"""
    question = str(record.get("question") or "")
    prediction = _normalize_prediction_text(
        str(record.get("predicted_answer") or record.get("response") or "")
    )
    candidates = _normalize_answer_candidates(
        record.get("answer"),
        record.get("reference_answer", record.get("answer", "")),
    )

    best_answer = ""
    max_f1 = 0.0
    max_bleu = 0.0
    best_llm_score = -1.0
    best_llm_reason = ""

    for candidate in candidates:
        if "f1" in enabled_metrics:
            max_f1 = max(max_f1, compute_token_f1(prediction, candidate))
        if "bleu" in enabled_metrics:
            max_bleu = max(max_bleu, compute_bleu1(prediction, candidate))
        if "llm" in enabled_metrics:
            llm_score, llm_reason = await evaluate_llm_judge(question, candidate, prediction)
            if llm_score > best_llm_score:
                best_llm_score = llm_score
                best_llm_reason = llm_reason
                best_answer = candidate

    out = dict(record)
    out["response"] = prediction
    if "f1" in enabled_metrics:
        out["f1_score"] = max_f1
    if "bleu" in enabled_metrics:
        out["bleu_score"] = max_bleu
    if "llm" in enabled_metrics:
        out["llm_score"] = max(0.0, best_llm_score)
        out["llm_reason"] = best_llm_reason
    if best_answer:
        out["scoring_answer"] = best_answer
    return out
