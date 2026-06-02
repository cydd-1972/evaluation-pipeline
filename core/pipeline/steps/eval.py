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
import os
import re
from pathlib import Path
from typing import Any, Sequence

from core.infra.checkpoint import has_eval_scores, load_json_list, write_json_list
from core.infra.env import load_runtime_env
from core.metrics.bleu_f1 import compute_bleu1, compute_token_f1
from core.infra.api_failure_budget import ApiFailureBudgetExceeded, reset_eval_budget
from core.metrics.llm_judge import (
    calibrate_evaluator_tpm_from_records,
    configure_evaluator,
    evaluate_llm_judge,
    evaluator_key_count,
    recommended_eval_concurrency,
    shutdown_evaluator,
)
from core.infra.progress import ProgressBar
from core.infra.scoring import apply_empty_answer_llm_score_rule

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
    prefer_evaluator_slots: bool = True,
    tpm_calibrate_sample_size: int = 32,
    progress_label: str | None = None,
) -> list[dict[str, Any]]:
    """并发评测 JSON 列表中每条记录，可选写入 output_path。"""
    enabled = {str(m).strip().lower() for m in (metrics or ["llm", "f1", "bleu"])} & SUPPORTED_METRICS
    if not enabled:
        enabled = set(SUPPORTED_METRICS)

    input_file = Path(input_path)
    payload = json.loads(input_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("eval step expects a JSON list")

    key_count = 1
    if "llm" in enabled:
        reset_eval_budget()
        base_concurrency = concurrency
        key_count = configure_evaluator(
            model=evaluator_model,
            base_url=evaluator_base_url if not prefer_evaluator_slots else None,
            api_key=evaluator_api_key if not prefer_evaluator_slots else None,
            prefer_slots=prefer_evaluator_slots,
        )
        token_stats = calibrate_evaluator_tpm_from_records(
            payload,
            sample_size=tpm_calibrate_sample_size,
        )
        if int(token_stats.get("count") or 0) > 0:
            tpm_limit = int(os.getenv("EVALUATOR_TPM_LIMIT", "40000") or 40000)
            est = int(os.getenv("EVALUATOR_EST_INPUT_TOKENS", "5000") or 5000)
            per_min = max(1, int(tpm_limit * 0.85 / est))
            print(
                f"[eval] judge prompt tokens (sample n={token_stats['count']}): "
                f"min={token_stats['min']} p50={token_stats['p50']} "
                f"p90={token_stats['p90']} max={token_stats['max']} mean={token_stats['mean']} "
                f"→ TPM est={est}/req (~{per_min} req/min)",
                flush=True,
            )
        concurrency = recommended_eval_concurrency(base_concurrency=base_concurrency, key_count=key_count)
        print(
            f"[eval] effective concurrency={concurrency} (keys={key_count}, base={base_concurrency})",
            flush=True,
        )

    output_file = Path(output_path) if output_path is not None else None
    results: list[dict[str, Any] | None] = [None] * len(payload)
    resumed = 0
    if output_file is not None and output_file.exists():
        existing = load_json_list(output_file)
        for index, item in enumerate(existing):
            if index < len(results) and has_eval_scores(item, enabled):
                results[index] = item
                resumed += 1
    pending = sum(1 for item in results if item is None)
    print(
        f"[eval] records={len(payload)} metrics={sorted(enabled)} concurrency={concurrency} "
        f"resumed={resumed} pending={pending}",
        flush=True,
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))
    progress = ProgressBar("eval", total=len(payload) or None, unit="qa", label=progress_label)
    if resumed:
        progress.update(resumed)
    progress_lock = asyncio.Lock()
    write_lock = asyncio.Lock()
    completed_since_save = 0

    async def _flush_partial() -> None:
        if output_file is None:
            return
        snapshot = [item if item is not None else payload[i] for i, item in enumerate(results)]
        async with write_lock:
            write_json_list(output_file, snapshot)

    abort_event = asyncio.Event()
    if "llm" in enabled:
        key_count = evaluator_key_count()
    use_shards = "llm" in enabled and key_count > 1

    async def _evaluate_record(
        index: int,
        record: dict[str, Any],
        *,
        slot_index: int = 0,
        limit_concurrency: bool = True,
    ) -> None:
        """信号量包装的单条评测任务（分片模式由 shard_sem 限流）。"""
        if results[index] is not None or abort_event.is_set():
            return

        async def _run() -> None:
            nonlocal completed_since_save
            if abort_event.is_set():
                return
            async with progress_lock:
                progress.set_description(
                    f"eval conv{record.get('conversation_idx')} qa{record.get('qa_index')}"
                )
            try:
                results[index] = await _evaluate_one(
                    record,
                    enabled_metrics=enabled,
                    slot_index=slot_index,
                )
            except ApiFailureBudgetExceeded as exc:
                abort_event.set()
                print(f"[eval] abort: {exc}", flush=True)
                return
            except Exception:
                if abort_event.is_set():
                    return
                raise
            async with progress_lock:
                progress.update(1)
            completed_since_save += 1
            if completed_since_save >= 5:
                completed_since_save = 0
                await _flush_partial()

        if limit_concurrency:
            async with semaphore:
                await _run()
        else:
            await _run()

    async def _shard_worker(shard_id: int) -> None:
        per_shard = max(1, concurrency // key_count)
        shard_sem = asyncio.Semaphore(per_shard)
        pending_indices = [
            index
            for index, item in enumerate(payload)
            if results[index] is None and index % key_count == shard_id
        ]

        async def _run_index(index: int) -> None:
            if abort_event.is_set():
                return
            async with shard_sem:
                await _evaluate_record(
                    index,
                    payload[index],
                    slot_index=shard_id,
                    limit_concurrency=False,
                )

        await asyncio.gather(*[_run_index(index) for index in pending_indices])

    try:
        if use_shards:
            print(
                f"[eval] shard mode: {key_count} workers (index % {key_count}), "
                f"~{max(1, concurrency // key_count)} concurrent per shard",
                flush=True,
            )
            await asyncio.gather(*[_shard_worker(shard_id) for shard_id in range(key_count)])
        else:
            await asyncio.gather(
                *[
                    _evaluate_record(index, record, slot_index=0)
                    for index, record in enumerate(payload)
                ]
            )
    except ApiFailureBudgetExceeded as exc:
        print(f"[eval] stopped after API failure budget: {exc}", flush=True)
    finally:
        progress.close()
        if "llm" in enabled:
            shutdown_evaluator()

    evaluated = [item if item is not None else payload[i] for i, item in enumerate(results)]
    await _flush_partial()
    if abort_event.is_set():
        pending = sum(1 for item in results if item is None)
        raise ApiFailureBudgetExceeded(
            f"eval incomplete: {pending} record(s) pending after API failure budget exhausted"
        )
    return evaluated


async def _evaluate_one(
    record: dict[str, Any],
    *,
    enabled_metrics: set[str],
    slot_index: int = 0,
) -> dict[str, Any]:
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
            llm_score, llm_reason = await evaluate_llm_judge(
                question,
                candidate,
                prediction,
                slot_index=slot_index,
            )
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
    return apply_empty_answer_llm_score_rule(out)
