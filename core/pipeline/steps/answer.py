"""answer 步骤：根据 search 的 retrieval 块生成 predicted_answer。

读 search_results.json，对每条 QA：
  - 格式化 speaker_a/b 的 selected 记忆 → 填入 prompts/answer_{mode}.txt
  - 调用 PipelineLLM（.env 里 key/api_base/model_name）
  - 后处理：取首行、去掉 "Answer:" 前缀（与 LoCoMo prompt 22 对齐）

写 search_results_answer{mode}.json（字段与 search 相同，仅填充 predicted_answer/response）。
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from core.infra.llm_client import PipelineLLM
from core.infra.progress import ProgressBar
from core.infra.time_resolver import parse_anchor_date, resolve_relative_time

PIPELINE_DIR = Path(__file__).resolve().parents[3]
# mode → 模板路径；history=单块 memory_history，22=双 speaker 分块
_ANSWER_TEMPLATES: dict[str, Path] = {
    "history": PIPELINE_DIR / "prompts" / "answer_history.txt",
    "22": PIPELINE_DIR / "prompts" / "answer_22.txt",
}
# 使用 {memory_history} 占位符的模板（与 22 的 speaker_* 区分）
_HISTORY_MODES = frozenset({"history"})


def _format_selected_memories(retrieval: dict[str, Any] | None) -> str:
    """将 retrieval.selected 格式化为带时间戳的 bullet 列表，供 answer prompt 使用。"""
    if not isinstance(retrieval, dict):
        return ""
    selected = retrieval.get("selected")
    if not isinstance(selected, list):
        return ""
    lines: list[str] = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        created_at = str(item.get("created_at") or "").strip()
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        anchor_time = str(meta.get("anchor_time") or meta.get("source_session_time") or "").strip()
        resolved_time = ""
        if anchor_time:
            anchor_date = parse_anchor_date(anchor_time)
            if anchor_date:
                resolved = resolve_relative_time(text, anchor_date)
                if resolved:
                    resolved_time = resolved.value
        extra = ""
        if anchor_time and resolved_time:
            extra = f" (anchor_time={anchor_time}; resolved_time={resolved_time})"
        elif anchor_time:
            extra = f" (anchor_time={anchor_time}; resolved_time=UNKNOWN)"
        if created_at:
            lines.append(f"- [{created_at}] {text}{extra}")
        else:
            lines.append(f"- {text}{extra}")
    return "\n".join(lines)


def _format_memory_history(record: dict[str, Any]) -> str:
    """合并 global 或双 speaker 的 selected 记忆为一块 MEMORY HISTORY 文本。"""
    global_block = record.get("retrieval")
    if isinstance(global_block, dict):
        block = _format_selected_memories(global_block)
        return block if block.strip() else "(no memories)"
    sections: list[str] = []
    pairs = (
        (str(record.get("speaker_a_name") or "speaker_a"), record.get("speaker_a_retrieval")),
        (str(record.get("speaker_b_name") or "speaker_b"), record.get("speaker_b_retrieval")),
    )
    for speaker_name, retrieval in pairs:
        block = _format_selected_memories(retrieval if isinstance(retrieval, dict) else None)
        if block.strip():
            sections.append(f"Memories for {speaker_name}:\n{block}")
    return "\n\n".join(sections) if sections else "(no memories)"


def _record_has_memories(record: dict[str, Any]) -> bool:
    """判断 search 记录是否含可用于 answer 的 selected 记忆。"""
    global_block = record.get("retrieval")
    if isinstance(global_block, dict):
        return bool(_format_selected_memories(global_block).strip())
    speaker_a = record.get("speaker_a_retrieval")
    speaker_b = record.get("speaker_b_retrieval")
    return bool(
        _format_selected_memories(speaker_a if isinstance(speaker_a, dict) else None)
        or _format_selected_memories(speaker_b if isinstance(speaker_b, dict) else None)
    )


def _postprocess_answer_minimal(answer: str) -> str:
    """prompt 22 风格后处理：去 Answer: 前缀、只保留首行、压空白。"""
    answer_text = str(answer or "").strip()
    if not answer_text:
        return answer_text
    answer_text = re.sub(r"^(answer:\s*)", "", answer_text, flags=re.IGNORECASE).strip()
    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    if lines:
        answer_text = lines[0]
    return " ".join(answer_text.split())


def _render_prompt(
    *,
    template_path: Path,
    speaker_1_name: str,
    speaker_1_memories: str,
    speaker_2_name: str,
    speaker_2_memories: str,
    question: str,
) -> str:
    """用 speaker/记忆/问题填充 answer 模板。"""
    template = template_path.read_text(encoding="utf-8")
    return template.format(
        speaker_1_name=speaker_1_name,
        speaker_1_memories=speaker_1_memories or "(no memories)",
        speaker_2_name=speaker_2_name,
        speaker_2_memories=speaker_2_memories or "(no memories)",
        question=question,
    )


def _render_prompt_history(
    *,
    template_path: Path,
    memory_history: str,
    question: str,
) -> str:
    """用合并后的 memory_history + question 填充 answer_history 模板。"""
    template = template_path.read_text(encoding="utf-8")
    return template.format(
        memory_history=memory_history or "(no memories)",
        question=question,
    )


def _build_answer_prompt(
    *,
    template_path: Path,
    mode: str,
    record: dict[str, Any],
) -> str:
    """按 prompt mode 选择双 speaker 或单块 memory_history 渲染方式。"""
    question = str(record.get("question") or "")
    if mode in _HISTORY_MODES:
        return _render_prompt_history(
            template_path=template_path,
            memory_history=_format_memory_history(record),
            question=question,
        )
    speaker_a = record.get("speaker_a_retrieval")
    speaker_b = record.get("speaker_b_retrieval")
    return _render_prompt(
        template_path=template_path,
        speaker_1_name=str(record.get("speaker_a_name") or ""),
        speaker_1_memories=_format_selected_memories(
            speaker_a if isinstance(speaker_a, dict) else None
        ),
        speaker_2_name=str(record.get("speaker_b_name") or ""),
        speaker_2_memories=_format_selected_memories(
            speaker_b if isinstance(speaker_b, dict) else None
        ),
        question=question,
    )


async def reanswer_dataset(
    *,
    input_path: str | Path,
    output_path: str | Path,
    concurrency: int = 2,
    answer_prompt_mode: str = "history",
    llm: PipelineLLM | None = None,
    progress_label: str | None = None,
) -> list[dict[str, Any]]:
    """并发对 search 结果逐条生成答案并写入 output_path。"""
    mode = str(answer_prompt_mode or "history").strip()
    template_path = _ANSWER_TEMPLATES.get(mode)
    if template_path is None:
        raise ValueError(f"unsupported answer_prompt_mode: {answer_prompt_mode}")

    resolved_llm = llm or PipelineLLM()
    input_file = Path(input_path)
    payload = json.loads(input_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("answer step expects a JSON list of search records")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any] | None] = [None] * len(payload)
    resumed = 0
    if output_file.exists():
        try:
            existing = json.loads(output_file.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                for index, item in enumerate(existing):
                    if index < len(results) and isinstance(item, dict):
                        if str(item.get("predicted_answer") or item.get("response") or "").strip():
                            results[index] = item
                            resumed += 1
        except (json.JSONDecodeError, OSError):
            pass
    pending = sum(1 for item in results if item is None)
    print(
        f"[answer] records={len(payload)} concurrency={concurrency} "
        f"resumed={resumed} pending={pending}",
        flush=True,
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))
    progress = ProgressBar("answer", total=len(payload) or None, unit="qa", label=progress_label)
    if resumed:
        progress.update(resumed)
    progress_lock = asyncio.Lock()
    write_lock = asyncio.Lock()
    completed_since_save = 0

    async def _flush_partial() -> None:
        """周期性落盘，便于 API 中断后 --start-from-step answer 续跑。"""
        snapshot = [item if item is not None else payload[i] for i, item in enumerate(results)]
        async with write_lock:
            output_file.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    async def _run_one(index: int, record: dict[str, Any]) -> None:
        """处理单条 QA：有 selected 记忆则调 LLM，否则 predicted_answer 为空。"""
        nonlocal completed_since_save
        if results[index] is not None:
            return
        async with semaphore:
            conv_idx = record.get("conversation_idx")
            qa_index = record.get("qa_index")
            async with progress_lock:
                progress.set_description(f"answer conv{conv_idx} qa{qa_index}")
            has_memories = _record_has_memories(record)
            predicted = ""
            if has_memories:
                prompt = _build_answer_prompt(
                    template_path=template_path,
                    mode=mode,
                    record=record,
                )
                raw = await asyncio.to_thread(resolved_llm.chat, prompt)
                predicted = _postprocess_answer_minimal(raw)
            updated = dict(record)
            updated["predicted_answer"] = predicted
            updated["response"] = predicted
            results[index] = updated
            async with progress_lock:
                progress.update(1)
            completed_since_save += 1
            if completed_since_save >= 5:
                completed_since_save = 0
                await _flush_partial()

    try:
        await asyncio.gather(*[_run_one(index, record) for index, record in enumerate(payload)])
    finally:
        progress.close()
    finalized = [item if item is not None else payload[i] for i, item in enumerate(results)]
    await _flush_partial()
    return finalized
