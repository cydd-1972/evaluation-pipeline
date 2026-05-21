"""加载 datasets/locomo_refined.json。

解析为 LoCoMoConversation（sessions + qa），供 add/search 遍历。
QA 字段含 question、answer、category、evidence 等，与 LoCoMo 官方格式一致。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoCoMoMessage:
    """单条对话消息（含可选图片 caption）。"""

    role: str
    content: str
    dia_id: str = ""
    speaker_name: str = ""
    message_index: int | None = None
    img_urls: list[str] = field(default_factory=list)
    blip_caption: str = ""


@dataclass
class LoCoMoSession:
    """一个 session：序号、时间戳、消息列表。"""

    index: int
    date_time: str
    messages: list[LoCoMoMessage]


@dataclass
class LoCoMoQA:
    """一道评测题及其参考答案、类别、证据等。"""

    question: str
    answer: str
    category: str
    evidence: list[str]
    answer_fixed: list[str] = field(default_factory=list)
    character: str = ""
    qa_type: str = ""
    options: list[str] = field(default_factory=list)
    answer_texts: list[str] = field(default_factory=list)
    answer_raw: str | list[str] = ""


@dataclass
class LoCoMoConversation:
    """一段完整对话：双 speaker、多 session、多道 QA。"""

    idx: int
    speaker_a: str
    speaker_b: str
    sessions: list[LoCoMoSession]
    qa: list[LoCoMoQA]

    @property
    def system_prompt(self) -> str:
        """首个 session 首条消息内容；标准 LoCoMo 通常为空，CL-bench 可能有 system。"""
        if not self.sessions:
            return ""
        first_session = self.sessions[0]
        if not first_session.messages:
            return ""
        return first_session.messages[0].content.strip()

    @property
    def all_participants(self) -> list[str]:
        """全对话出现过的说话人名（保持首次出现顺序）；无则回退 speaker_a/b。"""
        seen: dict[str, None] = {}
        for session in self.sessions:
            for message in session.messages:
                name = message.speaker_name.strip()
                if name:
                    seen[name] = None
        if seen:
            return list(seen.keys())
        return [p for p in [self.speaker_a, self.speaker_b] if p.strip()]


def load_locomo_dataset(path: str | Path, max_conversations: int | None = None) -> list[LoCoMoConversation]:
    """从 JSON 文件加载 LoCoMo 记录列表；max_conversations 限制条数。"""
    dataset_path = Path(path)
    raw_records = json.loads(dataset_path.read_text(encoding="utf-8"))
    conversations: list[LoCoMoConversation] = []

    for idx, record in enumerate(raw_records):
        if max_conversations is not None and idx >= max_conversations:
            break
        conversations.append(_parse_record(idx=idx, record=record))

    return conversations


def _parse_record(*, idx: int, record: dict[str, Any]) -> LoCoMoConversation:
    """解析单条原始 JSON 记录为 LoCoMoConversation。"""
    conversation = record.get("conversation")
    if not isinstance(conversation, dict):
        raise ValueError(f"LoCoMo record {idx} is missing a valid 'conversation' object")
    speaker_a = str(conversation.get("speaker_a") or "")
    speaker_b = str(conversation.get("speaker_b") or "")
    sessions = _parse_sessions(
        conversation,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
    )
    if not sessions:
        raise ValueError(f"LoCoMo record {idx} does not contain any parsed dialogue sessions")
    qa = _parse_qa_items(record.get("qa") or [])
    return LoCoMoConversation(
        idx=idx,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        sessions=sessions,
        qa=qa,
    )


def _parse_sessions(
    conversation: dict[str, Any],
    *,
    speaker_a: str,
    speaker_b: str,
) -> list[LoCoMoSession]:
    """解析 conversation 里 session_1, session_2, ... 及对应 date_time。"""
    sessions: list[LoCoMoSession] = []
    session_index = 1
    while True:
        date_key = f"session_{session_index}_date_time"
        items_key = f"session_{session_index}"
        if items_key not in conversation:
            break

        date_time = str(conversation.get(date_key) or "")
        raw_messages = conversation.get(items_key) or []
        messages: list[LoCoMoMessage] = []
        for position, item in enumerate(raw_messages, start=1):
            if not isinstance(item, dict):
                continue
            content = str(item.get("text") or "")
            if not content.strip():
                continue
            dia_id = str(item.get("dia_id") or "").strip()
            _, message_index = _parse_dia_id(dia_id)
            raw_img_urls = item.get("img_url") or []
            img_urls = [str(u) for u in raw_img_urls] if isinstance(raw_img_urls, list) else []
            blip_caption = str(item.get("blip_caption") or "").strip()
            messages.append(
                LoCoMoMessage(
                    role=_normalize_role(item.get("speaker"), speaker_a=speaker_a, speaker_b=speaker_b),
                    content=content,
                    dia_id=dia_id,
                    speaker_name=str(item.get("speaker") or ""),
                    message_index=message_index if message_index is not None else position,
                    img_urls=img_urls,
                    blip_caption=blip_caption,
                )
            )
        if not messages:
            break
        sessions.append(
            LoCoMoSession(
                index=session_index,
                date_time=date_time,
                messages=messages,
            )
        )
        session_index += 1

    return sessions


def _parse_qa_items(raw_items: list[Any]) -> list[LoCoMoQA]:
    """将 qa 数组转为 LoCoMoQA 列表。"""
    parsed: list[LoCoMoQA] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        normalized = normalize_qa_item(item)
        parsed.append(
            LoCoMoQA(
                question=normalized["question"],
                answer=normalized["answer"],
                category=normalized["category"],
                evidence=normalized["evidence"],
                answer_fixed=normalized["answer_fixed"],
                character=normalized["character"],
                qa_type=normalized["qa_type"],
                options=normalized["options"],
                answer_texts=normalized["answer_texts"],
                answer_raw=normalized["answer_raw"],
            )
        )
    return parsed


_QA_TYPE_VALUES = frozenset({"single_choice", "multi_select", "ordering"})


def normalize_qa_item(item: dict[str, Any]) -> dict[str, Any]:
    """规范化单条 QA 字典（含选择题字母、evidence、answer_raw 等）。"""
    question = str(item.get("question") or "")
    category = str(item.get("category") or "")
    character = str(item.get("character") or "").strip()
    qa_type = str(item.get("qa_type") or "").strip().lower()
    options = _normalize_option_values(item.get("option") or item.get("options"))
    answer_texts = _normalize_answer_texts(item.get("answer"))
    answer = _normalize_reference_answer(
        raw_answer=item.get("answer"),
        qa_type=qa_type,
        options=options,
    )
    answer_fixed = _normalize_answer_fixed(item.get("answer_fixed"))
    if not answer_fixed and qa_type in _QA_TYPE_VALUES and answer:
        answer_fixed = [answer]
    return {
        "question": question,
        "answer": answer,
        "category": category,
        "evidence": _normalize_evidence(item),
        "answer_fixed": answer_fixed,
        "character": character,
        "qa_type": qa_type,
        "options": options,
        "answer_texts": answer_texts,
        "answer_raw": _normalize_raw_answer(item.get("answer")),
    }


def _normalize_role(raw_speaker: Any, *, speaker_a: str, speaker_b: str) -> str:
    """将原始 speaker 名映射为 user / assistant / system / unknown。"""
    speaker = str(raw_speaker or "").strip()
    canonical_speaker = _canonicalize_speaker_name(speaker)
    if canonical_speaker == _canonicalize_speaker_name(speaker_a):
        return "user"
    if canonical_speaker == _canonicalize_speaker_name(speaker_b):
        return "assistant"
    if canonical_speaker == "system":
        return "system"
    return "unknown"


def _canonicalize_speaker_name(raw_value: Any) -> str:
    """说话人名归一化：压空白、转 casefold 便于比较。"""
    return re.sub(r"\s+", " ", str(raw_value or "").strip()).casefold()


def _parse_dia_id(raw_value: str) -> tuple[int | None, int | None]:
    """解析 dia_id 如 D3:12 → (session_idx, message_idx)。"""
    value = str(raw_value or "").strip()
    match = value and re.match(r"^D(\d+):(\d+)$", value)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _normalize_answer_fixed(raw_value: Any) -> list[str]:
    """解析 answer_fixed 字段为字符串列表。"""
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value if str(item).strip()]
    if raw_value in (None, ""):
        return []
    text = str(raw_value).strip()
    return [text] if text else []


def _normalize_evidence(item: dict[str, Any]) -> list[str]:
    """从 evidence / evidence_dialogues / evidence_qid 取第一个非空列表。"""
    for field in ("evidence", "evidence_dialogues", "evidence_qid"):
        raw_value = item.get(field)
        if isinstance(raw_value, list):
            values = [str(entry) for entry in raw_value if str(entry).strip()]
            if values:
                return values
    return []


def _normalize_option_values(raw_value: Any) -> list[str]:
    """解析选择题 option/options 为字符串列表。"""
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return []


def _normalize_answer_texts(raw_value: Any) -> list[str]:
    """把 answer 字段展平为文本列表（单值也包成 list）。"""
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    text = str(raw_value or "").strip()
    return [text] if text else []


def _normalize_raw_answer(raw_value: Any) -> str | list[str]:
    """保留 answer 原始形态（字符串或列表），写入 search 记录的 answer 字段。"""
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return str(raw_value or "")


def _normalize_reference_answer(*, raw_answer: Any, qa_type: str, options: list[str]) -> str:
    """结构化 QA 将答案转为选项字母；开放题则保留原文。"""
    char = str(qa_type or "").strip().lower()
    if char not in _QA_TYPE_VALUES:
        return str(raw_answer or "")

    answer_texts = _normalize_answer_texts(raw_answer)
    letters = _resolve_option_letters(answer_texts=answer_texts, options=options)
    if not letters:
        return str(raw_answer or "")
    if char == "single_choice":
        return letters[0]
    return ",".join(letters)


def _resolve_option_letters(*, answer_texts: list[str], options: list[str]) -> list[str]:
    """把每条答案文本解析为选项字母 A/B/C...。"""
    if not answer_texts:
        return []
    option_map = _build_option_map(options)
    if not option_map:
        return []

    resolved: list[str] = []
    for answer_text in answer_texts:
        letter = _resolve_single_option_letter(answer_text, option_map)
        if not letter:
            return []
        resolved.append(letter)
    return resolved


def _build_option_map(options: list[str]) -> dict[str, str]:
    """选项列表 → {字母: 选项正文} 映射。"""
    mapping: dict[str, str] = {}
    for idx, option in enumerate(options):
        fallback_letter = chr(ord("A") + idx)
        text = str(option or "").strip()
        if not text:
            continue
        match = re.match(r"^\s*([A-Z])\.\s*(.+?)\s*$", text)
        if match:
            mapping[match.group(1).upper()] = match.group(2).strip()
        else:
            mapping[fallback_letter] = text
    return mapping


def _resolve_single_option_letter(answer_text: str, option_map: dict[str, str]) -> str:
    """单条答案匹配字母：支持 \"A\"、\"A. xxx\" 或与选项正文完全一致。"""
    text = str(answer_text or "").strip()
    if not text:
        return ""
    direct_letter = re.fullmatch(r"\(?\s*([A-Z])\s*\)?", text, flags=re.IGNORECASE)
    if direct_letter:
        return direct_letter.group(1).upper()
    prefixed = re.match(r"^\s*([A-Z])\.\s*(.+?)\s*$", text, flags=re.IGNORECASE)
    if prefixed:
        return prefixed.group(1).upper()
    for letter, option_text in option_map.items():
        if text == option_text:
            return letter
        if text == f"{letter}. {option_text}":
            return letter
    return ""
