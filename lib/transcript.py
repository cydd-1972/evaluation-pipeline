"""把 LoCoMo session 格式化为 transcript 文本或 structured JSON（global add 用）。"""

from __future__ import annotations

import json
from typing import Any

from lib.data_loader import LoCoMoConversation, LoCoMoSession


def format_session_transcript(session: LoCoMoSession) -> str:
    """单 session 多轮对话 → 「说话人: 内容」多行文本，含可选图片 caption。"""
    lines: list[str] = []
    if session.date_time.strip():
        lines.append(f"[Session time: {session.date_time.strip()}]")
    for message in session.messages:
        speaker = message.speaker_name.strip() or message.role.strip() or "unknown"
        content = message.content.strip()
        if not content:
            continue
        lines.append(f"{speaker}: {content}")
        if message.blip_caption.strip():
            lines.append(f"[Image caption for {speaker}: {message.blip_caption.strip()}]")
    return "\n".join(lines)


def format_session_structured(session: LoCoMoSession) -> dict[str, Any]:
    """单 session → global memory prompt 用的 D_n JSON 对象。"""
    messages: list[dict[str, Any]] = []
    for message in session.messages:
        speaker = message.speaker_name.strip() or message.role.strip() or "unknown"
        content = message.content.strip()
        caption = message.blip_caption.strip()
        if not content and not caption:
            continue
        entry: dict[str, Any] = {"speaker": speaker, "text": content}
        if caption:
            entry["image_caption"] = caption
        else:
            entry["image_caption"] = None
        messages.append(entry)
    return {
        "session_index": int(session.index),
        "session_time": session.date_time.strip() or "unknown",
        "messages": messages,
    }


def format_sessions_structured_json(sessions: list[LoCoMoSession]) -> str:
    """多 session → history_sessions_json 占位符（JSON 数组字符串）。"""
    payload = [format_session_structured(session) for session in sessions]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_session_structured_json(session: LoCoMoSession) -> str:
    """单 session → current_session_json 占位符（JSON 对象字符串）。"""
    return json.dumps(format_session_structured(session), ensure_ascii=False, indent=2)


def iter_sessions(
    conversation: LoCoMoConversation,
    *,
    max_sessions: int | None,
) -> list[LoCoMoSession]:
    """返回要处理的 session 列表；max_sessions 截断前 N 个。"""
    sessions = list(conversation.sessions)
    if max_sessions is not None:
        sessions = sessions[: max(0, int(max_sessions))]
    return sessions
