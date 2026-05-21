"""把 LoCoMo session 消息格式化成 fact_extraction prompt 里的 transcript 文本。"""

from __future__ import annotations

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
