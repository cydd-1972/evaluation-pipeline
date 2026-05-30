"""LoCoMo speaker 的稳定 user_id（UUIDv5）。

同一 conversation_idx + speaker_role + speaker_name 多次运行得到相同 UUID，
保证 add/search 使用同一 user_id 读写 memories。
"""

from __future__ import annotations

from uuid import NAMESPACE_URL, UUID, uuid5


def build_speaker_user_id(conv_idx: int, speaker_role: str, speaker_name: str) -> UUID:
    """生成 LoCoMo 某对话某角色的稳定 Postgres user_id。"""
    return uuid5(
        NAMESPACE_URL,
        f"locomo:{conv_idx}:{speaker_role.strip().lower()}:{speaker_name.strip()}",
    )


def build_conversation_user_id(conv_idx: int) -> UUID:
    """方案③ global add/search：每 conversation 一个 user_id（非 per-speaker）。"""
    return uuid5(NAMESPACE_URL, f"locomo:conv:{int(conv_idx)}")
