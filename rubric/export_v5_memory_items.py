from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V5_WORKSPACES_DIR = ROOT / "v5" / "workspaces"
RUBRIC_MEMORY_DIR = ROOT / "rubric" / "memory_items"

TARGETS = {
    "minimax": V5_WORKSPACES_DIR / "allconv_v5_minimax",
    "deepseek": V5_WORKSPACES_DIR / "allconv_v5_deepseek",
    "gemini": V5_WORKSPACES_DIR / "allconv_v5_gemini",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_incremental_rows(add_model: str, workspace_name: str, snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for conversation in snapshot:
        conversation_idx = int(conversation.get("conversation_idx"))
        speaker_a = conversation.get("speaker_a")
        speaker_b = conversation.get("speaker_b")
        user_id = conversation.get("user_id")
        for session in conversation.get("sessions", []) or []:
            rows.append(
                {
                    "add_model": add_model,
                    "workspace_name": workspace_name,
                    "conversation_idx": conversation_idx,
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                    "user_id": user_id,
                    "session_index": int(session.get("session_index")),
                    "session_time": session.get("session_time"),
                    "memory_count_after_session": int(session.get("memory_count") or 0),
                    "written": int(session.get("written") or 0),
                    "delta_writes": int(session.get("delta_writes") or 0),
                    "db_added": int(session.get("db_added") or 0),
                    "db_updated": int(session.get("db_updated") or 0),
                    "operations": session.get("operations") or [],
                    "model_operations": session.get("model_operations") or [],
                }
            )
    return rows


def normalize_final_rows(add_model: str, workspace_name: str, snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for conversation in snapshot:
        sessions = conversation.get("sessions", []) or []
        last_session = sessions[-1] if sessions else {}
        rows.append(
            {
                "add_model": add_model,
                "workspace_name": workspace_name,
                "conversation_idx": int(conversation.get("conversation_idx")),
                "speaker_a": conversation.get("speaker_a"),
                "speaker_b": conversation.get("speaker_b"),
                "user_id": conversation.get("user_id"),
                "final_session_index": int(last_session.get("session_index") or 0),
                "final_session_time": last_session.get("session_time"),
                "final_memory_count": int(conversation.get("memory_count") or last_session.get("memory_count") or 0),
                "final_memory": last_session.get("memory") or [],
            }
        )
    return rows


def export_one(add_model: str, workspace_dir: Path) -> dict[str, Any]:
    snapshot_path = workspace_dir / "add_snapshot.json"
    if not snapshot_path.exists():
        return {
            "add_model": add_model,
            "workspace_dir": str(workspace_dir),
            "status": "missing_snapshot",
        }
    snapshot = load_json(snapshot_path)
    if not isinstance(snapshot, list):
        raise ValueError(f"snapshot must be a list: {snapshot_path}")

    workspace_name = workspace_dir.name
    incremental_rows = normalize_incremental_rows(add_model, workspace_name, snapshot)
    final_rows = normalize_final_rows(add_model, workspace_name, snapshot)

    workspace_export_dir = workspace_dir / "memory_items"
    rubric_export_dir = RUBRIC_MEMORY_DIR / workspace_name

    write_json(workspace_export_dir / "incremental_memory_items.json", incremental_rows)
    write_jsonl(workspace_export_dir / "incremental_memory_items.jsonl", incremental_rows)
    write_json(workspace_export_dir / "final_memory_items.json", final_rows)
    write_jsonl(workspace_export_dir / "final_memory_items.jsonl", final_rows)

    write_json(rubric_export_dir / "incremental_memory_items.json", incremental_rows)
    write_jsonl(rubric_export_dir / "incremental_memory_items.jsonl", incremental_rows)
    write_json(rubric_export_dir / "final_memory_items.json", final_rows)
    write_jsonl(rubric_export_dir / "final_memory_items.jsonl", final_rows)

    return {
        "add_model": add_model,
        "workspace_dir": str(workspace_dir),
        "workspace_export_dir": str(workspace_export_dir),
        "rubric_export_dir": str(rubric_export_dir),
        "conversation_count": len(final_rows),
        "incremental_session_count": len(incremental_rows),
        "status": "ok",
    }


def main() -> None:
    summary: list[dict[str, Any]] = []
    for add_model, workspace_dir in TARGETS.items():
        summary.append(export_one(add_model, workspace_dir))
    write_json(RUBRIC_MEMORY_DIR / "export_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
