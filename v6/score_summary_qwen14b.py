from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from openai import OpenAI


TYPE_LABELS = {
    "character": "Character Summary",
    "event": "Event Summary",
    "location": "Location Summary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score v6 summary items with Qwen3-14B")
    parser.add_argument(
        "--rubrics",
        type=Path,
        default=Path("evaluation_pipeline/v6/rubrics_summary.txt"),
    )
    parser.add_argument(
        "--workspaces",
        nargs="+",
        default=[
            "evaluation_pipeline/v6/workspaces/allconv_v6_addonly_20260618_qwen3-14b",
            "evaluation_pipeline/v6/workspaces/allconv_v6_addonly_20260618_dpsk-flash",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            f"evaluation_pipeline/v6/summary_rubric_scores_qwen14b_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-facts-per-item", type=int, default=12)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--half-only", action="store_true", default=True)
    parser.add_argument("--max-conversations", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = ["model", "type", "rubric_id", "rubric_name", "object_count", "evaluable_count", "avg_score"]
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for key in headers:
            value = row.get(key)
            cell = "" if value is None else str(value)
            if "," in cell or '"' in cell:
                cell = '"' + cell.replace('"', '""') + '"'
            values.append(cell)
        lines.append(",".join(values))
    path.write_text("\n".join(lines), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_rubrics(path: Path) -> dict[str, list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    ordered = [TYPE_LABELS["character"], TYPE_LABELS["event"], TYPE_LABELS["location"]]
    sections: dict[str, list[dict[str, Any]]] = {}
    for index, label in enumerate(ordered):
        start = text.index(label)
        end = text.index(ordered[index + 1]) if index + 1 < len(ordered) else len(text)
        block = text[start:end]
        block = block[block.index("[") : block.rindex("]") + 1]
        items = json.loads(block)
        if label == TYPE_LABELS["character"]:
            sections["character"] = items
        elif label == TYPE_LABELS["event"]:
            sections["event"] = items
        else:
            sections["location"] = items
    return sections


def normalize_workspace_name(path: str) -> str:
    name = Path(path).name
    if name.endswith("qwen3-14b"):
        return "qwen3-14b"
    if name.endswith("dpsk-flash"):
        return "dpsk-flash"
    return name


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def select_relevant_facts(item_text: str, facts: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    query_tokens = set(tokenize(item_text))
    scored: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for fact in facts:
        fact_text = str(fact.get("text") or "")
        fact_tokens = tokenize(fact_text)
        overlap = len(query_tokens.intersection(fact_tokens))
        fact_id_raw = str(fact.get("id") or "0")
        try:
            fact_id_num = int(fact_id_raw)
        except ValueError:
            fact_id_num = 0
        scored.append(((overlap, -len(fact_tokens), -fact_id_num), fact))
    scored.sort(reverse=True, key=lambda pair: pair[0])
    picked = [fact for score, fact in scored if score[0] > 0][:limit]
    if len(picked) < min(limit, len(facts)):
        seen = {str(item.get("id")) for item in picked}
        for fact in facts:
            fact_id = str(fact.get("id") or "")
            if fact_id in seen:
                continue
            picked.append(fact)
            seen.add(fact_id)
            if len(picked) >= limit:
                break
    return [
        {
            "id": str(fact.get("id") or ""),
            "text": str(fact.get("text") or ""),
            "anchor_time": str(fact.get("anchor_time") or ""),
        }
        for fact in picked
    ]


def build_tasks(
    workspace_path: Path,
    max_facts_per_item: int,
    *,
    half_only: bool,
    max_conversations: int | None,
) -> list[dict[str, Any]]:
    data = read_json(workspace_path / "add_snapshot.json")
    conversation_limit = len(data)
    if half_only:
        conversation_limit = max(1, len(data) // 2)
    if max_conversations is not None:
        conversation_limit = min(conversation_limit, max(0, int(max_conversations)))
    data = data[:conversation_limit]

    tasks: list[dict[str, Any]] = []
    model_name = normalize_workspace_name(str(workspace_path))
    for conversation in data:
        sessions = conversation.get("sessions") or []
        if not sessions:
            continue
        final_session = sessions[-1]
        final_memory = final_session.get("memory") or []
        facts = [item for item in final_memory if str(item.get("type") or "") == "fact"]
        for summary_type in ("character", "event", "location"):
            same_type_items = [item for item in final_memory if str(item.get("type") or "") == summary_type]
            for item in same_type_items:
                item_id = str(item.get("id") or "")
                peer_items = [
                    {
                        "id": str(peer.get("id") or ""),
                        "text": str(peer.get("text") or ""),
                        "anchor_time": str(peer.get("anchor_time") or ""),
                    }
                    for peer in same_type_items
                    if str(peer.get("id") or "") != item_id
                ]
                tasks.append(
                    {
                        "workspace": workspace_path.name,
                        "model": model_name,
                        "conversation_idx": int(conversation.get("conversation_idx")),
                        "speaker_a": conversation.get("speaker_a"),
                        "speaker_b": conversation.get("speaker_b"),
                        "summary_type": summary_type,
                        "item": {
                            "id": item_id,
                            "text": str(item.get("text") or ""),
                            "operation": str(item.get("operation") or ""),
                            "anchor_time": str(item.get("anchor_time") or ""),
                            "support_facts": select_relevant_facts(str(item.get("text") or ""), facts, max_facts_per_item),
                        },
                        "peer_items": peer_items,
                    }
                )
    return tasks


def extract_first_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    start = raw.find("{")
    if start < 0:
        raise ValueError("no json object start found")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start : index + 1])
    raise ValueError("unterminated json object")


def prompt_for_task(summary_type: str, rubrics: list[dict[str, Any]], task: dict[str, Any]) -> str:
    rubric_view = [
        {
            "id": item["id"],
            "name": item["name"],
            "question": item["question"],
            "why_it_matters": item["why_it_matters"],
            "notes": item.get("notes", ""),
        }
        for item in rubrics
    ]
    output_schema = {
        "item_score": {
            "id": "string",
            "rubrics": [
                {
                    "id": "rubric_id",
                    "evaluable": True,
                    "score": 0.0,
                    "reason": "short reason",
                }
            ],
        }
    }
    return (
        "你是严格的summary rubric评审器。\n"
        "请只依据提供的summary item、support facts、同类型peer summaries进行判断。\n"
        "不要使用外部知识，不要脑补原始对话中未提供的内容。\n"
        "评分规则：\n"
        "1. 对每个rubric输出 `evaluable` 和 `score`。\n"
        "2. `score` 使用连续分数 [0,1]，1最好，0最差。\n"
        "3. 如果该rubric对这个item无法判断或天然不适用，设 `evaluable=false`，`score=0`，并写简短原因。\n"
        "4. 如果可评估，设 `evaluable=true`，并给出严格分数；不要轻易给满分。\n"
        "5. 只输出一个JSON对象，不要输出解释性前后缀。\n\n"
        f"summary_type = {summary_type}\n"
        f"conversation_idx = {task['conversation_idx']}\n"
        f"speakers = {task['speaker_a']} / {task['speaker_b']}\n\n"
        f"rubrics = {json.dumps(rubric_view, ensure_ascii=False)}\n\n"
        f"peer_items_same_type = {json.dumps(task['peer_items'], ensure_ascii=False)}\n\n"
        f"item_to_score = {json.dumps(task['item'], ensure_ascii=False)}\n\n"
        f"output_schema = {json.dumps(output_schema, ensure_ascii=False)}\n"
    )


def score_task(
    client: OpenAI,
    model_name: str,
    rubrics: list[dict[str, Any]],
    task: dict[str, Any],
    trace_path: Path,
    max_retries: int,
) -> dict[str, Any]:
    prompt = prompt_for_task(task["summary_type"], rubrics, task)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a careful JSON-only evaluator."},
                    {"role": "user", "content": prompt},
                ],
                timeout=120,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            payload = extract_first_json_object(content)
            append_jsonl(
                trace_path,
                {
                    "workspace": task["workspace"],
                    "model": task["model"],
                    "conversation_idx": task["conversation_idx"],
                    "summary_type": task["summary_type"],
                    "item_id": task["item"]["id"],
                    "attempt": attempt,
                    "response": payload,
                },
            )
            return payload
        except Exception as exc:
            last_error = str(exc)
            append_jsonl(
                trace_path,
                {
                    "workspace": task["workspace"],
                    "model": task["model"],
                    "conversation_idx": task["conversation_idx"],
                    "summary_type": task["summary_type"],
                    "item_id": task["item"]["id"],
                    "attempt": attempt,
                    "error": last_error,
                },
            )
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(
        f"scoring failed workspace={task['workspace']} conv={task['conversation_idx']} "
        f"type={task['summary_type']} item={task['item']['id']}: {last_error}"
    )


def aggregate_results(
    rubrics_by_type: dict[str, list[dict[str, Any]]],
    task_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in task_results:
        grouped.setdefault((item["model"], item["summary_type"]), []).append(item)
    for (model, summary_type), records in sorted(grouped.items()):
        rubric_defs = {item["id"]: item for item in rubrics_by_type[summary_type]}
        buckets: dict[str, list[tuple[bool, float]]] = {rubric_id: [] for rubric_id in rubric_defs}
        object_count = len(records)
        for record in records:
            scored_item = record["item_score"]
            seen: set[str] = set()
            for rubric_score in scored_item.get("rubrics", []):
                rubric_id = str(rubric_score.get("id") or "")
                if rubric_id not in buckets or rubric_id in seen:
                    continue
                seen.add(rubric_id)
                evaluable = bool(rubric_score.get("evaluable"))
                score = float(rubric_score.get("score") or 0.0)
                buckets[rubric_id].append((evaluable, score))
            for rubric_id in rubric_defs:
                if rubric_id not in seen:
                    buckets[rubric_id].append((False, 0.0))
        for rubric_id, values in buckets.items():
            evaluable_scores = [score for evaluable, score in values if evaluable]
            rows.append(
                {
                    "model": model,
                    "type": summary_type,
                    "rubric_id": rubric_id,
                    "rubric_name": rubric_defs[rubric_id]["name"],
                    "object_count": object_count,
                    "evaluable_count": len(evaluable_scores),
                    "avg_score": round(mean(evaluable_scores), 4) if evaluable_scores else None,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    api_key = os.getenv("EVALUATOR_API_KEY", "").strip()
    api_base = os.getenv("EVALUATOR_API_BASE", "").strip() or "https://api.siliconflow.cn/v1"
    if not api_key:
        raise RuntimeError("missing EVALUATOR_API_KEY")

    client = OpenAI(api_key=api_key, base_url=api_base)
    model_name = "Qwen/Qwen3-14B"
    rubrics_by_type = parse_rubrics(args.rubrics)

    tasks: list[dict[str, Any]] = []
    for workspace in args.workspaces:
        tasks.extend(
            build_tasks(
                Path(workspace),
                args.max_facts_per_item,
                half_only=args.half_only,
                max_conversations=args.max_conversations,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "llm_trace.jsonl"
    task_results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as executor:
        future_map = {
            executor.submit(
                score_task,
                client,
                model_name,
                rubrics_by_type[task["summary_type"]],
                task,
                trace_path,
                args.max_retries,
            ): task
            for task in tasks
        }
        for future in as_completed(future_map):
            task = future_map[future]
            payload = future.result()
            task_results.append(
                {
                    "workspace": task["workspace"],
                    "model": task["model"],
                    "conversation_idx": task["conversation_idx"],
                    "summary_type": task["summary_type"],
                    "item_id": task["item"]["id"],
                    "item_score": payload.get("item_score", {}),
                }
            )

    task_results.sort(key=lambda item: (item["model"], item["summary_type"], item["conversation_idx"], item["item_id"]))
    aggregated = aggregate_results(rubrics_by_type, task_results)

    write_json(args.output_dir / "item_scores.json", task_results)
    write_json(args.output_dir / "aggregated_scores.json", aggregated)
    write_csv(args.output_dir / "aggregated_scores.csv", aggregated)

    summary = {
        "rubrics_path": str(args.rubrics),
        "workspaces": args.workspaces,
        "llm_model": model_name,
        "task_count": len(tasks),
        "aggregated_rows": len(aggregated),
        "output_dir": str(args.output_dir),
        "half_only": args.half_only,
        "max_conversations": args.max_conversations,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
