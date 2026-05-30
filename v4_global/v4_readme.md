
## 完整方案：提取 + 决策分离

### 核心思路

将原来的一次性任务拆分为两个独立的步骤，但**最终输出仍然是一个模型的输出**：

1. **步骤一（提取）**：模型从当前会话中提取所有原子化事实（不决策）
2. **步骤二（决策）**：后处理模块决定哪些是ADD、哪些是UPDATE

这样做的好处：
- 模型任务简化，错误率降低
- 决策逻辑可以用规则实现，更可控
- SFT数据中仍然包含完整的决策结果

---

## 新方案架构

```
输入: M_{n-1} (旧记忆) + D_n (当前会话)
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  步骤一：事实提取（模型执行）                  │
│  输出：新事实列表（无id，无operation）         │
│  每条事实是原子化的、带原始时间表达            │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  步骤二：决策（代码/规则执行）                │
│  比对新事实与M_{n-1}，决定：                 │
│  - ADD：全新信息，分配新id                   │
│  - UPDATE：信息更新，复用id                  │
│  - NONE：完全重复，丢弃                      │
└─────────────────────────────────────────────┘
                    │
                    ▼
              输出最终记忆列表
```

---

## 步骤一：新的 `add` 提示词（只提取，不决策）

```markdown
You are a fact extractor for a conversation between {speaker_a} and {speaker_b}.

Your task is to extract atomic facts from the current session. You do NOT need to compare with existing memories or decide ADD/UPDATE/NONE.

# INPUT

## M_{n-1} — existing memory snapshot (for context only, do NOT use for decision)
{old_memory_json}

## Prior sessions — context window only (for disambiguation)
{history_sessions_json}

## D_n — current session (primary source)
{current_session_json}

# YOUR TASK

Extract from D_n all **atomic facts** that could potentially be stored as memories.

## What is an atomic fact?

A single, self-contained piece of information. If a sentence contains multiple facts, split them.

- Wrong: "Caroline felt accepted and gained courage."
- Correct: "Caroline felt accepted at the support group."
- Correct: "Caroline gained courage to embrace herself."

## Types of facts to extract

| Type | Description | Example |
|------|-------------|---------|
| Event | Something that happened | "Melanie ran a charity race last Saturday." |
| Plan | A stated intention | "Caroline plans to pursue counseling." |
| State | A condition or relationship | "John has a one-year-old son named Kyle." |
| Feeling | An emotional response | "Caroline found the transgender stories inspiring." |
| Negative | A failure, rejection, or inability | "Tim received a rejection for a summer job." |
| Attribute | A possession or characteristic | "Joanna is allergic to most reptiles." |

## Time expression rules

Keep the **original time expression** as spoken. Do NOT convert to absolute dates.

- Correct: "Melanie ran a charity race last Saturday."
- Correct: "Caroline attended a support group yesterday."
- Wrong: "On 7 May 2023, Caroline attended a support group."

If the conversation has a session_time, you may append it as context:
- "Caroline attended a support group yesterday (session: 2023-05-08)."

## Subject rules

Use explicit names. Never use bare pronouns.

- Wrong: "She went to a group."
- Correct: "Caroline attended an LGBTQ support group."

## Output format

Return a JSON array of fact objects. Each object has:
- `"fact"`: the atomic fact sentence (string)
- `"type"`: one of "event", "plan", "state", "feeling", "negative", "attribute"

Example:
```json
[
  {"fact": "Caroline attended an LGBTQ support group yesterday.", "type": "event"},
  {"fact": "Caroline felt accepted at the support group.", "type": "feeling"},
  {"fact": "Caroline plans to pursue education in counseling or mental health.", "type": "plan"}
]
```

# OUTPUT

Return ONLY the JSON array. No other text.
```

---

## 步骤二：决策逻辑（代码实现）

```python
def decide_memory_operations(new_facts, old_memories):
    """
    决定每个新事实应该执行什么操作
    
    Args:
        new_facts: 步骤一提取的事实列表，每条包含 {"fact": str, "type": str}
        old_memories: 现有记忆列表，每条包含 {"id": str, "text": str, "event": str}
    
    Returns:
        需要输出到最终记忆列表的操作列表
    """
    operations = []
    used_old_ids = set()
    
    for fact_obj in new_facts:
        fact_text = fact_obj["fact"]
        fact_type = fact_obj["type"]
        
        # 1. 在旧记忆中寻找相关记忆
        best_match = find_best_match(fact_text, old_memories)
        
        if best_match is None:
            # 完全新的信息 → ADD
            operations.append({
                "action": "ADD",
                "id": generate_new_id(old_memories),
                "text": fact_text,
                "type": fact_type
            })
            continue
        
        # 2. 找到了匹配的记忆，判断是否需要UPDATE
        match_id, match_text, similarity = best_match
        
        if is_semantically_identical(fact_text, match_text):
            # 完全相同的语义 → NONE（忽略）
            continue
        
        # 判断是否是"更新"关系
        if is_update_relationship(match_text, fact_text, fact_type):
            # 新信息是对旧信息的补充/更正 → UPDATE
            operations.append({
                "action": "UPDATE",
                "id": match_id,
                "text": fact_text,  # 新版本
                "old_text": match_text,
                "type": fact_type
            })
            used_old_ids.add(match_id)
        else:
            # 不相关或平行信息 → ADD
            operations.append({
                "action": "ADD",
                "id": generate_new_id(old_memories),
                "text": fact_text,
                "type": fact_type
            })
    
    return operations


def find_best_match(fact_text, old_memories):
    """
    在旧记忆中寻找与当前事实最相似的记忆
    """
    best_score = 0.5  # 相似度阈值
    best_match = None
    
    for mem in old_memories:
        score = compute_similarity(fact_text, mem["text"])
        if score > best_score:
            best_score = score
            best_match = (mem["id"], mem["text"], score)
    
    return best_match


def is_semantically_identical(text1, text2):
    """
    判断两句话语义是否完全相同
    """
    # 实现：可以基于关键词重合度、向量相似度等
    # 简化版：提取核心词（动词+名词）比较
    pass


def is_update_relationship(old_text, new_text, fact_type):
    """
    判断新事实是否是对旧记忆的更新（而非独立的新事实）
    
    更新关系判断规则：
    1. 主语相同
    2. 核心动词相同或相关
    3. 新事实包含更具体或更正后的信息
    """
    # 规则1：plan类型 → 计划改变 → UPDATE
    if fact_type == "plan":
        return True
    
    # 规则2：event类型 → 时间更精确 → UPDATE
    # 规则3：state类型 → 属性补充 → UPDATE
    # 其他情况 → ADD
    
    # 具体实现需要根据实体识别和语义分析
    pass
```

---

## 决策规则的细化

为了让决策更准确，需要明确定义什么情况下应该UPDATE：

### UPDATE的适用场景

| 场景 | 示例 | 旧记忆 | 新事实 | 决策 |
|------|------|--------|--------|------|
| 计划改变 | 艺术展时间 | "Caroline plans to hold an art show in August." | "Caroline plans to hold an art show in September." | UPDATE |
| 信息细化 | 具体化描述 | "John is exploring endorsements." | "John signed a deal with Nike." | UPDATE |
| 状态演进 | 事件完成 | "Jon is preparing for a dance competition." | "Jon performed at the festival." | UPDATE |
| 事实更正 | 错误修正 | "Jon's dance crew won first place last year." | "In 2022, Jon's dance crew won first place." | UPDATE |
| 属性补充 | 增加细节 | "Melanie owns a dog named Luna." | "Melanie owns a dog named Luna and a cat named Oliver." | UPDATE |

### ADD的适用场景

| 场景 | 示例 | 旧记忆 | 新事实 | 决策 |
|------|------|--------|--------|------|
| 全新事件 | 新发生的事情 | 无 | "Caroline gave a talk at a school event." | ADD |
| 不同事件 | 同一个话题的不同事件 | "Caroline attended a support group." | "Caroline attended a counseling workshop." | ADD |
| 独立属性 | 不同方面的属性 | "Caroline owns a necklace from her grandma." | "Caroline owns a hand-painted bowl." | ADD |
| 平行感受 | 不同感受 | "Caroline felt accepted." | "Caroline felt inspired." | ADD |

### NONE的适用场景

| 场景 | 示例 | 旧记忆 | 新事实 | 决策 |
|------|------|--------|--------|------|
| 完全重复 | 一模一样的信息 | "Caroline attended an LGBTQ support group." | "Caroline attended an LGBTQ support group." | NONE |
| 同义重复 | 表述不同但意思相同 | "Caroline went to a support group." | "Caroline attended a support group." | NONE |

---

## 完整的代码接口设计

```python
def process_session(old_memories, history_sessions, current_session):
    """
    处理一个会话，输出更新后的记忆列表
    
    Args:
        old_memories: 现有记忆列表
        history_sessions: 历史会话（用于上下文）
        current_session: 当前会话
    
    Returns:
        更新后的记忆列表
    """
    # 步骤一：调用模型提取事实
    new_facts = call_extraction_model(
        old_memories=old_memories,
        history_sessions=history_sessions,
        current_session=current_session
    )
    # new_facts 格式: [{"fact": "Caroline attended...", "type": "event"}, ...]
    
    # 步骤二：决策
    operations = decide_memory_operations(new_facts, old_memories)
    
    # 步骤三：应用操作，生成新记忆列表
    new_memories = apply_operations(old_memories, operations)
    
    return new_memories


def apply_operations(old_memories, operations):
    """
    应用操作，生成新记忆列表
    """
    # 复制旧记忆
    new_memories = [dict(mem) for mem in old_memories]
    
    # 记录哪些id被UPDATE了
    updated_ids = set()
    
    for op in operations:
        if op["action"] == "ADD":
            new_memories.append({
                "id": op["id"],
                "text": op["text"],
                "event": "ADD"
            })
        elif op["action"] == "UPDATE":
            for mem in new_memories:
                if mem["id"] == op["id"]:
                    mem["text"] = op["text"]
                    mem["event"] = "UPDATE"
                    mem["old_text"] = op["old_text"]
                    updated_ids.add(op["id"])
                    break
    
    return new_memories
```

---

## 总结

| 组件 | 职责 | 实现方式 |
|------|------|---------|
| 步骤一（提取） | 从会话中提取原子化事实 | 模型（新的`add`提示词） |
| 步骤二（决策） | 比对、决定ADD/UPDATE/NONE | 代码（规则+相似度计算） |
| 步骤三（应用） | 生成最终记忆列表 | 代码 |

这样：
- 模型任务简化：只负责提取，不负责决策
- 决策可控：规则可以逐步优化
- SFT数据完整：最终输出的记忆列表仍然包含正确的operation标注
- 可调试：每一步的输出都可以独立检查

