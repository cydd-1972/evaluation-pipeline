好的，我展开详细说。

---

## 时间处理完整思路

### 一、核心原则

**记忆文本中保持原始时间表达，绝对不转换。** 转换放在检索时做。

原因：
- 模型不会算错（算错是必然的）
- 时间不会漂移（不会因为新session重新计算旧记忆）
- SFT数据更自然（模型学习人类说话方式）

---

### 二、三个层次的处理

| 层次 | 做什么 | 谁做 |
|------|--------|------|
| 提取 | 保留原始表达，如"yesterday"、"last week" | 模型 |
| 存储 | 添加anchor_time（当前session_time） | 代码 |
| 检索 | 动态计算绝对日期 | 代码 |

---

### 三、提取层（模型负责）

模型从对话中提取时间时，只做"复制粘贴"：

| 对话原文 | 模型输出 |
|----------|----------|
| "I went yesterday" | "Caroline attended a support group yesterday." |
| "Last Saturday I ran a race" | "Melanie ran a charity race last Saturday." |
| "I'm going next month" | "Caroline plans to attend a conference next month." |
| "On May 7th" | "Caroline attended a support group on May 7th." |

**特殊处理**：如果对话中没有时间词，就不加时间。

---

### 四、存储层（代码负责）

模型输出后，代码自动添加锚点：

```json
{
  "id": "0",
  "text": "Caroline attended a support group yesterday.",
  "anchor_time": "2023-05-08"
}
```

锚点永远不变，不会因为后续session而更新。

---

### 五、检索层（代码负责）

当用户问"具体哪天"时，动态计算：

**计算公式**：
```
绝对时间 = 锚点时间 + 时间表达偏移量
```

**示例**：

| 记忆文本 | 锚点 | 相对表达 | 计算结果 |
|----------|------|----------|----------|
| "...yesterday" | 2023-05-08 | yesterday | 2023-05-07 |
| "...last Saturday" | 2023-05-25 | last Saturday | 2023-05-20 |
| "...next month" | 2023-06-09 | next month | 2023-07 |

**注意**：`next month`这类表达，即使计算也是模糊的（可能是7月1日或7月9日），系统需要决定粒度（月份级）。

---

### 六、UPDATE决策中的时间规则

决策时，时间信息用于判断新旧记忆的关系：

| 情况 | 旧记忆 | 新事实 | 决策 |
|------|--------|--------|------|
| 补充时间 | "Caroline attended a group." | "...attended a group yesterday." | UPDATE |
| 时间细化 | "...last week" | "...last Saturday" | UPDATE |
| 时间冲突（同一事件） | "...yesterday (锚点A)" | "...yesterday (锚点B)" | ADD（新事件） |
| 计划改变 | "plans to go next month" | "plans to go this week" | UPDATE |

**关键**：判断是否同一事件时，不能只看时间。需要结合主语、动词、宾语一起判断。

---

### 七、绝对时间的处理

如果对话中明确说出绝对日期（如"on May 7th, 2023"）：

| 方式 | 做法 | 示例 |
|------|------|------|
| 保留原文 | 不转换 | "...on May 7th, 2023" |
| 添加绝对标记 | 同时存储绝对日期 | `absolute_time: "2023-05-07"` |

两种方式都可以，推荐方式一（保持原文），因为SFT数据更自然。

---

### 八、无时间信息的处理

```json
{
  "id": "0",
  "text": "Caroline plans to pursue education in counseling.",
  "anchor_time": "2023-05-08"  // 仍然记录锚点，表示这个信息是在这个时间点说的
}
```

锚点仍然有用，表示"这条信息是何时被提及的"。

---

### 九、汇总：完整的数据流

```
对话: "I went to a support group yesterday" (session_time: 2023-05-08)
                    │
                    ▼
模型提取: "Caroline attended a support group yesterday."
                    │
                    ▼
代码存储: {
  "id": "0",
  "text": "Caroline attended a support group yesterday.",
  "anchor_time": "2023-05-08"
}
                    │
                    ▼
用户问: "Caroline什么时候去的支持组？"
                    │
                    ▼
检索到记忆0 → 解析"yesterday" + 锚点2023-05-08 → 计算得2023-05-07
                    │
                    ▼
回答: "Caroline在2023年5月7日去的。"
```

---

### 十、一句话总结

> **模型只负责复制时间词，存储时钉上锚点，检索时动态换算。**

这样模型不用算、不会错、不漂移，时间处理完全交给代码。