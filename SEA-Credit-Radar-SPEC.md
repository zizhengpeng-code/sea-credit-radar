# SEA Credit Radar — 项目规格书

> 给 Claude Code 的构建说明。把本文件放进空文件夹，在该目录启动 Claude Code，输入：
> 「阅读 SEA-Credit-Radar-SPEC.md，按规格实现整个项目，分阶段提交，每阶段跑通测试后再进行下一阶段。」

## 1. 这是什么

一个每日运行的东南亚消费信贷情报 Agent。它监测印尼、菲律宾、新加坡、越南的数字信贷与监管动态，把非结构化新闻抽取成结构化记录，跨天追踪同一事件的进展，并按「对信贷业务哪个环节有影响」分类推送。

**与常见新闻摘要工作流的区别（这几点是本项目的重点，不可省略）**
1. **结构化抽取**：不只生成摘要，而是抽出国家、机构、事件类型、影响环节（获客 / 准入 / 额度 / 定价 / 贷后）、生效日期等字段
2. **状态记忆**：持久化已处理事件，新新闻若是旧事件的后续，标记为「进展更新」并附上上次结论
3. **分支处理**：监管类事件自动抓取全文做深度解读；其他类型只做一句话摘要，控制成本
4. **效果评估**：有一套人工标注的测试集，可量化分类准确率与漏报率，并记录 prompt 迭代前后的变化
5. **成本可见**：每次运行记录 token 消耗与费用

## 2. 技术栈

- Python 3.11 以上，不依赖 n8n（纯代码版更能体现工程能力；如需可视化，后续再导出 n8n 版本）
- `feedparser` 抓 RSS，`httpx` 发请求，`trafilatura` 提取正文
- `openai` SDK 调用大模型（走 OpenAI 兼容接口，默认 DeepSeek，可切换）
- `sqlite3` 持久化，`pydantic` 定义数据模型
- `pytest` 测试，`rich` 终端输出
- 密钥从环境变量读取，仓库内绝不出现任何 key

### 模型调用要求（重要）

所有大模型调用必须封装在 `radar/llm.py` 的统一 client 里，**不得在业务代码中直接调用任何厂商 SDK**。通过环境变量配置：

```
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=...
LLM_MODEL=deepseek-chat
```

支持的供应商（均兼容 OpenAI 接口格式，只需改这三个变量）：

| 供应商 | base_url | 模型示例 |
|---|---|---|
| DeepSeek（默认） | https://api.deepseek.com/v1 | deepseek-chat |
| 智谱 GLM | https://open.bigmodel.cn/api/paas/v4 | glm-4-flash |
| 通义千问 | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen-plus |
| Ollama 本地 | http://localhost:11434/v1 | qwen2.5:7b |
| Claude | 用 anthropic SDK，在 client 内部做适配分支 | claude-sonnet-4-6 |

client 需要提供：
- `complete(prompt, system, schema) -> dict`：统一入口，负责 JSON 解析与 pydantic 校验，失败重试一次
- 自动记录每次调用的输入输出 token 数与估算成本，写入 `runs` 表
- 超时、限流（429）时指数退避重试，最多三次
- 一个 `--model` 命令行参数可临时覆盖环境变量，便于模型对比

## 3. 目录结构

```
sea-credit-radar/
├── README.md
├── requirements.txt
├── .env.example
├── config.yaml              # 数据源、关注国家、推送方式
├── radar/
│   ├── fetch.py             # 抓取与正文提取
│   ├── dedupe.py            # 去重与事件归并
│   ├── llm.py               # 统一模型 client，多供应商切换
│   ├── extract.py           # 大模型结构化抽取
│   ├── enrich.py            # 监管类事件的深度解读分支
│   ├── store.py             # SQLite 读写
│   ├── digest.py            # 生成每日简报
│   ├── notify.py            # 推送
│   └── cli.py               # 命令行入口
├── eval/
│   ├── testset.jsonl        # 人工标注样本
│   └── run_eval.py          # 评估脚本
└── tests/
```

## 4. 数据源

Google News RSS，按国家与主题组合查询，无需 API key。示例：

```
https://news.google.com/rss/search?q=digital+lending+Indonesia+OJK&hl=en&gl=ID&ceid=ID:en
https://news.google.com/rss/search?q=BNPL+Philippines+BSP&hl=en&gl=PH&ceid=PH:en
https://news.google.com/rss/search?q=fintech+regulation+MAS+Singapore&hl=en-SG&gl=SG&ceid=SG:en
```

全部查询写在 config.yaml 里，可增删。另加监管机构官网 RSS（OJK、BSP、MAS）作为高可信来源，并在配置中标注来源等级。

## 5. 数据模型

```python
class Event(BaseModel):
    id: str                    # 事件指纹
    first_seen: date
    last_updated: date
    country: Literal["ID","PH","SG","VN","OTHER"]
    category: Literal["regulation","product","funding","market","risk"]
    entities: list[str]        # 涉及的机构或公司
    headline_cn: str           # 中文一句话摘要，30 字以内
    why_it_matters: str        # 为什么重要，40 字以内
    impact_stage: list[Literal["acquisition","underwriting","limit","pricing","collection"]]
    effective_date: date | None
    confidence: float          # 模型自评 0-1
    sources: list[Source]      # 多条新闻可指向同一事件
    updates: list[Update]      # 后续进展
```

## 6. 处理流程

1. **fetch**：并发抓取所有 RSS，保留 48 小时内条目
2. **dedupe**：URL 精确去重；标题用 embedding 或 SimHash 做近似去重；与库中既有事件比对，相似度超阈值则归并为同一事件的新来源
3. **extract**：把候选条目分批交给模型，要求严格输出 JSON（用 pydantic 校验，失败时带错误信息重试一次，仍失败则记录并跳过）
4. **enrich（分支）**：`category == "regulation"` 或 `confidence < 0.6` 时，抓取正文做深度解读，输出政策要点、生效时间、对业务的具体影响；其余跳过此步
5. **store**：写入 SQLite，既有事件则追加 update 记录
6. **digest**：生成当日简报，新事件与进展更新分区展示，按影响程度排序
7. **notify**：推送到 Server 酱（微信）或邮箱，由 config.yaml 决定

## 7. Prompt 要求

- 系统提示中写明角色：东南亚消费信贷市场分析师，服务对象是信贷策略与产品团队
- 明确输出 JSON schema，禁止任何前言和 Markdown 代码块
- 要求模型对不确定的字段给低 confidence，而不是编造
- 所有 prompt 集中放在 `radar/prompts.py`，带版本号，便于 A/B 对比
- 不依赖任何厂商特有能力（如 Anthropic 的 tool use 或 OpenAI 的 function calling），只用纯文本 JSON 输出，保证换模型即可用

## 8. 评估（必须实现）

- `eval/testset.jsonl`：至少 40 条人工标注样本，标注 category、impact_stage、是否应被推送
- `run_eval.py` 输出：分类准确率、各类别的精确率与召回率、漏报率（应推送但被过滤掉的比例）、平均每条成本
- **支持 `--model` 参数跑同一套测试集对比不同模型**，结果汇总成一张表：模型、准确率、漏报率、平均延迟、每千条成本。这张对比表是项目的核心产出之一
- 结果写入 `eval/results/` 并带时间戳与 prompt 版本，方便展示迭代过程

## 9. 运行方式

- `python -m radar.cli run --dry-run`：跑全流程但不推送，终端打印结果
- `python -m radar.cli run`：正式运行并推送
- `python -m radar.cli backfill --days 7`：回填历史
- GitHub Actions 定时任务每天 08:00 SGT 运行一次，密钥存 Secrets

## 10. 验收标准

1. `pytest` 全部通过，fetch、dedupe、extract 都有单元测试，模型调用用 mock
2. `--dry-run` 能完整跑通并输出格式正确的简报
3. 连续运行两天后，第二天的简报中能正确出现「进展更新」区块
4. 评估脚本能输出完整指标
5. 至少用两个不同供应商的模型各跑一次完整评估，README 中给出对比表与选型结论
6. README 包含架构图、示例输出截图、评估结果表、模型对比表、成本数据，以及一段「设计取舍」说明：为什么分支处理而不是全部深度解读、去重阈值怎么定、confidence 低时怎么处理、为什么最终选这个模型

## 11. 分阶段实现

- 阶段一：fetch + dedupe + SQLite，能存入原始条目
- 阶段二：llm.py 统一 client + extract，结构化抽取并通过校验（先用 DeepSeek 跑通）
- 阶段三：enrich 分支与事件归并、进展追踪
- 阶段四：digest 与 notify
- 阶段五：评估脚本与测试集，并完成至少两个模型的对比评估
- 阶段六：GitHub Actions、README、示例输出

每阶段结束时运行测试并提交一次 commit，commit message 写清楚做了什么。
