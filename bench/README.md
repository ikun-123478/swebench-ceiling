# bench —— Agent 能力上限测评方案

目标：用 **DeepSeek flash** 驱动 mini-swe-agent，在**尽可能高强度**的设置下探测其能力上限，
并把结果拆解到「模型能力 / 上下文窗口 / 输出与推理链长度 / scaffold 步数预算」四个可量化维度。

> 说明：这是**托管 API** 模型，**不存在"本地显存"变量**。归因框架因此把 VRAM 显式排除，
> 并用实测数据替代猜测。

---

## 1. Benchmark 选择与评分标准

| 项 | 选择 |
|---|---|
| Benchmark | **SWE-bench Verified**（`princeton-nlp/SWE-Bench_Verified`，`test`，500 条） |
| 任务类型 | 仓库级真实 issue 修复：给 PR 描述，在 `/testbed` 里改**非测试**源码，产出 git patch |
| 为什么选它 | ① mini-swe-agent 的原生评测场景，scaffold 与 prompt 都是为它调过的；② Verified 是人工校验过的**最难子集**，最能压出能力上限；③ 有公开 leaderboard 可对照 |
| 官方评分 | `resolved` = 该实例的 **FAIL_TO_PASS 测试全部通过** 且 **PASS_TO_PASS 无回归**（由 SWE-bench harness 判定） |
| 主指标 | `resolve_rate = resolved / N` |
| 辅助指标 | `submitted_rate`、`nonempty_patch_rate`、`resolve_rate_given_submitted`、步数、token、`finish_reason` 分布 |

**注意**：`Verified` 只有 `test` split（运行器 `--split` 默认是 `dev`，必须显式写 `--split test`）。

---

## 2. 测试用例的组织与抽样

`bench/select_instances.py` 做**按仓库分层 + 难度加权**抽样：

- 按 `repo` 分组；
- 仓库内按 **gold patch 字符数**排序，取第 **75 百分位**那一条（比中位数难，但不取病态长尾）；
- 每个仓库 1 条，最多 12 条 → 12 个仓库各 1 条；
- 输出 `bench/sample.json`：实例清单、仓库、gold_patch 长度、F2P/P2P 数量、以及可直接喂给运行器的 `filter_regex`。

```bash
python bench/select_instances.py --dataset verified --per-repo 1 \
    --percentile 0.75 --max-instances 12 --out bench/sample.json
```

当前样本（12 实例 / 12 仓库）：astropy-8872、django-14752、matplotlib-25122、seaborn-3069、
flask-5014、requests-1142、xarray-7229、pylint-6386、pytest-6197、scikit-learn-25931、
sphinx-7757、sympy-19954。

---

## 3. 批量运行方式

用上游运行器（保证与公开结果同源），通过 `--filter` 精确圈定样本：

```bash
docker desktop start
cd C:/Users/86153/mini-swe-agent
# 凭证（.env 带 BOM，先提取再 export）
eval "$(python -c "from dotenv import dotenv_values as v;e=v('.env');print('export OPENAI_API_KEY=%s'%e['OPENAI_API_KEY']);print('export OPENAI_BASE_URL=%s'%e['OPENAI_BASE_URL'])")"
export MSWEA_SILENT_STARTUP=1 MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=3

FILTER=$(python -c "import json;print(json.load(open('bench/sample.json'))['filter_regex'])")
mini-extra swebench --subset verified --split test --filter "$FILTER" \
  -c swebench.yaml -c bench/verified_deepseek.yaml \
  -o swebench_runs/ceiling_flash -w 3
```

**高强度设置**（`bench/verified_deepseek.yaml`）：

| 参数 | 值 | 理由 |
|---|---|---|
| `agent.step_limit` | **250** | 上游默认；给足预算，观察是否走满 |
| `agent.cost_limit` | **0**（不限） | 上游是 3.0，长轨迹会被钱截断，掩盖真实能力 |
| `model.model_kwargs.temperature` | **0** | 降随机性 |
| `model.model_name` | `openai/deepseek-flash` | 端点 canonical 名 |
| `environment.timeout` | 60 | 与上游一致 |

**断点续跑**：运行器按 `preds.json` 跳过已完成实例，重跑同一条命令即可。
**磁盘**：每实例一个 ~2 GB 镜像；12 个约 24 GB。跑完用
`docker images | grep swebench | xargs docker rmi -f` 清理。

---

## 4. 结果记录格式

```
swebench_runs/ceiling_flash/
├── preds.json                     # instance_id -> {model_name_or_path, instance_id, model_patch}
├── minisweagent.log
├── exit_statuses_*.yaml
└── <instance_id>/<instance_id>.traj.json   # 完整轨迹（messages 内含每次调用的 usage/finish_reason）
```

`bench/analyze.py` 把它们汇总成：

| 文件 | 内容 |
|---|---|
| `bench/out/metrics.json` | `meta` + `aggregate` + `per_instance`（机器可读） |
| `bench/out/metrics.csv` | 每实例一行，便于贴到 Excel/论文表格 |
| `bench/out/report.md` | 聚合表 + 逐实例表 + 归因结论 |

```bash
python bench/analyze.py --run-dir swebench_runs/ceiling_flash \
    --sample bench/sample.json --out-dir bench/out [--eval-results bench/out/resolved.json]
```

---

## 5. 官方评分（把 preds.json 变成 resolved）

**方式 A（推荐）**：`sb-cli` 云评测（免费、快、不占本地磁盘，但需要 API key）：

```bash
sb-cli submit swe-bench_verified test --predictions_path preds.json --run_id <id>
```

**方式 B**：本地 harness（独立 venv，避免污染）：

```bash
# preds.json(obj) -> jsonl
python -c "import json;p=json.load(open('preds.json'));open('preds.jsonl','w').write('\n'.join(json.dumps(v) for v in p.values()))"
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Verified \
  --predictions_path preds.jsonl --max_workers 3 --run_id <id>
```

---

## 6. 归因方法：四个候选因素的证据来源

| 候选因素 | 数据来源 | 判定规则 |
|---|---|---|
| **显存 VRAM** | 无（托管 API） | 直接排除：本机不加载权重 |
| **上下文窗口** | 每次调用的 `usage.prompt_tokens` | `peak / 1,048,576`；<50% 即非瓶颈 |
| **输出 / 推理链长度** | `finish_reason`、`completion_tokens_details.reasoning_tokens` | `finish_reason=="length"` 出现次数为 0 即非瓶颈；并给出推理 token 占比 |
| **模型能力** | 官方 `resolved` | 在「已提交且补丁非空」的实例上的 resolve 率 |
| **scaffold 步数预算** | `n_calls` vs `step_limit=250` | 走满 250 步的实例数 |

模型硬指标（已实测，写死为对照基线）：

| 指标 | 实测值 |
|---|---|
| 上下文窗口 | **1,048,576** tokens |
| `max_tokens` 合法区间 | **[1, 393,216]** |
| 默认输出预算 | 实测一次生成 59,453 tokens 仍 `finish=stop`（无小上限） |
| 是否推理模型 | 是（每次调用均有 `reasoning_tokens`） |
