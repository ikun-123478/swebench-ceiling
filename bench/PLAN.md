# SWE-bench Lite 能力上限归因实验方案

目标：复现 mini-swe-agent，用 **DeepSeek Flash** 在 **SWE-bench Lite** 上跑出一个可解释的分数，
并回答一个问题：**这个分数是被模型能力卡住的，还是被显存 / 上下文窗口 / 推理链长度卡住的？**

---

## 0. 先回答你的三个问题

**（1）我的设备能跑吗？**
能，但**用服务器跑，不要用本机**。SWE-bench 评测是 CPU / 内存 / 磁盘 / 网络密集型，
和显卡无关；换服务器换掉的是 Windows 上那套 Docker Desktop / WSL / 网络限制，不是算力。

**（2）能得到我的结果吗？**
能。36 个实例 × 6 组对照 ≈ 半天跑完 + 半天判分，产出一张「每个假设 → 证据 → 判定」的归因表。

**（3）不用本地部署模型能测出来吗？**
你要问的四件事里，**三件能测，一件不能**：

| 你想测的 | 不用本地模型能不能测 | 说明 |
|---|---|---|
| 模型能力限制 | ✅ 能 | 换 `deepseek-v4-pro` 做对照 |
| 上下文窗口限制 | ✅ 能 | 峰值占用率 + `finish_reason=length` 统计 |
| 推理链长度限制 | ✅ 能 | `reasoning_effort` 三档消融（已验证有效） |
| **显存大小限制** | ❌ **测不了，而且不该测** | 见下 |

---

## 1. 三个会让实验白做的前提

### 1.1 「显存」在你的跑法里不是变量，是伪命题

DeepSeek Flash 是**托管 API**，权重在 DeepSeek 的机房，本地一个字节都没加载。
`nvidia-smi` 全程 0% 占用。所以「是不是显存不够」这个假设在你的实验条件下**没有对应的可观测量**。

硬要测显存，只有一条路：在 5090 上自己部署开源模型（见 §8）——但那测的是**另一个模型**，
得出来的结论不能反过来解释 DeepSeek Flash 的分数。

**要在论文/汇报里怎么写**：不是「显存不是瓶颈」，而是「显存被实验设计显式排除——被测模型为远程托管服务，
本地显存与结果无因果关系」。这两句话的分量完全不同。

### 1.2 单次 resolve 率不是归因

「Flash 在 Lite 上拿了 25%」不构成任何归因。归因需要**控制变量对照**：
只动一个因素、其余全部锁死，看结果变不变。所以本方案是 6 组 arm 而不是 1 次跑批。

### 1.3 真正的「上下文瓶颈」可能不是 1M 窗口，而是 10k 观测截断

模型窗口是 **1,048,576** tokens，而 mini-swe-agent 的观测模板里写死了：

```jinja
{% if output.output | length < 10000 %}
```

即单条命令输出超过 1 万字符，模型**只看到头 5000 + 尾 5000，中间全部 elided**。
一个 `pytest` 长回溯、一次 `grep -rn` 大结果，关键的那几行很可能就在被吞掉的中间。

所以「上下文」拆成两个可测假设：
- **H3a 窗口容量**：峰值 prompt tokens / 1M —— 大概率远低于 30%，直接排除；
- **H3b 观测截断**：把上限从 10k 放到 40k，看分数变不变 —— 这个才是真候选。

---

## 2. 实验设计

### 2.1 固定子集（配对设计）

所有 arm 跑**完全相同**的 36 条实例（Lite test，12 仓库 × 3 条，按 gold patch 的 75 百分位取难度偏高的）：

```bash
python bench/select_instances.py --dataset lite --per-repo 3 \
    --percentile 0.75 --max-instances 36 --out bench/sample_lite36.json
```

配对设计的价值：同一道题横向比较，个体难度差异被消掉，所需样本量比两组独立实验小得多。

### 2.2 六组 arm

| arm | 唯一变量 | 检验的假设 |
|---|---|---|
| **A0** baseline | flash + 默认推理档 + 观测 10k + 250 步 | 参照组 |
| **A1** effort=low | 推理预算↓ | H4 推理链 |
| **A2** effort=max | 推理预算↑ | H4 推理链 |
| **A3** model=v4-pro | 模型更强 | H1 模型能力 |
| **A4** obs=40k | 观测上限 10k→40k | H3b 观测截断 |
| **A5** steps=50 | 步数 250→50 | scaffold 预算（常见混杂因子） |

配置文件已生成在 `bench/arms/`，每个文件只写该 arm 的差异项（配置合并是深合并，不会丢其它键）。

### 2.3 为什么 `effort` 是干净的推理链旋钮

已实测（同题、`temperature=0`、`max_tokens=30000`）：

| effort | reasoning tokens | finish_reason |
|---|---|---|
| low | 3,277 | stop |
| max | 7,148 | stop |

两档都正常结束（没有截断），推理长度差 **2.2 倍**。所以「推理链长度」这个变量可以**主动操纵**，
而不是只能在事后观察 token 数。

> ⚠️ **效应量不稳定，正式跑前要先做 pilot**。同一道题在另一台机器上复测时，
> `low 5,371` vs `max 6,346`（仅 **1.18 倍**）。也就是说这个旋钮的分离度在 1.2×–2.2× 之间波动。
> A1/A2 的对照可能功率不足 → **正式铺开前，每档连测 3 次确认真实分离度**；
> 若分离度太小，「推理链长度不是瓶颈」这个结论就站不住（无效假设的阴性结果不能当证据）。

---

## 3. 判定规则（先定判据，再看数据）

| 假设 | 操作化 | 判据 |
|---|---|---|
| **H3a 窗口** | `peak_prompt_tokens / 1,048,576` | < 50% 且 `length` 截断 = 0 → 排除 |
| **H3b 观测截断** | A4 vs A0 配对 Δ | Δ 显著为正 → 是限制；否则排除 |
| **H4 推理链** | A2 vs A0、A1 vs A0 | A2 显著↑ → 推理预算是限制；A1≈A2≈A0 → 已饱和，不是限制 |
| **H1 模型能力** | A3 vs A0 | 显著提升 → 能力上限由模型决定（其余全部锁死） |
| **S 步数** | A5 vs A0、`hit_step_limit` 计数 | A5≈A0 且走满上限的实例很少 → 250 步是浪费 |

**最终归因逻辑**：看哪个干预能**显著且最大幅度**地抬高 resolve 率，上限就归到那个因素；
如果**所有干预都无效**，说明分数不是被这些预算卡住的——剩下的解释是 scaffold 本身
（工具粒度、无多路检索、无测试反馈闭环）与该模型在真实仓库上的固有能力。

判分只认官方口径：该实例 **FAIL_TO_PASS 全通过且 PASS_TO_PASS 无回归**（由 SWE-bench harness 判定），
`resolve_rate = resolved / N`。

---

## 4. 执行步骤（服务器上照抄）

> 全部命令按 **Linux** 写。你的 5090 服务器用不到 GPU，只在第 0 步确认一下机器身份。

```bash
# ── Step 0 环境自检 ────────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv   # 只确认机器，本实验不用
docker info | grep -E "Server Version|Total Memory|CPUs"
free -g && nproc
df -h /            # 需要 ≥ 250 GB 空闲
```

```bash
# ── Step 1 代码与依赖 ─────────────────────────────
cd ~/minisweagent

# 系统没有 pip、也没有 ensurepip 时（Ubuntu 裸机常见），这样绕开 sudo：
curl -sSL https://bootstrap.pypa.io/get-pip.py -o ~/get-pip.py
python3 ~/get-pip.py --user --break-system-packages        # pip 落到 ~/.local/bin
python3 -m venv --without-pip .venv                        # 不需要 python3.12-venv
~/.local/bin/pip --python .venv/bin/python install -e . -i https://mirrors.aliyun.com/pypi/simple

# ⚠️ 必须钉死 litellm：>=1.106 会在调用路径上 import mcp，报 ModuleNotFoundError: mcp
~/.local/bin/pip --python .venv/bin/python install "litellm==1.100.0" -i https://mirrors.aliyun.com/pypi/simple

# 国内服务器建议
export HF_ENDPOINT=https://hf-mirror.com
```

> 已知坑：`openai/deepseek-flash` 不在 litellm 的价格表里。配置必须带
> `cost_tracking: ignore_errors`（`lite_deepseek.yaml` 已有），否则直接抛 `RuntimeError`。

```bash
# ── Step 2 生成 36 条固定子集 ─────────────────────
python bench/select_instances.py --dataset lite --per-repo 3 \
    --percentile 0.75 --max-instances 36 --out bench/sample_lite36.json
# 期望输出：36 instances across 12 repos
```

```bash
# ── Step 3 预拉镜像（关键，别跳过）─────────────────
# 上游 pull_timeout 默认 120s，实例镜像 ~1GB，现拉必然超时 → 必须提前拉好
python - <<'PY' > /tmp/images.txt
import json
d = json.load(open("bench/sample_lite36.json"))
for r in d["rows"]:
    iid = r["instance_id"].replace("__", "_1776_").lower()
    print(f"docker.io/swebench/sweb.eval.x86_64.{iid}:latest")
PY

xargs -a /tmp/images.txt -P 4 -n 1 sh -c 'docker pull "$0" || echo "FAIL $0"' | tail -5

# 校验：应该等于 36
docker images --format '{{.Repository}}' | grep -c sweb.eval
```

```bash
# ── Step 4 冒烟（用官方 dummy 实例，几分钟）────────
export OPENAI_API_KEY=sk-xxxxxxxx          # 你的 key，别写进仓库
export OPENAI_BASE_URL=https://api.deepseek.com
export MSWEA_COST_TRACKING=ignore_errors MSWEA_SILENT_STARTUP=1

mini-extra swebench --subset _test --split test \
    -c swebench.yaml -c bench/lite_deepseek.yaml -w 1 \
    -o swebench_runs/L0_smoke
# 期望：exit_statuses_*.yaml 里是 Submitted
```

```bash
# ── Step 5 跑 6 个 arm ────────────────────────────
FILTER=$(python -c "import json;print(json.load(open('bench/sample_lite36.json'))['filter_regex'])")

for ARM in A0_baseline A1_effort_low A2_effort_max A3_model_pro A4_obs_40k A5_steps_50; do
  mini-extra swebench --subset lite --split test --filter "$FILTER" \
      -c swebench.yaml -c bench/lite_deepseek.yaml -c "bench/arms/$ARM.yaml" \
      -w 6 -o "swebench_runs/$ARM"
done
# 断点续跑：同一条命令重跑即可（按 preds.json 跳过已完成的实例）
```

```bash
# ── Step 6 官方判分 ───────────────────────────────
pip install swebench

for ARM in A0_baseline A1_effort_low A2_effort_max A3_model_pro A4_obs_40k A5_steps_50; do
  python -c "
import json,sys
p=json.load(open('swebench_runs/$ARM/preds.json'))
open('swebench_runs/$ARM/preds.jsonl','w').write('\n'.join(json.dumps(v) for v in p.values()))"
  python -m swebench.harness.run_evaluation \
      --dataset_name princeton-nlp/SWE-bench_Lite \
      --predictions_path "swebench_runs/$ARM/preds.jsonl" \
      --max_workers 6 --run_id "$ARM"
done
# 产出 swebench_runs/<ARM>/<ARM>.<model>.json （含 resolved_instances）
```

```bash
# ── Step 7 归因分析 ───────────────────────────────
python bench/compare.py --sample bench/sample_lite36.json \
  --arm A0=swebench_runs/A0_baseline \
  --arm A1=swebench_runs/A1_effort_low \
  --arm A2=swebench_runs/A2_effort_max \
  --arm A3=swebench_runs/A3_model_pro \
  --arm A4=swebench_runs/A4_obs_40k \
  --arm A5=swebench_runs/A5_steps_50 \
  --baseline A0 --steps A5=50 \
  --resolved A1=swebench_runs/A1_effort_low/A1_effort_low.deepseek-flash.json \
  --resolved A2=swebench_runs/A2_effort_max/A2_effort_max.deepseek-flash.json \
  --resolved A3=swebench_runs/A3_model_pro/A3_model_pro.deepseek-v4-pro.json \
  --resolved A4=swebench_runs/A4_obs_40k/A4_obs_40k.deepseek-flash.json \
  --resolved A5=swebench_runs/A5_steps_50/A5_steps_50.deepseek-flash.json \
  --out-dir bench/out/compare
# → bench/out/compare/comparison.md：各 arm 表 + 配对 McNemar + 逐假设判定
```

单次运行想看细节（步数 / token / 上下文峰值 / finish_reason / format error）：

```bash
python bench/analyze.py --run-dir swebench_runs/A0_baseline \
    --sample bench/sample_lite36.json --out-dir bench/out/A0
```

---

## 5. 预算

**时间**：单实例约 40 步 × 20 s ≈ 13 min；36 实例 `-w 6` ≈ 1.5 h/arm；6 个 arm ≈ **9 h**，
加判分（每个 arm 3–5 h，可并行）→ **一整天能出全部结果**。先跑 A0 一个 arm 验证节奏再铺开。

**磁盘**：36 个镜像，同 repo 实例共享 base 层，实测单镜像压缩后约 1.0 GB → **去重后约 25–35 GB**。
（Lite 全量 300 条才是 120–180 GB 级别，本方案不需要。）

**token / 费用**：不要拍脑袋，先做 `--filter` 只跑 **3 条**，读 `bench/out/metrics.json` 里的
`total_prompt_tokens` / `total_completion_tokens`，乘以你账号的单价再 ×12 外推。
注意 agent 轨迹是**累积上下文**，prompt token 随步数近似线性增长，单实例常在百万级。

**统计诚实**：n=36 时每个实例值 2.8pp，配对设计下约能可靠检出 **15–20pp** 量级的差异。
小于 10pp 的差异在这个样本量下不可信——想要 10pp 分辨率，子集要大到 100+ 条。
报告里务必写清「差异不显著」≠「没有差异」。

---

## 6. 结果会长什么样

`bench/out/compare/comparison.md` 的核心两张表：

| arm | resolved | resolve_rate | median_steps | peak_ctx% | length_trunc | reasoning_share |
|---|---|---|---|---|---|---|
| A0 | ?/36 | ? | ? | ? | ? | ? |
| A1 | | | | | | |
| … | | | | | | |

| arm | n_paired | 仅本组对 | 仅基线对 | Δ | McNemar p |
|---|---|---|---|---|---|
| A3 | 36 | ? | ? | ? | ? |

三种典型结论形态，含义完全不同：

1. **A3 显著↑，A1/A2/A4 无差异** → 能力上限在**模型**。上下文与推理预算都够用，
   换更强模型直接涨分 = 最强证据。
2. **A2 显著↑** → 上限在**推理链预算**，模型还有余量没被榨出来，应放宽 `effort` 或改 prompt 让模型多想。
3. **A4 显著↑** → 上限在**观测截断**，scaffold 在"饿着"模型。这是最容易修、也最容易被忽略的一条。
4. **全都不显著** → 不是这些预算的问题。要么 scaffold 本身（工具粒度、无多路检索、无测试闭环），
   要么该模型在这类真实仓库任务上的固有能力天花板。

---

## 7. 已踩过的坑（实测，照抄可绕开）

| 坑 | 现象 | 处理 |
|---|---|---|
| `reasoning_effort` 被静默丢弃 | 直接写进 `model_kwargs` 会抛 `UnsupportedParamsError`；`drop_params=True` 时连报错都没有，两组数据看起来"随机" | 必须放进 **`extra_body`**（见 `bench/arms/A1_*.yaml`） |
| 镜像拉取超时 | `docker run` 返回 125 / `TimeoutExpired` | `pull_timeout` 调 1800 + **提前 `docker pull`** |
| Docker Hub 拉不动 | 加速器清单能读、blob 重定向后 0 字节 | 换网络、走代理，或在国内云上用厂商镜像加速 |
| 判分包 | 本地判分需要 `swebench` harness（较重），也可用 `sb-cli` 云判分 | 服务器上直接 `pip install swebench` 最省事 |
| 数据集加载 | 需要联网 HuggingFace | `HF_ENDPOINT=https://hf-mirror.com` 或离线缓存 |
| Windows 上跑 | Docker Desktop/WSL 一堆坑、`mini` CLI 依赖真实控制台 | **直接用服务器，别在本机跑** |

---

## 8. 如果你确实想测「显存」

那必须本地部署，且要接受一个前提：**测的是另一个模型**。

在 5090（32 GB）上可行的做法：

- 模型：30B 级 @ 4bit（Qwen3-Coder-30B-A3B / Devstral-Small-24B 量级）
- 服务：vLLM 起 OpenAI 兼容端点；**5090 是 Blackwell，必须用 CUDA 12.8+ 的构建**
- 接线：`model_name: openai/<served-name>`、`model_kwargs.api_base: http://localhost:8000/v1`、`api_key: EMPTY`
- 对照维度：**4bit vs 8bit**（显存占用对照）、**max_model_len 32k vs 128k**（KV cache 对照）

还要注意：显存实际上并不"直接"影响能力，它只通过三个中介变量起作用 —— **能装多大的模型 / 用什么精度 / 能开多长上下文**。
所以真正的结论应该写成「在 32 GB 显存约束下，模型规模与上下文长度被限制，导致……」，
而不是「显存不够导致能力受限」。

**建议顺序**：先用 §4 把 API 路线跑通拿到归因结论，再决定要不要花时间做本地部署对照。
前者一天能出结果，后者光调 vLLM + Blackwell 兼容就能烧掉两天。
