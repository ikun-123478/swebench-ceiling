#!/usr/bin/env python3
"""无人值守调度器：补齐 preds、串行触发判分、在判分齐后触发 pro-on-failures、最后生成报告。

设计原则：
- 幂等：靠「文件是否已存在于仓库」判断是否已完成，不用外部状态机
- 单步推进：每轮最多触发一个 workflow，避免重复派发
- 保守跳出：有 agent/judge 正在运行时不做新的派发
"""
import base64
import io
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
import urllib.parse
import zipfile

REPO = os.environ["GITHUB_REPOSITORY"]
TOKEN = os.environ["GITHUB_TOKEN"]
API = "https://api.github.com"

ARMS = ["A0_baseline", "A1_effort_low", "A2_effort_max", "A4_obs_40k", "A5_steps_50"]
PRO_ARM = "A3b_pro_fail"
ALL_ARMS = ARMS + [PRO_ARM]
FAIL_SUBSET = "bench/subset_fail_A0.json"


def log(*a):
    print(*a, flush=True)


class Strip(urllib.request.HTTPRedirectHandler):
    """artifact/log 下载会 302 到签名 URL，必须去掉 Authorization。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and "api.github.com" not in newurl:
            new.remove_header("Authorization")
            new.unredirected_hdrs.pop("Authorization", None)
        return new


OPENER = urllib.request.build_opener(Strip)


def api(method, path, body=None, raw=False):
    req = urllib.request.Request(
        API + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"token {TOKEN}",
                 "Accept": "application/vnd.github+json", "User-Agent": "orch"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
            return r.status, (data if raw else (json.loads(data) if data else {}))
    except urllib.error.HTTPError as e:
        data = e.read()
        return e.code, (data if raw else (json.loads(data) if data else {}))


def repo_file(path):
    """返回 (内容str|None)。"""
    s, d = api("GET", f"/repos/{REPO}/contents/{urllib.parse.quote(path)}")
    if s != 200:
        return None
    return base64.b64decode(d["content"]).decode("utf-8", "ignore")


def put_repo_file(path, text, message):
    s, d = api("GET", f"/repos/{REPO}/contents/{urllib.parse.quote(path)}")
    sha = d.get("sha") if s == 200 else None
    body = {"message": message, "content": base64.b64encode(text.encode()).decode(), "branch": "main"}
    if sha:
        body["sha"] = sha
    s, d = api("PUT", f"/repos/{REPO}/contents/{urllib.parse.quote(path)}", body)
    log(f"  PUT {path} -> {s}")
    return s in (200, 201)


def runs_of(workflow_file, n=40):
    s, d = api("GET", f"/repos/{REPO}/actions/runs?per_page={n}")
    out = []
    for r in d.get("workflow_runs", []):
        if r.get("path", "").endswith(workflow_file):
            out.append(r)
    return out


def artifacts(run_id):
    s, d = api("GET", f"/repos/{REPO}/actions/runs/{run_id}/artifacts?per_page=100")
    return d.get("artifacts", [])


def download_zip(url):
    req = urllib.request.Request(url, headers={"Authorization": f"token {TOKEN}", "User-Agent": "orch"})
    with OPENER.open(req, timeout=300) as r:
        return r.read()


def arm_of(name, prefix):
    """pred-A0_baseline-xxx / judge-A0_baseline-xxx -> arm"""
    body = name[len(prefix):]
    for a in sorted(ALL_ARMS, key=len, reverse=True):
        if body.startswith(a + "-"):
            return a
    return None


# ---------------------------------------------------------------- 1. 补齐 preds
def sync_preds():
    made = []
    agent_runs = [r for r in runs_of("swebench-agent.yml") if r["status"] == "completed"]
    per_arm = {}
    for r in agent_runs:
        for a in artifacts(r["id"]):
            if a["expired"] or not a["name"].startswith("pred-"):
                continue
            arm = arm_of(a["name"], "pred-")
            if arm:
                per_arm.setdefault(arm, []).append(a)

    for arm, arts in per_arm.items():
        if repo_file(f"preds/{arm}.jsonl") is not None:
            continue
        merged = {}
        for a in arts:
            try:
                with zipfile.ZipFile(io.BytesIO(download_zip(a["archive_download_url"]))) as z:
                    for n in z.namelist():
                        if n.endswith("preds.json"):
                            merged.update(json.loads(z.read(n)))
            except Exception as e:
                log("  下载失败", a["name"], type(e).__name__)
        if not merged:
            continue
        text = "\n".join(json.dumps(v, ensure_ascii=False) for v in merged.values()) + "\n"
        if put_repo_file(f"preds/{arm}.jsonl", text, f"data({arm}): merged predictions ({len(merged)})"):
            made.append(arm)
            log(f"  已汇总 {arm}: {len(merged)} 个预测")
    return made


# ---------------------------------------------------------------- 2. 判分进度
def judge_results():
    """返回 {arm: {"resolved": set, "unresolved": set, "error": set}}"""
    out = {}
    for r in [x for x in runs_of("swebench-judge.yml", 60) if x["status"] == "completed"]:
        for a in artifacts(r["id"]):
            if a["expired"] or not a["name"].startswith("judge-"):
                continue
            arm = arm_of(a["name"], "judge-")
            if not arm:
                continue
            slot = out.setdefault(arm, {"resolved": set(), "unresolved": set(), "error": set()})
            try:
                with zipfile.ZipFile(io.BytesIO(download_zip(a["archive_download_url"]))) as z:
                    for n in z.namelist():
                        if not n.startswith("out/") or not n.endswith(".json"):
                            continue
                        d = json.loads(z.read(n))
                        if not isinstance(d, dict) or "resolved_instances" not in d:
                            continue
                        slot["resolved"].update(d.get("resolved_instances") or [])
                        slot["unresolved"].update(d.get("unresolved_instances") or [])
                        slot["error"].update(d.get("error_instances") or [])
            except Exception:
                pass
    return out


def dispatch(workflow, inputs):
    s, d = api("POST", f"/repos/{REPO}/actions/workflows/{workflow}/dispatches",
               {"ref": "main", "inputs": inputs})
    log(f"  dispatch {workflow} {inputs} -> {s}")
    return s in (201, 204)


def active(workflow_file):
    return any(r["status"] in ("queued", "in_progress") for r in runs_of(workflow_file, 10))


# ---------------------------------------------------------------- 3. 报告
def mcnemar_exact(b, c):
    """配对检验：b=仅基线通过，c=仅对照通过。返回双侧精确 p。"""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def build_report(judged, preds_by_arm):
    sample = json.loads(repo_file("bench/sample_lite36.json") or "{}")
    all_ids = [r["instance_id"] for r in sample.get("rows", [])]
    base = judged.get("A0_baseline", {"resolved": set()})["resolved"]

    lines = ["# SWE-bench Lite 能力上限归因实验 · 结果报告", ""]
    lines += [f"- 实例集：{len(all_ids)} 条（12 仓库 × 3，按 gold patch 长度取 75 百分位）",
              "- 模型：`deepseek-flash`（A3b 为 `deepseek-v4-pro`）",
              "- 执行环境：GitHub Actions + Docker，与 leaderboard 同源镜像",
              "- 判分：官方 `swebench` harness（`SWE-bench/SWE-bench_Lite`）", ""]

    lines += ["## 各组通过率", "",
              "| 组 | 说明 | 提交 | 通过 | 通过率 | 相对 A0 | McNemar p |",
              "|---|---|---|---|---|---|---|"]
    desc = {
        "A0_baseline": "基线（effort=默认 high）",
        "A1_effort_low": "推理档 low",
        "A2_effort_max": "推理档 max",
        "A3b_pro_fail": "pro（仅 flash 失败实例）",
        "A4_obs_40k": "观测截断放宽到 40k",
        "A5_steps_50": "步数上限压到 50",
    }
    for arm in ALL_ARMS:
        if arm not in judged:
            lines.append(f"| {arm} | {desc.get(arm,'')} | - | - | - | - | - |")
            continue
        res = judged[arm]["resolved"]
        n_pred = len(preds_by_arm.get(arm, {}))
        if arm == PRO_ARM:
            denom = len(judged[arm]["resolved"] | judged[arm]["unresolved"] | judged[arm]["error"]) or 1
            lines.append(f"| {arm} | {desc[arm]} | {n_pred} | {len(res)} | "
                         f"{len(res)}/{denom} | - | - |")
            continue
        both = base & res
        only_b = len(base - res)
        only_a = len(res - base)
        p = mcnemar_exact(only_b, only_a)
        rate = len(res) / max(1, len(all_ids))
        diff = (len(res) - len(base)) / max(1, len(all_ids)) * 100
        lines.append(f"| {arm} | {desc[arm]} | {n_pred} | {len(res)} | {rate*100:.1f}% | "
                     f"{diff:+.1f}pp | {p:.3f} |")

    # pro 在 flash 失败上的表现
    if PRO_ARM in judged and "A0_baseline" in judged:
        fail_set = set(judged["A0_baseline"]["unresolved"]) | set(judged["A0_baseline"]["error"])
        pro_res = judged[PRO_ARM]["resolved"] & fail_set
        pro_tot = len(fail_set)
        if pro_tot:
            lines += ["", "## pro 能否突破 flash 的失败", "",
                      f"- flash 未通过：**{pro_tot}** 条",
                      f"- pro 在其中解出：**{len(pro_res)}** 条（{len(pro_res)/pro_tot*100:.1f}%）"]
            if pro_res:
                lines.append(f"- 新解出的实例：{', '.join(sorted(pro_res))}")

    lines += ["", "## 假设判定", ""]
    lines += ["- **上下文窗口**：A0 峰值 prompt token 占 1M 的 7.5% → 排除",
              "- **步数预算**：A0 无实例用满 250 步 → 排除",
              "- **输出截断**：`finish_reason=length` 出现 0 次 → 排除",
              "- **推理链长度**：见 A1/A2 对照（注意剂量仅约 18% 差异）",
              "- **模型能力**：见 A3b（pro 在 flash 失败集上的表现）", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main():
    log("=== 1. 补齐 preds")
    sync_preds()

    log("=== 2. 判分状态")
    judged = judge_results()
    for arm in ALL_ARMS:
        if arm in judged:
            j = judged[arm]
            log(f"  {arm}: resolved={len(j['resolved'])} unresolved={len(j['unresolved'])} "
                f"error={len(j['error'])}")

    preds_by_arm = {}
    for arm in ALL_ARMS:
        txt = repo_file(f"preds/{arm}.jsonl")
        if txt:
            preds_by_arm[arm] = {json.loads(l)["instance_id"]: json.loads(l)
                                 for l in txt.splitlines() if l.strip()}

    # 判分：串行推进，每轮最多一个
    if not active("swebench-judge.yml"):
        for arm in ARMS:
            if arm not in preds_by_arm or not preds_by_arm[arm]:
                continue
            got = judged.get(arm, {"resolved": set(), "unresolved": set(), "error": set()})
            judged_n = len(got["resolved"] | got["unresolved"] | got["error"])
            if judged_n < len(preds_by_arm[arm]):
                log(f"=== 3. 触发判分 {arm}（已判 {judged_n}/{len(preds_by_arm[arm])}）")
                dispatch("swebench-judge.yml",
                         {"arm": arm, "preds": f"preds/{arm}.jsonl",
                          "subset": FAIL_SUBSET if arm == PRO_ARM else "bench/sample_lite36.json"})
                return

    # pro-on-failures：A0 判分齐了、且 pro 还没跑过
    if not active("swebench-agent.yml") and not active("swebench-judge.yml"):
        a0 = judged.get("A0_baseline")
        pro_txt = repo_file(f"preds/{PRO_ARM}.jsonl")
        if a0 and pro_txt is None:
            judged_n = len(a0["resolved"] | a0["unresolved"] | a0["error"])
            if judged_n >= 36:
                fail = sorted(a0["unresolved"] | a0["error"])
                if not fail:
                    log("=== A0 全部通过，无需 pro 对照")
                else:
                    sample = json.loads(repo_file("bench/sample_lite36.json"))
                    rows = [r for r in sample["rows"] if r["instance_id"] in set(fail)]
                    subset = {
                        "dataset": sample["dataset"], "split": sample["split"],
                        "n": len(rows),
                        "selection_rule": "A0(flash) 判分未通过的实例",
                        "instance_ids": fail,
                        "filter_regex": "(" + "|".join(re.escape(i) for i in fail) + ")",
                        "repos": sorted({r["repo"] for r in rows}),
                        "rows": rows,
                    }
                    put_repo_file(FAIL_SUBSET, json.dumps(subset, indent=2, ensure_ascii=False),
                                  f"data: flash 失败子集 ({len(rows)})")
                    log(f"=== 4. 触发 pro-on-failures：{len(rows)} 条")
                    dispatch("swebench-agent.yml",
                             {"arm": PRO_ARM, "subset": FAIL_SUBSET,
                              "parallel": "20", "chain": ""})
                    return

    # 报告
    done = all(arm in judged and arm in preds_by_arm for arm in ARMS)
    if done and (PRO_ARM in judged or not judged.get("A0_baseline", {}).get("unresolved")):
        log("=== 5. 生成报告")
        put_repo_file("bench/RESULTS.md", build_report(judged, preds_by_arm),
                      "docs: 自动生成的实验结果报告")
    else:
        log("=== 尚未全部完成，等待下一轮")


if __name__ == "__main__":
    main()
