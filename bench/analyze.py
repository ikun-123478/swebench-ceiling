"""Aggregate a benchmark run into per-instance metrics + an attribution report.

Reads the trajectories written by ``mini-extra swebench`` (they already store the
full provider ``usage`` and ``finish_reason`` for every call), so no agent change
is needed to get token-level evidence.

Usage:
    python bench/analyze.py --run-dir swebench_runs/ceiling_flash \
        --sample bench/sample.json --out-dir bench/out [--eval-results eval.json]
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

CONTEXT_WINDOW = 1_048_576  # measured from the API: "maximum context length is 1048576 tokens"
STEP_LIMIT = 250            # from bench/verified_deepseek.yaml


def _load_traj(run_dir: Path, iid: str) -> dict | None:
    p = run_dir / iid / f"{iid}.traj.json"
    return json.loads(p.read_text()) if p.exists() else None


def metrics_for(iid: str, traj: dict | None, pred: dict, step_limit: int = STEP_LIMIT) -> dict:
    patch = (pred or {}).get("model_patch") or ""
    row = {
        "instance_id": iid,
        "exit_status": (traj or {}).get("info", {}).get("exit_status", "MISSING_TRAJ"),
        "n_calls": 0, "patch_chars": len(patch), "patch_empty": len(patch.strip()) == 0,
        "prompt_tokens_total": 0, "completion_tokens_total": 0, "reasoning_tokens_total": 0,
        "peak_prompt_tokens": 0, "ctx_util": 0.0,
        "n_tool_calls": 0, "n_parallel_responses": 0, "n_length_truncations": 0,
        "n_format_errors": 0, "finish_reasons": {}, "wall_seconds": 0.0,
        "hit_step_limit": False, "reasoning_share": 0.0,
    }
    if not traj:
        return row
    msgs = traj.get("messages", [])
    ts = [m.get("extra", {}).get("timestamp") for m in msgs if m.get("extra", {}).get("timestamp")]
    row["wall_seconds"] = round(max(ts) - min(ts), 1) if len(ts) >= 2 else 0.0

    reasons, n_calls = Counter(), 0
    for m in msgs:
        extra = m.get("extra", {})
        if extra.get("interrupt_type") == "FormatError":
            row["n_format_errors"] += 1
        resp = extra.get("response") or {}
        usage = resp.get("usage") or {}
        if not usage:
            continue
        n_calls += 1
        pt = usage.get("prompt_tokens", 0) or 0
        ct = usage.get("completion_tokens", 0) or 0
        rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        row["prompt_tokens_total"] += pt
        row["completion_tokens_total"] += ct
        row["reasoning_tokens_total"] += rt
        row["peak_prompt_tokens"] = max(row["peak_prompt_tokens"], pt)
        fr = (resp.get("choices") or [{}])[0].get("finish_reason")
        reasons[fr] += 1
        if fr == "length":
            row["n_length_truncations"] += 1
        n_actions = len(extra.get("actions") or [])
        row["n_tool_calls"] += n_actions
        if n_actions > 1:
            row["n_parallel_responses"] += 1

    row["n_calls"] = n_calls
    row["ctx_util"] = round(row["peak_prompt_tokens"] / CONTEXT_WINDOW, 4)
    row["finish_reasons"] = dict(reasons)
    row["hit_step_limit"] = n_calls >= step_limit
    row["reasoning_share"] = round(row["reasoning_tokens_total"] / row["completion_tokens_total"], 3) if row["completion_tokens_total"] else 0.0
    return row


def aggregate(rows: list[dict], resolved: dict[str, bool] | None) -> dict:
    n = len(rows)
    submitted = [r for r in rows if r["exit_status"] == "Submitted"]
    nonempty = [r for r in submitted if not r["patch_empty"]]
    agg = {
        "n_instances": n,
        "submitted": len(submitted), "submitted_rate": round(len(submitted) / n, 3),
        "nonempty_patch": len(nonempty), "nonempty_patch_rate": round(len(nonempty) / n, 3),
        "exit_status": dict(Counter(r["exit_status"] for r in rows)),
        "median_calls": sorted(r["n_calls"] for r in rows)[n // 2] if n else 0,
        "max_calls": max((r["n_calls"] for r in rows), default=0),
        "hit_step_limit": sum(1 for r in rows if r.get("hit_step_limit")),
        "peak_prompt_tokens": max((r["peak_prompt_tokens"] for r in rows), default=0),
        "max_ctx_util": max((r["ctx_util"] for r in rows), default=0.0),
        "total_prompt_tokens": sum(r["prompt_tokens_total"] for r in rows),
        "total_completion_tokens": sum(r["completion_tokens_total"] for r in rows),
        "total_reasoning_tokens": sum(r["reasoning_tokens_total"] for r in rows),
        "length_truncations": sum(r["n_length_truncations"] for r in rows),
        "format_errors": sum(r["n_format_errors"] for r in rows),
        "parallel_responses": sum(r["n_parallel_responses"] for r in rows),
        "median_wall_seconds": sorted(r["wall_seconds"] for r in rows)[n // 2] if n else 0,
    }
    agg["reasoning_share"] = round(agg["total_reasoning_tokens"] / agg["total_completion_tokens"], 3) if agg["total_completion_tokens"] else 0.0
    if resolved is not None:
        got = [i for i in resolved if resolved[i] is not None]
        agg["resolved"] = sum(1 for i in got if resolved[i])
        agg["resolved_evaluated"] = len(got)
        agg["resolve_rate"] = round(agg["resolved"] / len(got), 3) if got else None
        sub_ids = {r["instance_id"] for r in submitted}
        got_sub = [i for i in got if i in sub_ids]
        agg["resolve_rate_given_submitted"] = round(sum(resolved[i] for i in got_sub) / len(got_sub), 3) if got_sub else None
    return agg


def attribution(rows: list[dict], agg: dict) -> list[str]:
    out = []
    out.append("**显存（VRAM）**：不适用。被测模型是 DeepSeek 托管 API，推理不发生在本地，"
               "本地显存与结果无关（本机也没有加载任何模型权重）。")
    cu = agg["max_ctx_util"]
    verdict = "不是瓶颈" if cu < 0.5 else ("接近上限" if cu < 0.8 else "可能受限")
    out.append(f"**上下文窗口**：窗口 {CONTEXT_WINDOW:,} tokens；实测峰值 {agg['peak_prompt_tokens']:,} tokens，"
               f"占窗口 **{cu:.1%}** → {verdict}。")
    if agg["length_truncations"] == 0:
        out.append(f"**输出/推理链长度**：全部 {agg['total_completion_tokens']:,} 个输出 token 中，"
                   f"finish_reason=length 出现 **0** 次；推理 token 占输出 {agg['reasoning_share']:.1%}。"
                   "→ 未被输出上限截断。")
    else:
        out.append(f"**输出/推理链长度**：出现 {agg['length_truncations']} 次 finish_reason=length。"
                   f"推理 token 占输出 {agg['reasoning_share']:.1%} → 需要检查这些是输出上限还是 provider 截断。")
    out.append(f"**步数预算**：{agg['hit_step_limit']}/{agg['n_instances']} 个实例走满 {STEP_LIMIT} 步；"
               f"步数中位数 {agg['median_calls']}，最大 {agg['max_calls']}。")
    out.append(f"**格式化/工具调用**：FormatError {agg['format_errors']} 次；"
               f"一次响应发多个 tool call 的情况 {agg['parallel_responses']} 次。")
    if agg.get("resolve_rate") is not None:
        out.append(f"**模型能力**：resolve {agg['resolved']}/{agg['resolved_evaluated']} = "
                   f"**{agg['resolve_rate']:.1%}**（在已提交且补丁非空的实例上 "
                   f"{agg.get('resolve_rate_given_submitted')}）。")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sample", default="bench/sample.json")
    ap.add_argument("--out-dir", default="bench/out")
    ap.add_argument("--eval-results", default="")
    args = ap.parse_args()

    run_dir, out_dir = Path(args.run_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = json.loads((run_dir / "preds.json").read_text()) if (run_dir / "preds.json").exists() else {}

    sample = json.loads(Path(args.sample).read_text()) if Path(args.sample).exists() else {"rows": [], "instance_ids": []}
    ids = [r["instance_id"] for r in sample.get("rows", [])] or sorted(preds)
    repo_of = {r["instance_id"]: r["repo"] for r in sample.get("rows", [])}

    resolved = None
    if args.eval_results:
        resolved = json.loads(Path(args.eval_results).read_text())

    rows = []
    for iid in ids:
        r = metrics_for(iid, _load_traj(run_dir, iid), preds.get(iid, {}))
        r["repo"] = repo_of.get(iid, "")
        if resolved is not None:
            r["resolved"] = resolved.get(iid)
        rows.append(r)

    agg = aggregate(rows, resolved)
    payload = {"meta": {"run_dir": str(run_dir), "context_window": CONTEXT_WINDOW, "step_limit": STEP_LIMIT},
               "aggregate": agg, "per_instance": rows}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))

    cols = ["instance_id", "repo", "exit_status", "n_calls", "patch_chars", "patch_empty",
            "peak_prompt_tokens", "ctx_util", "completion_tokens_total", "reasoning_tokens_total",
            "n_length_truncations", "n_format_errors", "n_parallel_responses", "wall_seconds"]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols + (["resolved"] if resolved is not None else []), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = ["# Benchmark result summary", "", f"- run dir: `{run_dir}`", "",
             "## Aggregate", "", "| metric | value |", "|---|---|"]
    for k, v in agg.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Per instance", "",
              "| instance | exit | steps | patch | peak_ctx | ctx% | reason_tok | len_trunc |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['instance_id']} | {r['exit_status']} | {r['n_calls']} | {r['patch_chars']} | "
                     f"{r['peak_prompt_tokens']} | {r['ctx_util']:.1%} | {r['reasoning_tokens_total']} | "
                     f"{r['n_length_truncations']} |")
    lines += ["", "## Attribution", ""] + [f"- {a}" for a in attribution(rows, agg)]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_dir}/metrics.json, metrics.csv, report.md")
    print(json.dumps(agg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
