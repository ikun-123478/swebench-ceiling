"""Compare several benchmark runs (arms) against a baseline and emit an attribution table.

Each arm is a run directory produced by `mini-extra swebench`. Arms are compared on the
*same* instance subset (paired design), so the right test is McNemar's exact test on the
discordant pairs, not a two-sample proportion test.

Resolved labels: drop a `resolved.json` (`{instance_id: true|false}`) into each run
directory, or pass `--resolved label=path`. A raw SWE-bench harness report
(`<run_id>.<model>.json` with `resolved_instances`) is also accepted.

Usage:
    python bench/compare.py --sample bench/sample_lite36.json \
        --arm A0=swebench_runs/A0 --arm A1=swebench_runs/A1 \
        --baseline A0 --out-dir bench/out/compare
"""

import argparse
import json
import sys
from math import comb
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).parent))
from analyze import CONTEXT_WINDOW, metrics_for  # noqa: E402

HYPOTHESES = {
    "H1 模型能力": "A3 vs A0：更强模型是否显著更高",
    "H2 显存大小": "API 模式下不可测（不加载本地权重）",
    "H3a 上下文窗口": "A0 的 peak_prompt_tokens / 窗口",
    "H3b 观测截断": "A4 vs A0：放宽观测上限是否有收益",
    "H4 推理链长度": "A1/A2 vs A0：压低/顶高推理预算是否改变结果",
    "S 步数预算": "A5 vs A0：50 步是否已够",
}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b = (arm right, base wrong), c = opposite."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def load_resolved(raw: dict) -> dict[str, bool]:
    if "resolved_instances" in raw:  # raw SWE-bench harness report
        out = {i: False for i in raw.get("unresolved_instances", [])}
        out.update({i: True for i in raw["resolved_instances"]})
        return out
    return {k: bool(v) for k, v in raw.items() if v is not None}


def collect(run_dir: Path, ids: list[str], step_limit: int) -> list[dict]:
    preds = json.loads((run_dir / "preds.json").read_text()) if (run_dir / "preds.json").exists() else {}
    rows = []
    for iid in ids:
        p = run_dir / iid / f"{iid}.traj.json"
        traj = json.loads(p.read_text()) if p.exists() else None
        rows.append(metrics_for(iid, traj, preds.get(iid, {}), step_limit))
    return rows


def summary(rows: list[dict], resolved: dict[str, bool] | None) -> dict:
    n = len(rows)
    got = {r["instance_id"]: resolved[r["instance_id"]] for r in rows if resolved and r["instance_id"] in resolved}
    sub = [r for r in rows if r["exit_status"] == "Submitted"]
    return {
        "n": n,
        "n_scored": len(got),
        "resolved": sum(got.values()),
        "resolve_rate": round(sum(got.values()) / len(got), 3) if got else None,
        "submitted": len(sub),
        "nonempty_patch": sum(1 for r in sub if not r["patch_empty"]),
        "median_steps": median([r["n_calls"] for r in rows]) if n else 0,
        "hit_step_limit": sum(1 for r in rows if r["hit_step_limit"]),
        "peak_ctx_pct": round(100 * max((r["peak_prompt_tokens"] for r in rows), default=0) / CONTEXT_WINDOW, 2),
        "length_trunc": sum(r["n_length_truncations"] for r in rows),
        "format_errors": sum(r["n_format_errors"] for r in rows),
        "reasoning_tokens": sum(r["reasoning_tokens_total"] for r in rows),
        "completion_tokens": sum(r["completion_tokens_total"] for r in rows),
        "reasoning_share": (
            round(sum(r["reasoning_tokens_total"] for r in rows) / sum(r["completion_tokens_total"] for r in rows), 3)
            if sum(r["completion_tokens_total"] for r in rows)
            else None
        ),
    }


def paired(base: dict[str, bool], arm: dict[str, bool]) -> dict:
    ids = [i for i in base if i in arm]
    b = sum(1 for i in ids if arm[i] and not base[i])  # arm wins
    c = sum(1 for i in ids if base[i] and not arm[i])  # baseline wins
    return {
        "n_paired": len(ids),
        "arm_only": b,
        "base_only": c,
        "delta": round((b - c) / len(ids), 3) if ids else None,
        "p_value": round(mcnemar_exact(b, c), 4) if ids else None,
    }


def verdict(arms: dict[str, dict], comps: dict[str, dict]) -> list[str]:
    """Turn the pairwise deltas into a per-hypothesis read-out."""
    out = []

    def delta(label: str) -> str:
        if label not in comps:
            return "未跑"
        c = comps[label]
        if c["delta"] is None:
            return "未判分"
        sig = "显著" if (c["p_value"] or 1) < 0.05 else "不显著"
        return f"Δ={c['delta']:+.1%}（p={c['p_value']}, {sig}）"

    a0 = arms.get("A0", {})
    out.append(f"**H3a 上下文窗口**：窗口 {CONTEXT_WINDOW:,}；峰值占用 **{a0.get('peak_ctx_pct', 0)}%**，"
               f"`finish_reason=length` {a0.get('length_trunc', 0)} 次 → "
               f"{'不是瓶颈' if a0.get('peak_ctx_pct', 0) < 50 else '需复查'}")
    out.append(f"**H2 显存**：被测模型走 DeepSeek 托管 API，本地不加载权重，显存与结果无关 → 不可测/不适用")
    out.append(f"**H4 推理链长度**：A1(low) {delta('A1')}；A2(max) {delta('A2')}"
               f"（基线推理 token 占比 {a0.get('reasoning_share')}）")
    out.append(f"**H3b 观测截断**：A4(观测 10k→40k) {delta('A4')}")
    out.append(f"**H1 模型能力**：A3(v4-pro) {delta('A3')}")
    out.append(f"**S 步数预算**：A5(50 步) {delta('A5')}；基线走满上限 {a0.get('hit_step_limit', 0)}/{a0.get('n', 0)}")
    improving = {k: v["delta"] for k, v in comps.items() if v["delta"] and v["delta"] > 0 and (v["p_value"] or 1) < 0.05}
    out.append("")
    if improving:
        top = max(improving, key=improving.get)
        out.append(f"**归因**：在当前实验里，唯一能显著抬高 resolve 率的是 `{top}`（{improving[top]:+.1%}）→ "
                   f"能力上限主要落在对应因素上，而非被排除的那些。")
    else:
        out.append("**归因**：换模型/改推理预算/放宽观测/砍步数都没有显著收益 → 当前 resolve 率不是被这些预算卡住的，"
                   "剩余解释是 scaffold 本身（工具粒度、无多文件检索、无测试反馈闭环）与该模型在真实仓库上的固有能力。")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="bench/sample_lite36.json")
    ap.add_argument("--arm", action="append", required=True, help="label=run_dir")
    ap.add_argument("--baseline", default="A0")
    ap.add_argument("--steps", action="append", default=[], help="label=step_limit")
    ap.add_argument("--resolved", action="append", default=[], help="label=path/to/resolved.json")
    ap.add_argument("--out-dir", default="bench/out/compare")
    args = ap.parse_args()

    sample = json.loads(Path(args.sample).read_text())
    ids = [r["instance_id"] for r in sample["rows"]]
    step_of = dict(s.split("=", 1) for s in args.steps)
    resolved_of = dict(s.split("=", 1) for s in args.resolved)

    arms, labels = {}, []
    for spec in args.arm:
        label, run_dir = spec.split("=", 1)
        run_dir = Path(run_dir)
        explicit = resolved_of.get(label)
        cand = Path(explicit) if explicit else run_dir / "resolved.json"
        resolved = load_resolved(json.loads(cand.read_text())) if cand.exists() else None
        rows = collect(run_dir, ids, int(step_of.get(label, 250)))
        arms[label] = {**summary(rows, resolved), "resolved_map": resolved, "run_dir": str(run_dir)}
        labels.append(label)

    comps = {}
    base = arms.get(args.baseline, {}).get("resolved_map")
    if base:
        for label in labels:
            if label == args.baseline or not arms[label]["resolved_map"]:
                continue
            comps[label] = paired(base, arms[label]["resolved_map"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample": args.sample,
        "baseline": args.baseline,
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "resolved_map"} for k, v in arms.items()},
        "paired_vs_baseline": comps,
        "hypotheses": HYPOTHESES,
    }
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    cols = ["n", "n_scored", "resolved", "resolve_rate", "submitted", "nonempty_patch",
            "median_steps", "hit_step_limit", "peak_ctx_pct", "length_trunc", "format_errors", "reasoning_share"]
    lines = ["# 对照实验汇总", "", f"- sample: `{args.sample}`（{len(ids)} 实例）", f"- baseline: `{args.baseline}`", "",
             "## 各 arm", "", "| arm | " + " | ".join(cols) + " |",
             "|---|" + "---|" * len(cols)]
    for label in labels:
        a = arms[label]
        lines.append(f"| {label} | " + " | ".join(str(a.get(c, "")) for c in cols) + " |")
    lines += ["", f"## 配对比较（vs {args.baseline}）", "",
              "| arm | n_paired | 仅本组对 | 仅基线对 | Δ | McNemar p |", "|---|---|---|---|---|---|"]
    for label, c in comps.items():
        lines.append(f"| {label} | {c['n_paired']} | {c['arm_only']} | {c['base_only']} | "
                     f"{c['delta']:+.1%} | {c['p_value']} |")
    lines += ["", "## 归因", ""] + [f"- {v}" for v in verdict(arms, comps) if v]
    (out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_dir}/comparison.md and comparison.json")


if __name__ == "__main__":
    main()
