"""Pick a stratified, difficulty-weighted sample of SWE-bench instances.

Ceiling-probing intent: within each repo take the instance at the ``--percentile``
of gold-patch size (default 75th), i.e. harder than median but not the pathological
tail. One instance per repo keeps the Docker image budget bounded.

Usage:
    python bench/select_instances.py --dataset verified --per-repo 1 \
        --percentile 0.75 --out bench/sample.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

DATASETS = {
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="verified", choices=sorted(DATASETS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-repo", type=int, default=1)
    ap.add_argument("--percentile", type=float, default=0.75)
    ap.add_argument("--max-instances", type=int, default=16)
    ap.add_argument("--out", default="bench/sample.json")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(DATASETS[args.dataset], split=args.split)
    by_repo: dict[str, list] = defaultdict(list)
    for row in ds:
        by_repo[row["repo"]].append(row)

    rows = []
    for repo in sorted(by_repo):
        items = sorted(by_repo[repo], key=lambda r: len(r.get("patch") or ""))
        n = len(items)
        want = min(args.per_repo, n)
        fracs = [args.percentile] if want == 1 else [
            args.percentile + (0.95 - args.percentile) * k / (want - 1) for k in range(want)
        ]
        # quantile positions can collide in small repos: slide to a free slot instead of duplicating
        taken: set[int] = set()
        picks = []
        for frac in fracs:
            i = min(n - 1, round(frac * (n - 1)))
            while i in taken and i > 0:
                i -= 1
            if i in taken:
                continue
            taken.add(i)
            picks.append(items[i])
        for r in picks:
            rows.append(
                {
                    "instance_id": r["instance_id"],
                    "repo": r["repo"],
                    "base_commit": r["base_commit"],
                    "gold_patch_chars": len(r.get("patch") or ""),
                    "problem_chars": len(r.get("problem_statement") or ""),
                    "n_fail_to_pass": len(json.loads(r.get("FAIL_TO_PASS") or "[]")),
                    "n_pass_to_pass": len(json.loads(r.get("PASS_TO_PASS") or "[]")),
                }
            )

    # hardest first, then cap
    rows.sort(key=lambda x: x["gold_patch_chars"], reverse=True)
    rows = rows[: args.max_instances]
    rows.sort(key=lambda x: x["instance_id"])

    ids = [r["instance_id"] for r in rows]
    out = {
        "dataset": DATASETS[args.dataset],
        "split": args.split,
        "n": len(ids),
        "selection_rule": (
            f"stratified by repo; within repo pick the instance at p={args.percentile} "
            "of gold-patch size; 1+ per repo; capped at --max-instances"
        ),
        "instance_ids": ids,
        # handy for `mini-extra swebench --filter`
        "filter_regex": "^(" + "|".join(re.escape(i) for i in ids) + ")$",
        "repos": sorted({r["repo"] for r in rows}),
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}: {len(ids)} instances across {len(out['repos'])} repos")

    print(f"\n{'instance_id':40s} {'gold_patch':>10s} {'problem':>8s} {'F2P':>4s} {'P2P':>5s}")
    for r in rows:
        print(f"{r['instance_id']:40s} {r['gold_patch_chars']:10d} {r['problem_chars']:8d} "
              f"{r['n_fail_to_pass']:4d} {r['n_pass_to_pass']:5d}")


if __name__ == "__main__":
    main()
