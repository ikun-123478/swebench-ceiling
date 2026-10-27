#!/usr/bin/env bash
# 服务器端执行器。用法：
#   bash ~/run_arms.sh prefetch        # 从镜像站拉 36 个镜像并打回官方名（幂等）
#   bash ~/run_arms.sh smoke           # 官方 dummy 实例冒烟
#   bash ~/run_arms.sh arm A0_baseline # 跑单个 arm
#   bash ~/run_arms.sh all             # 依次跑 6 个 arm
#
# 共享服务器约定：容器限 4 核/8GB（见 bench/lite_deepseek.yaml），进程 nice 10，并发默认 8。
set -euo pipefail

REPO="${REPO:-$HOME/minisweagent}"
MIRROR="${MIRROR:-docker.1ms.run}"
WORKERS="${WORKERS:-8}"
ARMS=(A0_baseline A1_effort_low A2_effort_max A3_model_pro A4_obs_40k A5_steps_50)

cd "$REPO"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
[[ -f "$HOME/.mswea_env" ]] && source "$HOME/.mswea_env"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MSWEA_COST_TRACKING=ignore_errors
export MSWEA_SILENT_STARTUP=1
: "${OPENAI_API_KEY:?请先 export OPENAI_API_KEY 或写入 ~/.mswea_env}"
: "${OPENAI_BASE_URL:=https://api.deepseek.com}"
export OPENAI_BASE_URL

say() { printf '\n=== %s ===\n' "$*"; }

images() {
  python - "$@" <<'PY'
import json, sys
for r in json.load(open("bench/sample_lite36.json"))["rows"]:
    iid = r["instance_id"].replace("__", "_1776_").lower()
    print(f"swebench/sweb.eval.x86_64.{iid}:latest")
PY
}

prefetch_one() {
  local img="$1"
  local canonical="docker.io/$img"
  if docker image inspect "$canonical" >/dev/null 2>&1; then
    echo "SKIP $img"
    return 0
  fi
  for attempt in 1 2 3; do
    if docker pull -q "$MIRROR/$img" >/dev/null 2>&1; then
      docker tag "$MIRROR/$img" "$canonical"
      echo "OK   $img"
      return 0
    fi
    echo "RETRY($attempt) $img"
    sleep 5
  done
  echo "FAIL $img"
}
export -f prefetch_one
export MIRROR

case "${1:-}" in
  prefetch)
    say "预拉 36 个镜像（经 $MIRROR，打回 docker.io 官方名）"
    images | xargs -P 4 -n 1 bash -c 'prefetch_one "$0"' | tee bench/prefetch.log
    echo "--- 校验"
    have=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -c '^docker.io/swebench/sweb.eval' || true)
    echo "已有官方名镜像: $have / 36"
    grep -c '^FAIL' bench/prefetch.log 2>/dev/null | grep -v '^0$' && echo "有失败项，见 bench/prefetch.log" || echo "全部成功"
    ;;

  smoke)
    say "冒烟（官方 dummy 实例）"
    EXTRA=()
    [[ "${MODAL:-0}" == "1" ]] && EXTRA=(-c bench/modal.yaml)
    nice -n 10 mini-extra swebench --subset _test --split test \
      -c swebench.yaml -c bench/lite_deepseek.yaml "${EXTRA[@]}" -w 1 \
      -o swebench_runs/L0_smoke
    ;;

  arm)
    ARM="${2:?用法: run_arms.sh arm <ARM_NAME>}"
    : "${ARM:?}"
    say "跑 $ARM（workers=$WORKERS, modal=${MODAL:-0}）"
    FILTER=$(python -c "import json;print(json.load(open('bench/sample_lite36.json'))['filter_regex'])")
    EXTRA=()
    [[ "${MODAL:-0}" == "1" ]] && EXTRA=(-c bench/modal.yaml)
    nice -n 10 mini-extra swebench --subset lite --split test --filter "$FILTER" \
      -c swebench.yaml -c bench/lite_deepseek.yaml -c "bench/arms/$ARM.yaml" "${EXTRA[@]}" \
      -w "$WORKERS" -o "swebench_runs/$ARM"
    ;;

  all)
    for a in "${ARMS[@]}"; do
      bash "$0" arm "$a"
    done
    ;;

  *)
    sed -n '2,10p' "$0"
    exit 1
    ;;
esac
