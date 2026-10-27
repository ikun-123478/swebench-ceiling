#!/usr/bin/env bash
# 服务器端一次性引导：环境自检 → 装依赖 → 生成固定子集 → 预拉镜像。
# 幂等：可重复执行。任何一步失败都会明确报错，不会静默继续。
set -euo pipefail

REPO="${REPO:-$HOME/minisweagent}"
SAMPLE="$REPO/bench/sample_lite36.json"
IMAGES_LOG="$REPO/bench/images.txt"
PULL_LOG="$REPO/bench/pull_all.log"

say() { printf '\n=== %s ===\n' "$*"; }

say "0. 机器身份"
hostname; whoami; date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "(无 nvidia-smi，不影响本实验)"
echo "--- CPU / 内存 / 磁盘"
nproc
free -g | head -2
df -h "$HOME" | tail -1

say "1. 检查仓库"
if [[ ! -f "$REPO/pyproject.toml" ]]; then
  echo "ERROR: $REPO 下没有 pyproject.toml，仓库没放对位置或还没克隆" >&2
  exit 1
fi
cd "$REPO"
echo "repo ok: $REPO"
git log --oneline -1 2>/dev/null || echo "(不是 git 仓库，跳过)"

say "2. Python 与依赖"
PY=python3
"$PY" -c 'import sys; assert sys.version_info >= (3,10), sys.version; print("python", sys.version.split()[0])'
if [[ ! -x .venv/bin/python ]]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
# 国内走阿里云镜像；失败则回落官方源
python -m pip install -q -e . -i https://mirrors.aliyun.com/pypi/simple \
  || python -m pip install -q -e .
python -c 'import minisweagent, litellm; print("minisweagent", minisweagent.__version__)'

say "3. Docker"
docker info --format 'Server {{.ServerVersion}} | {{.OperatingSystem}} | CPUs {{.NCPU}} | Mem {{.MemTotal}}' 2>/dev/null || {
  echo "ERROR: docker 不可用（未启动或无权限）。确认 sudo usermod -aG docker \$USER 后重新登录" >&2
  exit 1
}

say "4. 生成固定子集（36 条 / 12 仓库）"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ -f "$SAMPLE" ]]; then
  echo "已存在，跳过：$SAMPLE"
else
  python bench/select_instances.py --dataset lite --per-repo 3 \
    --percentile 0.75 --max-instances 36 --out "$SAMPLE"
fi

say "5. 预拉镜像（关键：上游 pull_timeout 只有 120s）"
python - "$SAMPLE" > "$IMAGES_LOG" <<'PY'
import json, sys
for r in json.load(open(sys.argv[1]))["rows"]:
    iid = r["instance_id"].replace("__", "_1776_").lower()
    print(f"docker.io/swebench/sweb.eval.x86_64.{iid}:latest")
PY
echo "待拉取: $(wc -l < "$IMAGES_LOG") 个镜像"

pull_one() {
  img="$1"
  for attempt in 1 2 3; do
    if docker pull "$img" >/dev/null 2>&1; then
      echo "OK   $img"
      return 0
    fi
    echo "RETRY($attempt) $img"
    sleep 5
  done
  echo "FAIL $img"
}
export -f pull_one

: > "$PULL_LOG"
xargs -a "$IMAGES_LOG" -P 4 -n 1 bash -c 'pull_one "$0"' | tee "$PULL_LOG"

say "6. 校验"
want=$(wc -l < "$IMAGES_LOG")
have=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -c 'sweb.eval' || true)
echo "需要 $want 个，本地已有 $have 个"
grep -c '^FAIL' "$PULL_LOG" 2>/dev/null && echo "有失败项，见 $PULL_LOG" || echo "无失败项"

df -h "$HOME" | tail -1
say "引导完成"
