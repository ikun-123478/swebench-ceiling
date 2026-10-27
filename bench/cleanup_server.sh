#!/usr/bin/env bash
# 只清理【本次实验】产生的东西。共享服务器上禁止全局 prune。
# 默认 dry-run，只打印不动手；加 --yes 才真的删。
set -euo pipefail

REPO="${REPO:-$HOME/minisweagent}"
YES=0
[[ "${1:-}" == "--yes" ]] && YES=1

run() {
  if [[ $YES -eq 1 ]]; then "$@"; else echo "  [dry-run] $*"; fi
}

say() { printf '\n=== %s ===\n' "$*"; }
df_before=$(df -h "$HOME" | tail -1 | awk '{print $4}')
echo "清理前可用空间: $df_before"

if ! docker info >/dev/null 2>&1; then
  echo "SKIP: 当前账号没有 docker 权限，镜像部分跳过（不影响目录清理）"
  DOCKER_OK=0
else
  DOCKER_OK=1
fi

if [[ $DOCKER_OK -eq 1 ]]; then
  say "1. 本实验残留容器（只认 minisweagent- 前缀）"
  docker ps -a --format '{{.Names}}\t{{.Status}}' | grep '^minisweagent-' || echo "  无"

  say "2. 本实验镜像（只认 sweb.eval，逐个删；绝不 docker system prune）"
  # 绝不使用：docker system prune / docker volume prune / docker builder prune
  # 那几个是全局操作，会把其他用户的镜像和缓存一并删掉。
  imgs=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep 'sweb.eval' || true)
  if [[ -z "$imgs" ]]; then
    echo "  无 sweb.eval 镜像"
  else
    echo "$imgs" | sed 's/^/  - /'
    echo "$imgs" | while read -r img; do run docker rmi "$img"; done
  fi

  say "3. 顺带确认没有本实验的卷 / 网络残留"
  docker volume ls --format '{{.Name}}' | grep '^minisweagent' || echo "  无自定义卷"
fi

say "4. 项目目录（含 venv）"
if [[ -d "$REPO" ]]; then
  echo "  $REPO  $(du -sh "$REPO" 2>/dev/null | cut -f1)"
  run rm -rf "$REPO"
else
  echo "  不存在"
fi

say "5. 本账号的缓存（不碰系统级目录）"
for d in "$HOME/.cache/huggingface" "$HOME/.cache/pip"; do
  [[ -e "$d" ]] && { echo "  $d  $(du -sh "$d" 2>/dev/null | cut -f1)"; run rm -rf "$d"; }
done
# 装依赖时可能写下的密钥文件
[[ -f "$HOME/.mswea_env" ]] && { echo "  $HOME/.mswea_env  (含 API key)"; run rm -f "$HOME/.mswea_env"; }
echo "  （~/.docker/config.json 里可能留了镜像站登录信息，如需也清: rm -f ~/.docker/config.json）"

if [[ $DOCKER_OK -eq 1 ]]; then
  say "6. 校验：本实验相关镜像应为空"
  docker images --format '{{.Repository}}:{{.Tag}}' | grep -c 'sweb.eval' || echo "  0 个残留"
fi

say "7. 空间对比"
df -h "$HOME" | tail -1
echo "清理前: $df_before"

cat <<'TIP'

如需连提权一起还原（把账号退出 docker 组，恢复到实验前状态）：
    sudo gpasswd -d $USER docker
该改动需要重新登录才生效。注意：若实验室本来就把你放在 docker 组里，不要执行这条。
TIP
