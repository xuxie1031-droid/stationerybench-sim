#!/usr/bin/env bash
# Serve a policy on 127.0.0.1:PORT (foreground).   scripts/serve.sh MODEL|base [PORT=8230] [noclip|stock]
#   base -> stock server + $MOLMOACT2_BASE_DIR; HF-format dir -> eval/host_server_yam_noclip.py (default) or stock;
#   training checkpoint -> eval/serve_ckpt.sh (export to HF, then serve).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
MODEL=${1:?MODEL}; PORT=${2:-8230}; SRV=${3:-noclip}
cd "$MOLMOACT2_DIR"
if [ "$MODEL" = base ]; then exec "$SIM_PY" examples/yam/host_server_yam.py --host 127.0.0.1 --port "$PORT" --repo-id "$MOLMOACT2_BASE_DIR"; fi
if [ -f "$MODEL/config.json" ] && [ -f "$MODEL/norm_stats.json" ]; then
  S=$([ "$SRV" = stock ] && echo examples/yam/host_server_yam.py || echo "$GYRE_ROOT/eval/host_server_yam_noclip.py")
  exec "$SIM_PY" "$S" --host 127.0.0.1 --port "$PORT" --repo-id "$(readlink -f "$MODEL")"
fi
NOCLIP=$([ "$SRV" = stock ] && echo 0 || echo 1) exec bash "$GYRE_ROOT/eval/serve_ckpt.sh" "$MODEL" "$PORT"
