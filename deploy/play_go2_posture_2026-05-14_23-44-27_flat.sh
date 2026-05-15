#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="$REPO_ROOT/tested_policies/go2/go2_posture_guidance/2026-05-14_23-44-27_terrain_A_air0.5_ori-0.005_ht-0.1"
LATEST="$(find "$RUN_DIR" -maxdepth 1 -name 'model_*.pt' -printf '%f\n' | sed 's/model_//;s/.pt//' | sort -n | tail -1)"

if [[ -z "$LATEST" ]]; then
  echo "No model_*.pt checkpoint found in $RUN_DIR" >&2
  exit 1
fi

cd "$REPO_ROOT"
SCENE=flat \
GO2_POSTURE_RUN_DIR="$RUN_DIR" \
GO2_POSTURE_CHECKPOINT="$RUN_DIR/model_${LATEST}.pt" \
GO2_POSTURE_USE_EXPORTED_ADAPTATION=1 \
python3 deploy/play_go2_posture_mujoco.py
