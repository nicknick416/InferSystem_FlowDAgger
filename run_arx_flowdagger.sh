#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:-demonstration}
shift || true

CONFIG=/home/xinzhi/InferSystem_FlowDAgger/Config/NeoVTLA/arx5_bimanual_neovtla.yaml
PYTHON=/home/xinzhi/miniconda3/envs/infersystem/bin/python

"${PYTHON}" /home/xinzhi/InferSystem_FlowDAgger/flowdagger_preflight.py \
  "${CONFIG}" --stage "${STAGE}"

exec "${PYTHON}" \
  /home/xinzhi/InferSystem_FlowDAgger/Example/robot_inference.py \
  "${CONFIG}" \
  --flow-stage "${STAGE}" --publish-diagnostics "$@"
