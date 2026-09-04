#!/usr/bin/env bash
set -euo pipefail

CONFIG=/home/xinzhi/InferSystem_FlowDAgger/Config/NeoVTLA/arx5_bimanual_neovtla.yaml
PYTHON=/home/xinzhi/miniconda3/envs/infersystem/bin/python

exec "${PYTHON}" \
  /home/xinzhi/InferSystem_FlowDAgger/Example/robot_inference.py \
  "${CONFIG}" \
  --flow-stage demo \
  --steering-version active \
  "$@"
