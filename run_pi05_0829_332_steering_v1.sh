#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/xinzhi/InferSystem_FlowDAgger

# `closed_loop` overrides the YAML defaults to run_stage=closed_loop and
# shadow_mode=false.  Preflight runs before any robot control is initialized.
exec "${ROOT}/run_arx_flowdagger.sh" closed_loop "$@"
