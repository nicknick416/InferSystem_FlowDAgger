#!/usr/bin/env bash
set -euo pipefail

# Start ARX5 CAN interfaces via CANable slcand bridge.
# Default mapping:
#   /dev/canable_left  -> can0
#   /dev/canable_right -> can1
#
# Usage:
#   scripts/start_arx5_can.sh
#   scripts/start_arx5_can.sh /dev/canable_left /dev/canable_right can0 can1

LEFT_DEV="${1:-/dev/canable_left}"
RIGHT_DEV="${2:-/dev/canable_right}"
LEFT_IF="${3:-can0}"
RIGHT_IF="${4:-can1}"

echo "[1/5] Stop old slcand processes ..."
sudo killall slcand 2>/dev/null || true
sleep 0.5

if [[ ! -e "${LEFT_DEV}" ]]; then
  echo "ERROR: left CANable device not found: ${LEFT_DEV}" >&2
  exit 1
fi
if [[ ! -e "${RIGHT_DEV}" ]]; then
  echo "ERROR: right CANable device not found: ${RIGHT_DEV}" >&2
  exit 1
fi

echo "[2/5] Bridge ${LEFT_DEV} -> ${LEFT_IF}"
sudo slcand -o -c -s8 "${LEFT_DEV}" "${LEFT_IF}"

echo "[3/5] Bridge ${RIGHT_DEV} -> ${RIGHT_IF}"
sudo slcand -o -c -s8 "${RIGHT_DEV}" "${RIGHT_IF}"

sleep 0.5
echo "[4/5] Bring CAN links up ..."
sudo ip link set "${LEFT_IF}" up
sudo ip link set "${RIGHT_IF}" up

echo "[5/5] Verify links"
ip link show "${LEFT_IF}"
ip link show "${RIGHT_IF}"
echo "CAN done: ${LEFT_IF}, ${RIGHT_IF}"
