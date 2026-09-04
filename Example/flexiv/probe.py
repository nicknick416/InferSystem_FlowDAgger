"""机器人连接探针 — 连接并轮询状态快照。

用法:
    python Example/flexiv/probe.py Config/rizon4_example.yaml
    python Example/flexiv/probe.py Config/rizon4_example.yaml --polls 10 --enable
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Core.logging import setup_run_logger
from Robot import BaseRobot

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="机器人连接探针 — 轮询状态快照")
    parser.add_argument("config", help="YAML 配置文件路径")
    parser.add_argument("--polls", type=int, default=5, help="轮询次数")
    parser.add_argument("--interval", type=float, default=0.5, help="轮询间隔 (秒)")
    parser.add_argument("--timeout", type=float, default=10.0, help="等待 operational 超时 (秒)")
    parser.add_argument("--enable", action="store_true", help="连接后自动使能")
    parser.add_argument("--clear-fault", action="store_true", help="连接后自动清除故障")
    args = parser.parse_args()
    setup_run_logger(__file__, args.config)

    try:
        with BaseRobot.from_config(args.config) as robot:
            print(json.dumps({
                "robot_name": robot.name,
                "robot_type": robot.robot_type,
                "dof": robot.dof,
            }, ensure_ascii=False))

            if args.clear_fault and robot.is_fault():
                robot.clear_fault()
                time.sleep(1.0)

            if args.enable:
                robot.enable()
                ok = robot.wait_until_operational(timeout_s=args.timeout)
                print(json.dumps({"enabled": ok}))

            for poll_index in range(args.polls):
                state = robot.observe()
                payload = {
                    "poll": poll_index,
                    "timestamp": state.timestamp,
                    "connected": robot.is_connected(),
                    "operational": robot.is_operational(),
                    "fault": robot.is_fault(),
                    "joint_positions": state.joint_positions,
                    "eef_pose": state.eef_pose,
                }
                print(json.dumps(payload, ensure_ascii=True))
                if poll_index + 1 < args.polls:
                    time.sleep(args.interval)

    except Exception as exc:
        raise SystemExit(f"Probe failed: {exc}") from exc


if __name__ == "__main__":
    main()
