"""ARX5 连接探针 — 连接并轮询状态快照。

用法:
    python Example/arx5/probe.py Config/arx5_bimanual_example.yaml
    python Example/arx5/probe.py Config/arx5_bimanual_example.yaml --polls 10 --enable
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Core.logging import setup_run_logger
from Robot import BaseRobot

log = logging.getLogger(__name__)

# Ctrl-C 优雅退出标志
_shutdown = False


def _sigint_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] 收到中断信号，正在退出...", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ARX5 连接探针 — 轮询状态快照")
    parser.add_argument("config", help="YAML 配置文件路径")
    parser.add_argument("--polls", type=int, default=5, help="轮询次数")
    parser.add_argument("--interval", type=float, default=0.5, help="轮询间隔 (秒)")
    parser.add_argument("--timeout", type=float, default=10.0, help="等待 operational 超时 (秒)")
    parser.add_argument("--enable", action="store_true", help="连接后自动使能")
    parser.add_argument("--clear-fault", action="store_true", help="连接后自动清除故障")
    args = parser.parse_args()
    setup_run_logger(__file__, args.config)

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    with BaseRobot.from_config(args.config) as robot:
        print(json.dumps({
            "robot_name": robot.name,
            "robot_type": robot.robot_type,
            "dof": robot.dof,
        }, ensure_ascii=False), flush=True)

        if _shutdown:
            return

        if args.clear_fault and robot.is_fault():
            robot.clear_fault()
            time.sleep(1.0)

        if args.enable:
            robot.enable()
            ok = robot.wait_until_operational(timeout_s=args.timeout)
            print(json.dumps({"enabled": ok}), flush=True)

        for idx in range(args.polls):
            if _shutdown:
                break
            try:
                state = robot.observe()
            except Exception as exc:
                print(json.dumps({"poll": idx, "error": str(exc)}), flush=True)
                break

            q = state.joint_positions
            payload: dict = {
                "poll": idx,
                "timestamp": round(state.timestamp, 6),
                "connected": robot.is_connected(),
                "operational": robot.is_operational(),
                "fault": robot.is_fault(),
            }

            # 按维度分拆显示，方便排查
            if len(q) >= 14:
                payload["left_joints"] = [round(v, 4) for v in q[:6]]
                payload["left_gripper"] = round(q[6], 4)
                payload["right_joints"] = [round(v, 4) for v in q[7:13]]
                payload["right_gripper"] = round(q[13], 4)
            elif len(q) >= 7:
                payload["joints"] = [round(v, 4) for v in q[:6]]
                payload["gripper"] = round(q[6], 4)
            else:
                payload["joint_positions"] = [round(v, 4) for v in q]

            if state.joint_velocities:
                payload["joint_velocities"] = [round(v, 4) for v in state.joint_velocities]

            print(json.dumps(payload, ensure_ascii=False), flush=True)

            if idx + 1 < args.polls and not _shutdown:
                time.sleep(args.interval)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
