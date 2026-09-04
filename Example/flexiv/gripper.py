"""控制机器人夹爪开合。

用法:
    python Example/flexiv/gripper.py Config/rizon4_example.yaml open
    python Example/flexiv/gripper.py Config/rizon4_example.yaml close
    python Example/flexiv/gripper.py Config/rizon4_example.yaml open close open
    python Example/flexiv/gripper.py Config/rizon4_example.yaml move --width 0.05
"""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="控制机器人夹爪")
    parser.add_argument("config", help="YAML 配置文件路径")
    parser.add_argument(
        "actions", nargs="+", choices=["open", "close", "move"],
        help="夹爪动作：open / close / move",
    )
    parser.add_argument("--timeout", type=float, default=15.0, help="等待 operational 超时秒数")
    parser.add_argument("--width", type=float, default=0.09, help="open/move 目标宽度 (米)")
    parser.add_argument("--velocity", type=float, default=0.1, help="夹爪速度 (米/秒)")
    parser.add_argument("--force", type=float, default=30.0, help="夹持力 (N)")
    parser.add_argument("--interval", type=float, default=2.0, help="动作间等待 (秒)")
    args = parser.parse_args()
    setup_run_logger(__file__, args.config)

    log.info("[1] 从配置创建机器人...")
    with BaseRobot.from_config(args.config) as robot:
        if robot.is_fault():
            log.info("[2] 检测到故障，正在清除...")
            robot.clear_fault()
            time.sleep(2.0)

        if not robot.is_operational():
            log.info("[2] 使能机器人...")
            robot.enable()
            if not robot.wait_until_operational(timeout_s=args.timeout):
                raise SystemExit("机器人未能在超时时间内变为 operational")
        log.info(f"[2] 机器人 '{robot.name}' 已 operational")

        log.info("[3] 初始化夹爪...")
        gripper = robot.create_gripper()
        if gripper is None:
            raise SystemExit("该机器人未配置夹爪")

        with gripper:
            gs = gripper.observe()
            log.info(f"    宽度: {gs.width:.4f} m, 力: {gs.force:.2f} N")

            for i, action in enumerate(args.actions):
                if i > 0:
                    time.sleep(args.interval)

                if action == "open":
                    log.info(f"[{4 + i}] 打开夹爪 (宽度={args.width}m) ...")
                    gripper.move(args.width, velocity=args.velocity, force=args.force)
                elif action == "close":
                    log.info(f"[{4 + i}] 关闭夹爪 (力={args.force}N) ...")
                    gripper.set(False)
                elif action == "move":
                    log.info(f"[{4 + i}] 移动夹爪到 {args.width}m ...")
                    gripper.move(args.width, velocity=args.velocity, force=args.force)

                gs = gripper.observe()
                log.info("    宽度: %.4f m, 力: %.2f N", gs.width, gs.force)

    log.info("完成!")


if __name__ == "__main__":
    main()
