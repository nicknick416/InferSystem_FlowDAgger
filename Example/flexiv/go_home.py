"""控制机器人回到原点 (Home)。

用法:
    python Example/flexiv/go_home.py Config/rizon4_example.yaml
    python Example/flexiv/go_home.py Config/rizon4_example.yaml --velocity 50
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
    parser = argparse.ArgumentParser(description="控制机器人回到原点")
    parser.add_argument("config", help="YAML 配置文件路径")
    parser.add_argument("--velocity", type=int, default=50, help="速度百分比 (1-100)")
    parser.add_argument("--timeout", type=float, default=15.0, help="等待 operational 超时秒数")
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
        log.info("[2] 机器人 '%s' 已 operational", robot.name)

        log.info("[3] 执行 Home (速度 %d%%) ...", args.velocity)
        reached = robot.go_home(velocity=args.velocity / 100.0)

        state = robot.observe()
        log.info("[4] 回原点%s!", "完成" if reached else "超时")
        log.info("    TCP: [%s]", ", ".join(f"{v:.4f}" for v in state.eef_pose))
        log.info("    关节: [%s]", ", ".join(f"{v:.4f}" for v in state.joint_positions))


if __name__ == "__main__":
    main()
