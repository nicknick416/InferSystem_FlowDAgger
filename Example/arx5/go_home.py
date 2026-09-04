"""控制 ARX5 回到 Home 位置。

原理: ARX5 是 PD 控制臂，reset_to_home 执行 SDK 内置回零轨迹。
      回零后必须以控制频率持续下发 Home 位置指令维持姿态，
      否则 disconnect → set_to_damping 会使机械臂立刻松弛。

用法:
    python Example/arx5/go_home.py Config/arx5_example.yaml
    python Example/arx5/go_home.py Config/arx5_bimanual_example.yaml
    python Example/arx5/go_home.py Config/arx5_example.yaml --hold 5
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Core import Action, ActionSpace
from Core.logging import setup_run_logger
from Robot import BaseRobot

log = logging.getLogger(__name__)

_shutdown = False


def _sigint_handler(signum, frame):
    global _shutdown
    _shutdown = True
    log.info("收到中断信号，正在退出...")


def _hold_position(robot: BaseRobot, target: list[float], duration: float) -> None:
    """以控制频率持续下发位置指令，维持机械臂姿态。"""
    dt = 1.0 / robot.get_params().control_frequency_hz
    steps = max(int(round(duration / dt)), 1)
    action = Action(ActionSpace.JOINT_POSITION, target)
    next_time = time.perf_counter()
    for _ in range(steps):
        if _shutdown:
            break
        robot.act(action)
        next_time += dt
        robot._rt_sleep_until(next_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="ARX5 回 Home")
    parser.add_argument("config", help="YAML 配置文件路径")
    parser.add_argument("--timeout", type=float, default=20.0, help="等待 operational 超时秒数")
    parser.add_argument("--hold", type=float, default=3.0, help="到达 Home 后保持位置的秒数")
    args = parser.parse_args()
    setup_run_logger(__file__, args.config)

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    log.info("[1] 从配置创建机器人...")
    with BaseRobot.from_config(args.config) as robot:
        if robot.is_fault():
            log.info("[2] 检测到故障，正在清除...")
            robot.clear_fault()
            time.sleep(1.5)

        if not robot.is_operational():
            log.info("[2] 使能机器人...")
            robot.enable()
            if not robot.wait_until_operational(timeout_s=args.timeout):
                raise SystemExit("机器人未能在超时时间内变为 operational")

        log.info(f"[3] 机器人 '{robot.name}' 已 operational (dof={robot.dof})")

        # SDK 回零轨迹
        log.info("[4] 执行 go_home ...")
        robot.go_home()

        # 读取回零后的位置，持续下发维持姿态
        obs = robot.observe()
        q = obs.joint_positions
        log.info(f"[5] go_home 完成, q={[round(v, 4) for v in q[:min(len(q), 7)]]}")

        if args.hold > 0 and not _shutdown:
            log.info(f"[6] 保持 Home 位置 {args.hold:.1f}s (Ctrl-C 提前退出) ...")
            _hold_position(robot, list(q), args.hold)

        log.info("[7] 完成，断开连接")


if __name__ == "__main__":
    main()
