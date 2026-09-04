"""控制 ARX5 夹爪（单臂/双臂）。

原理: ARX5 夹爪由 PD 控制器驱动，需要以控制频率（500Hz）持续下发
      目标位置指令，PD 控制器才能平稳将夹爪驱动到目标开口。

说明:
  - 单臂(arx5): 7 维命令 [q0..q5, gripper]
  - 双臂(arx5_bimanual): 14 维命令 [lq0..lq5, lgrip, rq0..rq5, rgrip]
    默认左右夹爪同时动作，可用 --left-only / --right-only 单独控制。

用法:
    python Example/arx5/gripper.py Config/arx5_example.yaml open
    python Example/arx5/gripper.py Config/arx5_example.yaml close
    python Example/arx5/gripper.py Config/arx5_example.yaml open close
    python Example/arx5/gripper.py Config/arx5_bimanual_example.yaml open
    python Example/arx5/gripper.py Config/arx5_bimanual_example.yaml open --left-only
    python Example/arx5/gripper.py Config/arx5_bimanual_example.yaml close --right-only
    python Example/arx5/gripper.py Config/arx5_bimanual_example.yaml move --width 0.05
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

from Core import Action, ActionSpace
from Core.logging import setup_run_logger
from Robot import BaseRobot

log = logging.getLogger(__name__)


def _map_target(action_name: str, width: float, gmin: float, gmax: float) -> float:
    """将动作名映射到夹爪目标位置。"""
    if action_name == "close":
        return gmin
    elif action_name == "open":
        return gmax
    else:  # move
        return max(gmin, min(width, gmax))


def _hold_action(robot: BaseRobot, cmd: list[float], duration: float) -> None:
    """以控制频率持续下发同一 JOINT_POSITION 指令，驱动 PD 控制器到位。"""
    dt = 1.0 / robot.get_params().control_frequency_hz
    steps = max(int(round(duration / dt)), 1)
    action = Action(ActionSpace.JOINT_POSITION, cmd)
    next_time = time.perf_counter()
    for _ in range(steps):
        robot.act(action)
        next_time += dt
        robot._rt_sleep_until(next_time)


def _gripper_limits_single(robot) -> tuple[float, float]:
    """单臂夹爪的规范开合范围 (m)。"""
    gmin = float(getattr(robot, "_gripper_canonical_min", 0.0))
    gmax = float(getattr(robot, "_gripper_width", 0.1))
    return gmin, gmax


def main() -> None:
    parser = argparse.ArgumentParser(description="控制 ARX5 夹爪")
    parser.add_argument("config", help="YAML 配置文件路径")
    parser.add_argument("actions", nargs="+", choices=["open", "close", "move"])
    parser.add_argument("--timeout", type=float, default=15.0, help="等待 operational 超时秒数")
    parser.add_argument("--width", type=float, default=0.05, help="move 的目标开口 (m)")
    parser.add_argument("--hold", type=float, default=1.5, help="每次动作的持续下发时间 (秒)")
    # 双臂选择：默认两边一起动
    side = parser.add_mutually_exclusive_group()
    side.add_argument("--left-only", action="store_true", help="双臂时仅控制左夹爪")
    side.add_argument("--right-only", action="store_true", help="双臂时仅控制右夹爪")
    args = parser.parse_args()
    setup_run_logger(__file__, args.config)

    log.info("[1] 从配置创建机器人...")
    with BaseRobot.from_config(args.config) as robot:
        if robot.is_fault():
            log.info("[2] 检测到故障，正在清除...")
            robot.clear_fault()
            time.sleep(1.5)
        if not robot.is_operational():
            robot.enable()
            if not robot.wait_until_operational(timeout_s=args.timeout):
                raise SystemExit("机器人未能在超时时间内变为 operational")

        obs = robot.observe()
        dof = len(obs.joint_positions)
        log.info(f"[2] 机器人 '{robot.name}' 已 operational (dof={dof})")

        # ── 单臂 7 维 ───────────────────────────────────────────
        if dof == 7:
            if not getattr(robot, "_enable_gripper", False):
                raise SystemExit(
                    "当前配置 enable_gripper=false，驱动层不会下发夹爪目标。\n"
                    "请在 YAML 中设置 enable_gripper: true。"
                )
            gmin, gmax = _gripper_limits_single(robot)
            for i, act_name in enumerate(args.actions):
                q_now = list(robot.observe().joint_positions)
                g = _map_target(act_name, args.width, gmin, gmax)
                cmd = q_now[:6] + [g]
                log.info(f"[{3 + i}] {act_name}: gripper={g:.4f} (range [{gmin:.4f}, {gmax:.4f}])")
                _hold_action(robot, cmd, args.hold)

        # ── 双臂 14 维 ──────────────────────────────────────────
        elif dof == 14:
            if not getattr(robot, "_enable_gripper", True):
                raise SystemExit(
                    "当前配置 enable_gripper=false，驱动层不会下发夹爪目标。\n"
                    "请在 YAML 中设置 enable_gripper: true。"
                )
            p = robot.get_params()
            lg_min, lg_max = p.joint_position_min[6], p.joint_position_max[6]
            rg_min, rg_max = p.joint_position_min[13], p.joint_position_max[13]

            ctrl_left = not args.right_only
            ctrl_right = not args.left_only

            for i, act_name in enumerate(args.actions):
                q_now = list(robot.observe().joint_positions)
                cmd = list(q_now)
                if ctrl_left:
                    cmd[6] = _map_target(act_name, args.width, lg_min, lg_max)
                if ctrl_right:
                    cmd[13] = _map_target(act_name, args.width, rg_min, rg_max)
                log(
                    f"[{3 + i}] {act_name}: left={cmd[6]:.4f} right={cmd[13]:.4f}"
                    f" (left={'on' if ctrl_left else 'off'}, right={'on' if ctrl_right else 'off'})"
                )
                _hold_action(robot, cmd, args.hold)

        else:
            raise SystemExit(f"不支持的维度: dof={dof}")

        log.info("[done] 完成")


if __name__ == "__main__":
    main()
