"""ARX5 运动功能测试：关节移动、笛卡尔移动、夹爪控制。

依次测试:
  1. go_home — SDK 原生回零
  2. move_joint_position — smoothstep 插值关节移动（摆动后回零）
  3. cartesian — 在当前末端位姿上施加小位移后用 IK 下发
  4. gripper — 开合夹爪

单臂:
    python Example/arx5/motion_test.py Config/arx5_example.yaml
    python Example/arx5/motion_test.py Config/arx5_example.yaml --skip cartesian gripper

双臂:
    python Example/arx5/motion_test.py Config/arx5_bimanual_example.yaml
    python Example/arx5/motion_test.py Config/arx5_bimanual_example.yaml --skip cartesian
"""
from __future__ import annotations

import argparse
import logging
import math
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


def wait_ready(robot: BaseRobot, timeout_s: float) -> None:
    if robot.is_fault():
        log.info("  检测到故障，正在 clear_fault ...")
        robot.clear_fault()
        time.sleep(1.5)
    if not robot.is_operational():
        robot.enable()
        if not robot.wait_until_operational(timeout_s=timeout_s):
            raise RuntimeError("机器人未在超时内进入 operational")


def _is_bimanual(robot: BaseRobot) -> bool:
    return robot.dof == 14


# ── 测试项 ─────────────────────────────────────────────────


def test_go_home(robot: BaseRobot) -> None:
    log.info("[go_home] 回零 ...")
    robot.go_home()
    obs = robot.observe()
    log.info(f"[go_home] 完成, q[:3]={_fmt(obs.joint_positions[:3])}")


def test_move_joint(robot: BaseRobot, *, amp_deg: float, velocity: float) -> None:
    """在 joint_0 上做 ±amp 摆动，验证 smoothstep 插值。双臂时左右同时动。"""
    obs = robot.observe()
    home_q = list(obs.joint_positions)
    dof = robot.dof
    amp = math.radians(amp_deg)
    bimanual = _is_bimanual(robot)

    # 正向
    target = list(home_q[:dof])
    target[0] = home_q[0] + amp
    if bimanual:
        target[7] = home_q[7] + amp
    label = "左右 joint_0" if bimanual else "joint_0"
    log.info(f"[move_joint] {label} +{amp_deg:.1f}deg (velocity={velocity:.2f} rad/s) ...")
    ok = robot.move_joint_position(target, velocity=velocity)
    obs = robot.observe()
    q = obs.joint_positions
    if bimanual:
        log.info(f"[move_joint] 到达={ok}, left_j0={q[0]:.4f}, right_j0={q[7]:.4f}")
    else:
        log.info(f"[move_joint] 到达={ok}, j0={q[0]:.4f}")

    # 反向
    target[0] = home_q[0] - amp
    if bimanual:
        target[7] = home_q[7] - amp
    log.info(f"[move_joint] {label} -{amp_deg:.1f}deg ...")
    ok = robot.move_joint_position(target, velocity=velocity)
    obs = robot.observe()
    q = obs.joint_positions
    if bimanual:
        log.info(f"[move_joint] 到达={ok}, left_j0={q[0]:.4f}, right_j0={q[7]:.4f}")
    else:
        log.info(f"[move_joint] 到达={ok}, j0={q[0]:.4f}")

    # 回原位
    target[0] = home_q[0]
    if bimanual:
        target[7] = home_q[7]
    robot.move_joint_position(target, velocity=velocity)
    log.info("[move_joint] 回原位完成")


def test_cartesian(robot: BaseRobot, *, delta_m: float, velocity: float) -> None:
    """在末端 x 方向施加小位移，验证 CARTESIAN IK 控制。双臂时左右同时动。"""
    from Robot.arx5 import _pose6d_to_pose7

    obs = robot.observe()
    bimanual = _is_bimanual(robot)

    if bimanual:
        ctrls = robot._ctrls  # type: ignore[attr-defined]
        left_eef = ctrls[0].get_eef_state()
        right_eef = ctrls[1].get_eef_state()
        left_pose7 = _pose6d_to_pose7(left_eef.pose_6d().tolist())
        right_pose7 = _pose6d_to_pose7(right_eef.pose_6d().tolist())
        if not left_pose7 or not right_pose7:
            log.info("[cartesian] 无法获取末端位姿，跳过")
            return
        q14 = obs.joint_positions
        base = left_pose7 + [q14[6]] + right_pose7 + [q14[13]]

        # 左右臂 x 同时 +delta
        fwd = list(base)
        fwd[0] += delta_m   # 左臂 x
        fwd[8] += delta_m   # 右臂 x
        log.info(f"[cartesian] 左右臂 x +{delta_m * 1000:.1f}mm ...")
        _stream_cartesian(robot, base, fwd, velocity=velocity)

        # 回原位
        _stream_cartesian(robot, fwd, base, velocity=velocity)
        log.info("[cartesian] 回原位完成")
    else:
        eef_pose = obs.eef_pose
        if len(eef_pose) < 7:
            log.info("[cartesian] 无法获取末端位姿，跳过")
            return
        gripper_now = obs.joint_positions[robot.dof]
        base = list(eef_pose[:7]) + [gripper_now]

        fwd = list(base)
        fwd[0] += delta_m
        log.info(f"[cartesian] x +{delta_m * 1000:.1f}mm ...")
        _stream_cartesian(robot, base, fwd, velocity=velocity)

        _stream_cartesian(robot, fwd, base, velocity=velocity)
        log.info("[cartesian] 回原位完成")


def _stream_cartesian(
    robot: BaseRobot,
    start: list[float],
    target: list[float],
    *,
    velocity: float,
) -> None:
    """smoothstep 插值笛卡尔位置并流式下发 CARTESIAN action。

    仅对平移分量计算位移和插值系数，姿态分量（四元数）在两端相同时自然保持不变。
    """
    # 只用平移分量（index 0-2，双臂还有 8-10）计算最大位移
    pos_indices = [0, 1, 2]
    if len(start) > 8:
        pos_indices += [8, 9, 10]
    max_disp = max(abs(target[i] - start[i]) for i in pos_indices)
    if max_disp < 1e-6:
        return

    duration = max_disp / velocity if velocity > 0 else 0.5
    dt = 1.0 / robot.get_params().control_frequency_hz
    steps = max(int(round(duration / dt)), 1)

    next_time = time.perf_counter()
    for i in range(1, steps + 1):
        t = i / steps
        alpha = t * t * (3.0 - 2.0 * t)  # smoothstep
        cmd = [s + (tgt - s) * alpha for s, tgt in zip(start, target)]
        robot.act(Action(ActionSpace.CARTESIAN, cmd))
        next_time += dt
        robot._rt_sleep_until(next_time)


def test_gripper(robot: BaseRobot, *, hold_s: float) -> None:
    """测试夹爪开合。"""
    obs = robot.observe()
    q = list(obs.joint_positions)
    bimanual = _is_bimanual(robot)

    if bimanual:
        p = robot.get_params()
        lg_max, rg_max = p.joint_position_max[6], p.joint_position_max[13]

        log.info(f"[gripper] 双臂张开 (left={lg_max:.3f}m, right={rg_max:.3f}m) ...")
        cmd = list(q)
        cmd[6], cmd[13] = lg_max, rg_max
        _hold_action(robot, cmd, hold_s)

        log.info("[gripper] 双臂闭合 ...")
        cmd[6], cmd[13] = 0.0, 0.0
        _hold_action(robot, cmd, hold_s)
        log.info("[gripper] 完成")
    else:
        enable_gripper = getattr(robot, "_enable_gripper", False)
        if not enable_gripper:
            log.info("[gripper] enable_gripper=false, 跳过夹爪测试")
            return
        gripper_width = getattr(robot, "_gripper_width", 0.08)

        log.info(f"[gripper] 张开 ({gripper_width:.3f}m) ...")
        cmd = q[:6] + [gripper_width]
        _hold_action(robot, cmd, hold_s)

        log.info("[gripper] 闭合 ...")
        cmd = q[:6] + [0.0]
        _hold_action(robot, cmd, hold_s)
        log.info("[gripper] 完成")


def _hold_action(robot: BaseRobot, cmd: list[float], duration: float) -> None:
    """持续下发同一 JOINT_POSITION 指令一段时间。"""
    dt = 1.0 / robot.get_params().control_frequency_hz
    steps = max(int(round(duration / dt)), 1)
    action = Action(ActionSpace.JOINT_POSITION, cmd)
    next_time = time.perf_counter()
    for _ in range(steps):
        robot.act(action)
        next_time += dt
        robot._rt_sleep_until(next_time)


def _fmt(vals) -> str:
    return ", ".join(f"{v:.4f}" for v in vals[:3])


def main() -> None:
    parser = argparse.ArgumentParser(description="ARX5 运动功能测试")
    parser.add_argument("config", help="YAML 配置文件路径")
    parser.add_argument("--skip", nargs="+", default=[], choices=["home", "joint", "cartesian", "gripper"],
                        help="跳过指定测试项")
    parser.add_argument("--amp-deg", type=float, default=10.0, help="关节摆动幅度 (度)")
    parser.add_argument("--delta-mm", type=float, default=30.0, help="笛卡尔位移 (mm)")
    parser.add_argument("--velocity", type=float, default=0.5, help="关节移动速度 (rad/s)")
    parser.add_argument("--cart-velocity", type=float, default=0.05, help="笛卡尔移动速度 (m/s)")
    parser.add_argument("--hold", type=float, default=1.0, help="夹爪保持时间 (秒)")
    parser.add_argument("--timeout", type=float, default=20.0, help="等待 operational 超时秒数")
    args = parser.parse_args()
    setup_run_logger(__file__, args.config)
    skip = set(args.skip)

    log.info(f"[init] 配置: {args.config}")
    with BaseRobot.from_config(args.config) as robot:
        wait_ready(robot, args.timeout)
        log.info(f"[init] 机器人 '{robot.name}' 已就绪 (dof={robot.dof})")

        if "home" not in skip:
            test_go_home(robot)

        if "joint" not in skip:
            test_move_joint(robot, amp_deg=args.amp_deg, velocity=args.velocity)

        if "cartesian" not in skip:
            test_cartesian(robot, delta_m=args.delta_mm / 1000.0, velocity=args.cart_velocity)

        if "gripper" not in skip:
            test_gripper(robot, hold_s=args.hold)

        log.info("[done] 所有测试完成")


if __name__ == "__main__":
    main()
