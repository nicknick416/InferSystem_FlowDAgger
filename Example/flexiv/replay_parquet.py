"""从 parquet 文件读取 action 数据并在机械臂上 replay。

数据格式 (来自 DataCollectionSystem/lerobot):
  action: List[float], 长度 8 = [joint_1, ..., joint_7, gripper] (rad)
    - joint_1..7: 关节角度 (rad)
    - gripper: 归一化宽度 [0, 1]，0=全闭, 1=全开

用法:
    python Example/flexiv/replay_parquet.py Config/rizon4_example.yaml /path/to/file.parquet
    python Example/flexiv/replay_parquet.py Config/rizon4_example.yaml /path/to/file.parquet --fps 30 --speed 0.5
    python Example/flexiv/replay_parquet.py Config/rizon4_example.yaml /path/to/file.parquet --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Core import Action, ActionSpace
from Core.logging import setup_run_logger
from Robot import BaseRobot

log = logging.getLogger(__name__)


def load_actions(parquet_path: str) -> list[list[float]]:
    """读取 parquet 文件，返回 actions 列表。"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("需要 pyarrow: pip install pyarrow")

    table = pq.ParquetFile(parquet_path).read()
    actions = table["action"].to_pylist()

    print(f"加载完成: {len(actions)} 帧, action 维度={len(actions[0])}")
    print(f"  首帧 (deg): {[round(np.degrees(v), 1) for v in actions[0][:7]]} gripper={actions[0][7]:.2f}")
    print(f"  末帧 (deg): {[round(np.degrees(v), 1) for v in actions[-1][:7]]} gripper={actions[-1][7]:.2f}")

    grip_vals = [a[7] for a in actions]
    print(f"  夹爪范围: [{min(grip_vals):.3f}, {max(grip_vals):.3f}]")

    return actions


def replay(
    config_path: str,
    parquet_path: str,
    fps: float = 30.0,
    speed: float = 1.0,
    gripper_force: float = 15.0,
    gripper_speed: float = 0.2,
    dry_run: bool = False,
) -> None:
    """在机械臂上 replay 动作序列。"""

    actions = load_actions(parquet_path)
    n_frames = len(actions)
    dt = 1.0 / (fps * speed)

    if dry_run:
        print(f"\n[DRY RUN] 共 {n_frames} 帧, fps={fps}, speed={speed}x, dt={dt:.4f}s")
        for i, act in enumerate(actions[:5]):
            print(f"  [{i:4d}] joints={[round(v, 4) for v in act[:7]]} gripper={act[7]:.3f}")
        print(f"...共 {n_frames} 帧")
        return

    # ── 1. 从配置创建机器人 ──
    print("[1] 从配置创建机器人...")
    with BaseRobot.from_config(config_path) as robot:
        if robot.is_fault():
            print("[!] 检测到故障，正在清除...")
            robot.clear_fault()
            time.sleep(2.0)
            if robot.is_fault():
                raise RuntimeError("无法清除机器人故障")

        robot.enable()
        if not robot.wait_until_operational(timeout_s=30.0):
            raise RuntimeError("机器人未能在超时时间内变为 operational")
        print(f"[1] 机器人 '{robot.name}' 已 operational")

        # ── 2. 初始化夹爪 ──
        print("[2] 初始化夹爪...")
        gripper = robot.create_gripper()
        gripper_ok = False
        gripper_max_width = 0.09

        if gripper is not None:
            try:
                gripper.connect()
                gripper.observe()
                params = robot.get_params()
                if params.gripper:
                    gripper_max_width = params.gripper.max_width
                gripper_ok = True
                print(f"[2] 夹爪就绪 (max_width={gripper_max_width:.4f}m)")
            except Exception as e:
                print(f"[2] 夹爪初始化失败 ({e})，跳过夹爪控制")

        try:
            # 设置首帧夹爪位置
            if gripper_ok and gripper is not None:
                target_width = actions[0][7] * gripper_max_width
                gripper.move(target_width, velocity=gripper_speed, force=gripper_force)
                print(f"[2] 夹爪移至首帧位置: {target_width*1000:.1f}mm")
                time.sleep(1.0)

            # ── 3. MoveJ 到第一帧位置 ──
            first_joints = actions[0][:7]
            print(f"[3] MoveJ 到起始位置 (deg): {[round(np.degrees(v), 1) for v in first_joints]}")
            robot.move_joint_position(first_joints, velocity=0.3)
            print("[3] 已到达起始位置")

            # ── 4. Replay ──
            print(f"[4] 开始 replay: {n_frames} 帧, 预计 {n_frames * dt:.1f}s")
            print("    按 Ctrl+C 可安全停止")

            frame_count = 0
            try:
                for i, act in enumerate(actions):
                    t_loop = time.perf_counter()

                    joint_targets = act[:7]
                    gripper_val = act[7]

                    # 发送关节位置
                    try:
                        action = Action(ActionSpace.JOINT_POSITION, list(joint_targets))
                        robot.act(action)
                    except Exception as e:
                        print(f"\n[!] 帧 {i}: act() 失败: {e}")
                        if robot.is_fault():
                            print("[!] 检测到故障，尝试清除...")
                            robot.clear_fault()
                            time.sleep(1.0)
                            if robot.is_fault():
                                print("[!] 无法清除故障，停止 replay")
                                break
                        continue

                    # 夹爪连续宽度控制
                    if gripper_ok and gripper is not None:
                        try:
                            target_width = gripper_val * gripper_max_width
                            gripper.move(target_width, velocity=gripper_speed, force=gripper_force)
                        except Exception as e:
                            if frame_count % 100 == 0:
                                print(f"  [{i:4d}] 夹爪控制失败: {e}")

                    frame_count += 1

                    # 进度显示 (~1Hz)
                    if i % int(fps) == 0:
                        state = robot.observe()
                        err = [
                            abs(joint_targets[j] - state.joint_positions[j])
                            for j in range(len(joint_targets))
                        ]
                        max_err_deg = max(err) * 180.0 / np.pi
                        grip_info = ""
                        if gripper_ok and gripper is not None:
                            try:
                                gs = gripper.observe()
                                grip_info = f" gripper_cmd={gripper_val:.3f} actual={gs.width/gripper_max_width:.3f}"
                            except Exception:
                                log.warning("夹爪状态读取失败", exc_info=True)
                        print(
                            f"  [{i:4d}/{n_frames}] "
                            f"t={i/fps:.1f}s "
                            f"max_err={max_err_deg:.2f}deg"
                            f"{grip_info}",
                            flush=True,
                        )

                    # 精确定时
                    elapsed = time.perf_counter() - t_loop
                    sleep_time = dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except KeyboardInterrupt:
                print(f"\n[!] 用户中断 (已执行 {frame_count}/{n_frames} 帧)")

            # ── 5. 完成 ──
            print("[5] Replay 完成")
            state = robot.observe()
            print(f"    最终关节 (deg): {[round(np.degrees(v), 1) for v in state.joint_positions]}")
            if gripper_ok and gripper is not None:
                gs = gripper.observe()
                print(f"    最终夹爪宽度: {gs.width:.4f}m")
            print(f"    实际执行帧数: {frame_count}/{n_frames}")

        finally:
            if gripper_ok and gripper is not None:
                gripper.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 parquet 文件 replay 录制的动作到机械臂"
    )
    parser.add_argument("config", help="YAML 配置文件路径")
    parser.add_argument("parquet_path", help="parquet 数据文件路径")
    parser.add_argument("--fps", type=float, default=30.0, help="录制时的帧率 (default: 30)")
    parser.add_argument("--speed", type=float, default=1.0, help="回放速度倍率 (default: 1.0)")
    parser.add_argument("--gripper-force", type=float, default=15.0, help="夹爪力 (N)")
    parser.add_argument("--gripper-speed", type=float, default=0.2, help="夹爪速度 (m/s)")
    parser.add_argument("--dry-run", action="store_true", help="仅读取打印数据，不连接机器人")
    return parser.parse_args()


if __name__ == "__main__":
    args = main()
    setup_run_logger(__file__, args.config)
    replay(
        config_path=args.config,
        parquet_path=args.parquet_path,
        fps=args.fps,
        speed=args.speed,
        gripper_force=args.gripper_force,
        gripper_speed=args.gripper_speed,
        dry_run=args.dry_run,
    )
