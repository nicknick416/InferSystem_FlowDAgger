#!/usr/bin/env python3
"""Replay bimanual EEF actions from a DataCollectionSystemV2 episode on ARX5.

The on-disk action layout is 20D::

    left [xyz3, column-major rot6d6, gripper1]
    right [xyz3, column-major rot6d6, gripper1]

The program is dry-run by default. Real motion requires ``--execute`` and two
operator confirmations. Any Cartesian command explicitly rejected by the ARX5
driver (including unrecoverable IK failure) terminates replay immediately.
"""
from __future__ import annotations

import argparse
import csv
import json
import select
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from Core import Action, ActionSpace
from Inference.action_processing import (
    blend_action_values,
    canonicalize_action_values,
    max_eef_action_delta,
)


_ARM_FIELDS = ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6")
_ACTION_COLUMNS = (
    tuple(f"left_tcp.{name}" for name in _ARM_FIELDS)
    + ("left_gripper.pos",)
    + tuple(f"right_tcp.{name}" for name in _ARM_FIELDS)
    + ("right_gripper.pos",)
)


@dataclass(slots=True)
class EefEpisode:
    """Validated absolute dual-arm EEF actions in on-disk rot6d format."""

    episode_dir: Path
    actions_rot6d: list[list[float]]
    timestamps_s: list[float]
    frame_indices: list[int]
    metadata: dict[str, Any]


@dataclass(slots=True, frozen=True)
class ContinuityStats:
    """Peak adjacent-frame Cartesian deltas after rot6d canonicalization."""

    max_translation_m: float
    max_translation_frame: int
    max_rotation_rad: float
    max_rotation_frame: int


@dataclass(slots=True, frozen=True)
class ReplayResult:
    """Outcome of the frame loop; a rejected IK target is never skipped."""

    completed: bool
    executed_frames: int
    rejected_frame: int | None = None
    stopped_by_user: bool = False


def load_eef_episode(
    episode_dir: str | Path,
    *,
    start: int | None = None,
    end: int | None = None,
) -> EefEpisode:
    """Load `actions.eef_pose/data.csv` as 20D dual-arm absolute actions."""
    root = Path(episode_dir).expanduser().resolve()
    csv_path = root / "actions.eef_pose" / "data.csv"
    metadata_path = root / "metadata.json"
    if not csv_path.is_file():
        raise FileNotFoundError(f"EEF 动作文件不存在: {csv_path}")

    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)

    timestamps_ms: list[float] = []
    actions: list[list[float]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        actual_columns = set(reader.fieldnames or ())
        required = {"timestamp_ms", *_ACTION_COLUMNS}
        missing = sorted(required - actual_columns)
        if missing:
            raise ValueError(f"actions.eef_pose 缺少列: {missing}")
        for row_index, row in enumerate(reader):
            try:
                timestamp_ms = float(row["timestamp_ms"])
                action = [float(row[name]) for name in _ACTION_COLUMNS]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"第 {row_index} 行含无法解析的数值") from exc
            values = np.asarray([timestamp_ms, *action], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"第 {row_index} 行含非有限数值")
            timestamps_ms.append(timestamp_ms)
            actions.append(action)

    if not actions:
        raise ValueError(f"EEF 动作数据为空: {csv_path}")
    if any(b <= a for a, b in zip(timestamps_ms, timestamps_ms[1:])):
        raise ValueError("timestamp_ms 必须严格递增")

    frame_indices = list(range(len(actions)))[start:end]
    selected_actions = actions[start:end]
    selected_timestamps = timestamps_ms[start:end]
    if not selected_actions:
        raise ValueError(f"帧范围为空: start={start}, end={end}")
    t0 = selected_timestamps[0]
    timestamps_s = [(timestamp - t0) / 1000.0 for timestamp in selected_timestamps]
    return EefEpisode(
        episode_dir=root,
        actions_rot6d=selected_actions,
        timestamps_s=timestamps_s,
        frame_indices=frame_indices,
        metadata=metadata,
    )


def canonicalize_episode_actions(
    actions_rot6d: list[list[float]],
) -> list[list[float]]:
    """Convert 20D dual-arm `[xyz, rot6d, grip]` rows into 16D pose7 rows."""
    return [
        canonicalize_action_values(action, ActionSpace.CARTESIAN)
        for action in actions_rot6d
    ]


def interpolate_bimanual_action(
    start_action: list[float], target_action: list[float], alpha: float
) -> list[float]:
    """Interpolate two 16D dual-arm actions using quaternion slerp."""
    if len(start_action) != 16 or len(target_action) != 16:
        raise ValueError("双臂 EEF 插值要求两个 16 维 action")
    t = float(alpha)
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"alpha 必须位于 [0, 1]，实际 {alpha}")
    return blend_action_values(
        start_action,
        target_action,
        weight_old=1.0 - t,
        action_space=ActionSpace.CARTESIAN,
    )


def validate_action_continuity(
    actions: list[list[float]],
    *,
    max_translation_m: float,
    max_rotation_rad: float,
) -> ContinuityStats:
    """Reject adjacent EEF jumps before connecting to hardware."""
    if not actions:
        raise ValueError("EEF action 序列为空")
    if max_translation_m <= 0.0 or max_rotation_rad <= 0.0:
        raise ValueError("EEF 连续性阈值必须为正数")
    for index, action in enumerate(actions):
        if len(action) != 16:
            raise ValueError(f"frame {index} 不是 16 维 canonical 双臂 EEF action")

    peak_translation = 0.0
    peak_translation_frame = 0
    peak_rotation = 0.0
    peak_rotation_frame = 0
    for index in range(1, len(actions)):
        previous = actions[index - 1]
        previous_eef = previous[:7] + previous[8:15]
        translation, rotation = max_eef_action_delta(previous_eef, actions[index])
        if translation > peak_translation:
            peak_translation = translation
            peak_translation_frame = index
        if rotation > peak_rotation:
            peak_rotation = rotation
            peak_rotation_frame = index
        if translation > max_translation_m or rotation > max_rotation_rad:
            raise ValueError(
                f"EEF 跳变过大: frame {index - 1} -> {index}, "
                f"translation={translation:.4f} m (limit={max_translation_m:.4f}), "
                f"rotation={rotation:.4f} rad (limit={max_rotation_rad:.4f})"
            )
    return ContinuityStats(
        max_translation_m=peak_translation,
        max_translation_frame=peak_translation_frame,
        max_rotation_rad=peak_rotation,
        max_rotation_frame=peak_rotation_frame,
    )


def scaled_recorded_gaps(
    timestamps_s: list[float], *, speed: float, max_gap_s: float
) -> list[float]:
    """Return per-frame sleeps; each is scaled independently to avoid catch-up."""
    if speed <= 0.0:
        raise ValueError("speed 必须大于 0")
    if max_gap_s <= 0.0:
        raise ValueError("max_gap_s 必须大于 0")
    return [
        min((end - begin) / speed, max_gap_s)
        for begin, end in zip(timestamps_s, timestamps_s[1:])
    ]


def _action_from_observation(state: Any) -> list[float]:
    eef = [float(value) for value in state.eef_pose]
    joints = [float(value) for value in state.joint_positions]
    if len(eef) < 14 or len(joints) < 14:
        raise ValueError("ARX5 双臂状态必须包含 14 维 EEF pose 和 14 维关节/夹爪状态")
    return eef[:7] + [joints[6]] + eef[7:14] + [joints[13]]


def approach_first_action(
    robot: Any,
    target_action: list[float],
    *,
    duration_s: float,
    rate_hz: float,
    sleep_fn: Callable[[float], Any] = time.sleep,
    should_stop: Callable[[], bool] = lambda: False,
    wait_if_paused: Callable[[], Any] = lambda: None,
) -> bool:
    """Smoothly approach the first EEF target; return False on reject/abort."""
    if duration_s <= 0.0 or rate_hz <= 0.0:
        raise ValueError("approach duration/rate 必须大于 0")
    start_action = _action_from_observation(robot.observe())
    steps = max(1, int(round(duration_s * rate_hz)))
    dt = duration_s / steps
    for step in range(steps):
        wait_if_paused()
        if should_stop():
            return False
        alpha = (step + 1) / steps
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        values = interpolate_bimanual_action(start_action, target_action, alpha)
        state = robot.observe()
        accepted = robot.act(Action(ActionSpace.CARTESIAN, values), state=state)
        if accepted is False:
            return False
        sleep_fn(dt)
    return True


def replay_canonical_actions(
    robot: Any,
    actions: list[list[float]],
    timestamps_s: list[float],
    *,
    speed: float,
    max_gap_s: float,
    sleep_fn: Callable[[float], Any] = time.sleep,
    should_stop: Callable[[], bool] = lambda: False,
    wait_if_paused: Callable[[], Any] = lambda: None,
    progress_fn: Callable[[int, int], Any] | None = None,
) -> ReplayResult:
    """Replay canonical EEF actions, aborting rather than skipping rejected IK."""
    if len(actions) != len(timestamps_s):
        raise ValueError("action 数量与 timestamp 数量不一致")
    gaps = scaled_recorded_gaps(timestamps_s, speed=speed, max_gap_s=max_gap_s)
    executed = 0
    total = len(actions)
    for frame_index, values in enumerate(actions):
        wait_if_paused()
        if should_stop():
            return ReplayResult(
                completed=False,
                executed_frames=executed,
                stopped_by_user=True,
            )
        state = robot.observe()
        accepted = robot.act(
            Action(ActionSpace.CARTESIAN, list(values)), state=state
        )
        if accepted is False:
            return ReplayResult(
                completed=False,
                executed_frames=executed,
                rejected_frame=frame_index,
            )
        executed += 1
        if progress_fn is not None:
            progress_fn(frame_index, total)
        if frame_index < len(gaps):
            # Always sleep the recorded interval after this frame. We do not
            # subtract controller latency, so a slow cycle never causes catch-up.
            sleep_fn(gaps[frame_index])
    return ReplayResult(completed=True, executed_frames=executed)


class KeyboardControl:
    """Non-blocking terminal controls: Space toggles pause and q requests stop."""

    def __init__(self) -> None:
        self._paused = threading.Event()
        self._stopped = threading.Event()

    def start(self) -> None:
        if not sys.stdin.isatty():
            return
        threading.Thread(target=self._listen, daemon=True).start()

    def stop(self) -> None:
        self._stopped.set()

    def should_stop(self) -> bool:
        return self._stopped.is_set()

    def wait_if_paused(self) -> None:
        while self._paused.is_set() and not self._stopped.is_set():
            time.sleep(0.05)

    def _listen(self) -> None:
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
        except Exception:
            return
        try:
            tty.setcbreak(fd)
            while not self._stopped.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not ready:
                    continue
                key = sys.stdin.read(1)
                if key == " ":
                    if self._paused.is_set():
                        self._paused.clear()
                        print("\n[键盘] 继续", flush=True)
                    else:
                        self._paused.set()
                        print("\n[键盘] 已暂停；空格继续，q 退出", flush=True)
                elif key.lower() == "q":
                    self._stopped.set()
                    print("\n[键盘] 请求退出，当前帧结束后停止", flush=True)
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass


def _confirm(prompt: str, *, assume_yes: bool) -> None:
    print(prompt)
    if assume_yes:
        print("[--yes] 已自动确认")
        return
    try:
        response = input("请输入 REPLAY 继续，其他输入取消: ").strip()
    except EOFError as exc:
        raise RuntimeError("非交互终端必须显式传 --yes 才能执行") from exc
    if response != "REPLAY":
        raise RuntimeError("操作员取消 replay")


def _validate_args(args: argparse.Namespace) -> None:
    if args.speed <= 0.0:
        raise ValueError("speed 必须大于 0")
    if args.start is not None and args.start < 0:
        raise ValueError("start 不能为负数")
    if args.end is not None and args.end < 0:
        raise ValueError("end 不能为负数")
    if args.start is not None and args.end is not None and args.end <= args.start:
        raise ValueError("end 必须大于 start")
    positive_names = (
        "max_gap_s",
        "max_step_translation_m",
        "max_step_rotation_rad",
        "approach_s",
        "approach_hz",
        "max_approach_translation_m",
        "max_approach_rotation_rad",
    )
    for name in positive_names:
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"{name} 必须大于 0")


def _print_preflight(
    episode: EefEpisode,
    canonical_actions: list[list[float]],
    continuity: ContinuityStats,
    *,
    speed: float,
) -> None:
    raw = np.asarray(episode.actions_rot6d, dtype=np.float64)
    recorded_duration = episode.timestamps_s[-1] if len(episode.timestamps_s) > 1 else 0.0
    task = episode.metadata.get("task_title", "")
    print("\n================ ARX5 EEF replay 离线预检 ================")
    print(f"episode: {episode.episode_dir}")
    if task:
        print(f"task: {task}")
    print(
        f"frames: {len(canonical_actions)} 帧；20D rot6d -> 16D pose7；"
        f"frame range: {episode.frame_indices[0]}..{episode.frame_indices[-1]}"
    )
    print(
        f"recorded duration: {recorded_duration:.3f}s；speed={speed:.3f}x；"
        f"预计 replay 数据段约 {recorded_duration / speed:.3f}s"
    )
    print(
        f"max adjacent translation: {continuity.max_translation_m:.5f}m "
        f"@ frame {continuity.max_translation_frame}"
    )
    print(
        f"max adjacent rotation: {continuity.max_rotation_rad:.5f}rad "
        f"@ frame {continuity.max_rotation_frame}"
    )
    print(
        f"gripper ranges: left=[{raw[:, 9].min():.5f}, {raw[:, 9].max():.5f}]m, "
        f"right=[{raw[:, 19].min():.5f}, {raw[:, 19].max():.5f}]m"
    )


def _execute_hardware(
    args: argparse.Namespace,
    episode: EefEpisode,
    canonical_actions: list[list[float]],
) -> int:
    from Robot import BaseRobot

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    _confirm(
        "\n阶段 1/2：即将连接 ARX5 双臂。\n"
        f"config={config_path}\n"
        "请清空工作空间、确认左右 CAN 对应关系、手握急停，并保证末端无外部接触。",
        assume_yes=args.yes,
    )

    robot = BaseRobot.from_config(config_path)
    if robot.robot_type != "arx5_bimanual":
        raise ValueError(
            f"EEF episode 是双臂 20D 动作，配置必须为 arx5_bimanual，实际 {robot.robot_type}"
        )
    keyboard = KeyboardControl()
    connected = False
    try:
        print("[连接] 初始化双臂控制器和 IK solver...")
        robot.connect()
        connected = True
        robot.enable()
        if not robot.wait_until_operational(timeout_s=20.0):
            raise RuntimeError("ARX5 未在 20 秒内进入 operational")
        if not args.no_home_before:
            print("[Home] replay 前回 Home...")
            robot.go_home()

        initial_state = robot.observe()
        approach_translation, approach_rotation = max_eef_action_delta(
            initial_state.eef_pose, canonical_actions[0]
        )
        print(
            f"[起点差值] translation={approach_translation:.4f}m, "
            f"rotation={approach_rotation:.4f}rad"
        )
        if approach_translation > args.max_approach_translation_m:
            raise ValueError(
                f"当前姿态到首帧位移 {approach_translation:.4f}m 超过 "
                f"{args.max_approach_translation_m:.4f}m"
            )
        if approach_rotation > args.max_approach_rotation_rad:
            raise ValueError(
                f"当前姿态到首帧转角 {approach_rotation:.4f}rad 超过 "
                f"{args.max_approach_rotation_rad:.4f}rad"
            )
        _confirm(
            "\n阶段 2/2：即将开始真实运动。\n"
            f"先用 {args.approach_s:.1f}s 平滑逼近首帧，然后按记录时间戳回放。\n"
            "空格暂停/继续，q 退出；任一侧 IK 最终失败会双臂保持并立即终止。",
            assume_yes=args.yes,
        )

        keyboard.start()
        print("[逼近] 平滑移动到首帧...")
        approached = approach_first_action(
            robot,
            canonical_actions[0],
            duration_s=args.approach_s,
            rate_hz=args.approach_hz,
            should_stop=keyboard.should_stop,
            wait_if_paused=keyboard.wait_if_paused,
        )
        if not approached:
            print("[终止] 首帧逼近被用户中止或 IK 被拒绝", file=sys.stderr)
            return 3

        progress_interval = max(1, int(episode.metadata.get("fps_config", 30)))

        def report_progress(index: int, total: int) -> None:
            if index % progress_interval == 0 or index + 1 == total:
                source_index = episode.frame_indices[index]
                print(f"[回放] {index + 1}/{total} (source frame {source_index})")

        print("[回放] 开始；控制周期变慢时不会追赶后续帧")
        result = replay_canonical_actions(
            robot,
            canonical_actions,
            episode.timestamps_s,
            speed=args.speed,
            max_gap_s=args.max_gap_s,
            should_stop=keyboard.should_stop,
            wait_if_paused=keyboard.wait_if_paused,
            progress_fn=report_progress,
        )
        if result.rejected_frame is not None:
            source_frame = episode.frame_indices[result.rejected_frame]
            print(
                f"[IK 终止] replay frame={result.rejected_frame}, "
                f"source frame={source_frame}；未继续执行后续 action",
                file=sys.stderr,
            )
            return 3
        if result.stopped_by_user:
            print(f"[用户终止] 已执行 {result.executed_frames} 帧")
            return 130
        print(f"[完成] 已执行 {result.executed_frames}/{len(canonical_actions)} 帧")
        return 0
    finally:
        keyboard.stop()
        if connected:
            print("[退出] 驱动将执行 disconnect 安全流程（ARX5 会尝试回 Home 后进入 damping）")
            robot.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="安全回放 DCSv2 ARX5 双臂 EEF episode（默认 dry-run）"
    )
    parser.add_argument("config", help="ARX5 双臂 YAML 配置")
    parser.add_argument("episode", help="包含 actions.eef_pose/data.csv 的 episode 目录")
    parser.add_argument("--start", type=int, default=None, help="起始帧（含）")
    parser.add_argument("--end", type=int, default=None, help="结束帧（不含）")
    parser.add_argument("--speed", type=float, default=1.0, help="回放速度倍率")
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=0.2,
        help="异常长时间戳间隔的最大等待时间",
    )
    parser.add_argument(
        "--max-step-translation-m", type=float, default=0.03,
        help="离线连续性检查：最大相邻帧末端位移",
    )
    parser.add_argument(
        "--max-step-rotation-rad", type=float, default=0.2,
        help="离线连续性检查：最大相邻帧末端转角",
    )
    parser.add_argument("--approach-s", type=float, default=5.0)
    parser.add_argument("--approach-hz", type=float, default=30.0)
    parser.add_argument(
        "--max-approach-translation-m", type=float, default=0.20,
        help="当前姿态到首帧允许的最大位移",
    )
    parser.add_argument(
        "--max-approach-rotation-rad", type=float, default=1.57,
        help="当前姿态到首帧允许的最大转角",
    )
    parser.add_argument("--no-home-before", action="store_true")
    parser.add_argument(
        "--execute", action="store_true", help="真实连接并运动；省略时只做 dry-run"
    )
    parser.add_argument(
        "--yes", action="store_true", help="跳过两次人工确认（首次实机运行不建议）"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
        episode = load_eef_episode(args.episode, start=args.start, end=args.end)
        canonical_actions = canonicalize_episode_actions(episode.actions_rot6d)
        continuity = validate_action_continuity(
            canonical_actions,
            max_translation_m=args.max_step_translation_m,
            max_rotation_rad=args.max_step_rotation_rad,
        )
        # Validate timing even in dry-run mode.
        scaled_recorded_gaps(
            episode.timestamps_s, speed=args.speed, max_gap_s=args.max_gap_s
        )
        _print_preflight(
            episode, canonical_actions, continuity, speed=args.speed
        )
        if not args.execute:
            print("\n[DRY RUN] 离线检查完成，未连接机器人；确认数据后添加 --execute。")
            return 0
        return _execute_hardware(args, episode, canonical_actions)
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl-C", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
