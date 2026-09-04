"""从 parquet / LeRobot v3.0 / DataCollectionSystemV2 数据集读取动作并在 ARX5 上 replay（默认双臂）。

支持 action 维度:
  - 14 维: [lq0..lq5, lgrip, rq0..rq5, rgrip]（双臂）
  - 7 维: [q0..q5, gripper]（单臂）
  - 12 / 6 维: 仅关节，夹爪保持当前

用法:
    # 直接指定 parquet 文件
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml data.parquet

    # LeRobot v3.0 数据集目录（自动读取 info.json 中的 fps）
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml ~/ARX/005
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml ~/ARX/005 --episode 0
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml ~/ARX/005 --episode 2 --fps 30 --speed 0.5

    # DataCollectionSystemV2 episode 目录
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml ~/data/2026_04_08/lirui/fold_pants/episode_0001
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml ~/data/2026_04_08/lirui/fold_pants/episode_0001 --play-video

    # 同步播放数据集中的视频
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml ~/ARX/005 --episode 0 --play-video
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml ~/ARX/005 --episode 0 --play-video --cameras top left_wrist

    # 其他选项
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml data.parquet --no-home-before
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml data.parquet --home-after
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml data.parquet --column action --start 100 --end 500
    python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml data.parquet --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from Core import Action, ActionSpace
from Core.logging import setup_run_logger
from Robot import BaseRobot

# 14 维关节名称
JOINT_NAMES_14 = [
    "L_j0", "L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_grip",
    "R_j0", "R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_grip",
]


log = logging.getLogger(__name__)


def save_tracking_log(
    out_dir: Path,
    timestamps: list[float],
    targets: list[list[float]],
    actuals: list[list[float]],
) -> Path:
    """将目标与实际关节状态保存为 CSV 文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "tracking_log.csv"
    n_dim = len(targets[0]) if targets else 0
    names = JOINT_NAMES_14[:n_dim] if n_dim <= 14 else [f"j{i}" for i in range(n_dim)]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["frame", "time_s"]
        for n in names:
            header += [f"target_{n}", f"actual_{n}", f"error_{n}"]
        writer.writerow(header)

        t0 = timestamps[0] if timestamps else 0.0
        for idx, (ts, tgt, act) in enumerate(zip(timestamps, targets, actuals)):
            row: list = [idx, f"{ts - t0:.4f}"]
            for j in range(n_dim):
                err = tgt[j] - act[j]
                row += [f"{tgt[j]:.6f}", f"{act[j]:.6f}", f"{err:.6f}"]
            writer.writerow(row)

    log.info(f"  跟踪日志已保存: {csv_path}")
    return csv_path


def plot_tracking(
    out_dir: Path,
    timestamps: list[float],
    targets: list[list[float]],
    actuals: list[list[float]],
) -> None:
    """画出每个关节的目标 vs 实际 + 误差子图，保存为 PNG。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning(" matplotlib 未安装，跳过画图 (pip install matplotlib)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    t_arr = np.array(timestamps)
    t_arr = t_arr - t_arr[0]
    tgt_arr = np.array(targets)   # (N, D)
    act_arr = np.array(actuals)   # (N, D)
    err_arr = tgt_arr - act_arr
    n_dim = tgt_arr.shape[1]
    names = JOINT_NAMES_14[:n_dim] if n_dim <= 14 else [f"j{i}" for i in range(n_dim)]

    # ---------- 全关节总览图 ----------
    n_cols = 2
    n_rows = (n_dim + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)
    axes = axes.flatten()
    for j in range(n_dim):
        ax = axes[j]
        ax.plot(t_arr, np.degrees(tgt_arr[:, j]) if j not in (6, 13) else tgt_arr[:, j],
                label="target", linewidth=0.8, alpha=0.9)
        ax.plot(t_arr, np.degrees(act_arr[:, j]) if j not in (6, 13) else act_arr[:, j],
                label="actual", linewidth=0.8, alpha=0.9)
        unit = "m" if j in (6, 13) else "deg"
        ax.set_ylabel(f"{names[j]} ({unit})")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
    for j in range(n_dim, len(axes)):
        axes[j].set_visible(False)
    axes[-2 if n_dim % 2 == 0 else -1].set_xlabel("Time (s)")
    if n_dim > 1:
        axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joint Tracking: Target vs Actual", fontsize=14)
    fig.tight_layout()
    overview_path = out_dir / "tracking_overview.png"
    fig.savefig(overview_path, dpi=150)
    plt.close(fig)
    log.info(f"  总览图已保存: {overview_path}")

    # ---------- 误差图 ----------
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)
    axes2 = axes2.flatten()
    for j in range(n_dim):
        ax = axes2[j]
        err_vals = np.degrees(err_arr[:, j]) if j not in (6, 13) else err_arr[:, j]
        ax.plot(t_arr, err_vals, linewidth=0.8, color="red", alpha=0.8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        unit = "m" if j in (6, 13) else "deg"
        ax.set_ylabel(f"{names[j]} err ({unit})")
        rms = np.sqrt(np.mean(err_vals ** 2))
        ax.set_title(f"RMS={rms:.4f} {unit}", fontsize=9)
        ax.grid(True, alpha=0.3)
    for j in range(n_dim, len(axes2)):
        axes2[j].set_visible(False)
    axes2[-2 if n_dim % 2 == 0 else -1].set_xlabel("Time (s)")
    if n_dim > 1:
        axes2[-1].set_xlabel("Time (s)")
    fig2.suptitle("Joint Tracking Error (target - actual)", fontsize=14)
    fig2.tight_layout()
    error_path = out_dir / "tracking_error.png"
    fig2.savefig(error_path, dpi=150)
    plt.close(fig2)
    log.info(f"  误差图已保存: {error_path}")

    # ---------- PD 诊断摘要 ----------
    log.info("\n  ══ PD 跟踪诊断摘要 ══")
    for j in range(n_dim):
        if j in (6, 13):
            rms = np.sqrt(np.mean(err_arr[:, j] ** 2)) * 1000  # mm
            peak = np.max(np.abs(err_arr[:, j])) * 1000
            log.info(f"  {names[j]:>8s}: RMS={rms:7.2f} mm, Peak={peak:7.2f} mm")
        else:
            rms = np.degrees(np.sqrt(np.mean(err_arr[:, j] ** 2)))
            peak = np.degrees(np.max(np.abs(err_arr[:, j])))
            log.info(f"  {names[j]:>8s}: RMS={rms:7.3f} deg, Peak={peak:7.3f} deg")
    log.info("")


def _find_video_keys(info: dict) -> list[str]:
    """从 info.json 的 features 中提取所有 video 类型的 key。"""
    keys = []
    for k, v in info.get("features", {}).items():
        if v.get("dtype") == "video":
            keys.append(k)
    return sorted(keys)


def _resolve_video_path(
    dataset_dir: Path,
    info: dict,
    video_key: str,
    chunk_index: int,
    file_index: int,
) -> Path:
    """根据 info.json 的 video_path 模板解析实际视频文件路径。"""
    tmpl = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    rel = tmpl.format(video_key=video_key, chunk_index=chunk_index, file_index=file_index)
    return dataset_dir / rel


class VideoPlayer:
    """同步播放多路视频，每次 show_frame(i) 显示第 i 帧。"""

    def __init__(self, video_paths: dict[str, Path]) -> None:
        import cv2
        self._cv2 = cv2
        self._caps: dict[str, cv2.VideoCapture] = {}
        for name, path in video_paths.items():
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                log.warning(f" 无法打开视频: {path}")
                continue
            self._caps[name] = cap
        if self._caps:
            names = ", ".join(self._caps.keys())
            log.info(f"视频播放器已打开: {names}")

    def show_frame(self, frame_idx: int) -> bool:
        """显示第 frame_idx 帧，返回 False 表示用户按 q 退出。"""
        cv2 = self._cv2
        for name, cap in self._caps.items():
            # 如果当前位置不是目标帧，seek 过去
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if cur != frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                cv2.imshow(name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def release(self) -> None:
        for cap in self._caps.values():
            cap.release()
        if self._caps:
            self._cv2.destroyAllWindows()

    @property
    def active(self) -> bool:
        return len(self._caps) > 0


def _is_lerobot_dataset(path: Path) -> bool:
    """判断路径是否为 LeRobot v3.0 数据集目录。"""
    return path.is_dir() and (path / "meta" / "info.json").exists()


def _load_lerobot_info(dataset_dir: Path) -> dict:
    """读取 LeRobot 数据集的 info.json。"""
    with open(dataset_dir / "meta" / "info.json") as f:
        return json.load(f)


def _load_lerobot_episode_meta(dataset_dir: Path) -> list[dict]:
    """从 meta/episodes/ 读取所有 episode 元信息，按 episode_index 排序返回。"""
    import pyarrow.parquet as pq

    episodes_dir = dataset_dir / "meta" / "episodes"
    episodes = []
    for chunk_dir in sorted(episodes_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue
        for pf in sorted(chunk_dir.glob("*.parquet")):
            table = pq.read_table(pf, columns=["episode_index", "length", "data/chunk_index", "data/file_index"])
            for i in range(table.num_rows):
                episodes.append({
                    "episode_index": table["episode_index"][i].as_py(),
                    "length": table["length"][i].as_py(),
                    "chunk_index": table["data/chunk_index"][i].as_py(),
                    "file_index": table["data/file_index"][i].as_py(),
                })
    episodes.sort(key=lambda e: e["episode_index"])
    return episodes


def load_actions_from_lerobot(
    dataset_dir: Path,
    *,
    episode: int | None = None,
    column: str = "action",
    start: int | None = None,
    end: int | None = None,
) -> list[list[float]]:
    """从 LeRobot v3.0 数据集目录加载动作序列。

    如果指定 episode，只加载该 episode 的帧；否则加载所有帧。
    """
    import pyarrow.parquet as pq

    episodes_meta = _load_lerobot_episode_meta(dataset_dir)
    if not episodes_meta:
        raise ValueError(f"数据集 {dataset_dir} 中无 episode 元信息")

    available_eps = [e["episode_index"] for e in episodes_meta]
    log.info(f"数据集共 {len(episodes_meta)} 个 episode: {available_eps}")

    if episode is not None:
        targets = [e for e in episodes_meta if e["episode_index"] == episode]
        if not targets:
            raise ValueError(f"episode {episode} 不存在，可用: {available_eps}")
    else:
        targets = episodes_meta

    # 按 (chunk_index, file_index) 分组，避免重复读同一个文件
    file_groups: dict[tuple[int, int], list[int]] = {}
    for ep in targets:
        key = (ep["chunk_index"], ep["file_index"])
        file_groups.setdefault(key, []).append(ep["episode_index"])

    all_actions: list[list[float]] = []
    for (chunk_idx, file_idx), ep_indices in sorted(file_groups.items()):
        parquet_path = dataset_dir / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {parquet_path}")
        table = pq.read_table(parquet_path, columns=[column, "episode_index", "frame_index"])

        for ep_idx in sorted(ep_indices):
            # 筛选该 episode 的行并按 frame_index 排序
            mask = [table["episode_index"][i].as_py() == ep_idx for i in range(table.num_rows)]
            indices = [i for i, m in enumerate(mask) if m]
            indices.sort(key=lambda i: table["frame_index"][i].as_py())
            for i in indices:
                all_actions.append(table[column][i].as_py())

    all_actions = all_actions[start:end]
    log.info(f"加载完成: {len(all_actions)} 帧, action_dim={len(all_actions[0]) if all_actions else '?'}")
    return all_actions


def load_actions(
    parquet_path: str,
    column: str = "action",
    start: int | None = None,
    end: int | None = None,
) -> list[list[float]]:
    """从单个 parquet 文件加载动作序列。

    Args:
        parquet_path: parquet 文件路径。
        column: 动作列名（默认 "action"）。
        start: 起始帧索引（含），None 表示从头。
        end: 结束帧索引（不含），None 表示到尾。
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("需要 pyarrow: pip install pyarrow") from exc

    table = pq.read_table(parquet_path, columns=[column])
    actions = table[column].to_pylist()
    if not actions:
        raise ValueError(f"parquet 列 '{column}' 为空")

    actions = actions[start:end]
    log.info(f"加载完成: {len(actions)} 帧, action_dim={len(actions[0])}")
    return actions


def _is_dcsv2_dataset(path: Path) -> bool:
    """判断路径是否为 DataCollectionSystemV2 episode 目录。"""
    return path.is_dir() and (path / "metadata.json").exists()


def _load_dcsv2_metadata(episode_dir: Path) -> dict:
    """读取 DataCollectionSystemV2 的 metadata.json。"""
    with open(episode_dir / "metadata.json") as f:
        return json.load(f)


def _read_csv_numeric_rows(csv_path: Path) -> list[list[float]]:
    """Read a DCSv2 CSV, skipping timestamp_ms in the first column."""
    rows: list[list[float]] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            rows.append([float(v) for v in row[1:]])
    return rows


def _interleave_joints_and_gripper(
    jp: list[float], grip: list[float] | None
) -> list[float]:
    """Build a 12/14D dual-arm joint action from DCSv2 joint/gripper rows."""
    if len(jp) == 14:
        return jp
    if len(jp) == 12:
        if grip is not None and len(grip) == 2:
            return jp[:6] + [grip[0]] + jp[6:12] + [grip[1]]
        return jp
    if grip is not None:
        raise ValueError(
            f"无法合并 DCSv2 关节/夹爪: joint_dim={len(jp)}, gripper_dim={len(grip)}"
        )
    raise ValueError(f"DCSv2 关节维数必须是 12 或 14，实际 {len(jp)}")


def load_actions_from_dcsv2(
    episode_dir: Path,
    *,
    start: int | None = None,
    end: int | None = None,
) -> list[list[float]]:
    """从 DataCollectionSystemV2 episode 目录加载动作序列。

    优先读取 ``actions.joint_position``（已是 12/14 维）。
    否则合并 observation.state.joint_position 与 observation.state.gripper。
    当前采集布局的 joint CSV 已含夹爪，得到 14 维::

        [left_j1..j6, left_gripper, right_j1..j6, right_gripper]
    """
    action_csv = episode_dir / "actions.joint_position" / "data.csv"
    if action_csv.exists():
        actions = [
            _interleave_joints_and_gripper(jp, None)
            for jp in _read_csv_numeric_rows(action_csv)
        ]
    else:
        jp_csv = episode_dir / "observation.state.joint_position" / "data.csv"
        grip_csv = episode_dir / "observation.state.gripper" / "data.csv"
        if not jp_csv.exists():
            raise FileNotFoundError(f"关节数据不存在: {jp_csv}")
        jp_rows = _read_csv_numeric_rows(jp_csv)
        grip_rows = _read_csv_numeric_rows(grip_csv) if grip_csv.exists() else []
        if not jp_rows:
            raise ValueError("数据为空")
        n = len(jp_rows) if not grip_rows else min(len(jp_rows), len(grip_rows))
        actions = [
            _interleave_joints_and_gripper(
                jp_rows[i], grip_rows[i] if grip_rows else None
            )
            for i in range(n)
        ]

    if not actions:
        raise ValueError("数据为空")
    actions = actions[start:end]
    if not actions:
        raise ValueError(f"帧范围为空: start={start}, end={end}")
    log.info(f"加载完成: {len(actions)} 帧, action_dim={len(actions[0])}")
    return actions


def _find_dcsv2_video_keys(episode_dir: Path) -> list[str]:
    """扫描 DataCollectionSystemV2 episode 目录下的视频。"""
    keys = []
    for d in sorted(episode_dir.iterdir()):
        if d.is_dir() and d.name.startswith("observation.image."):
            video_file = d / "video.mp4"
            if video_file.exists():
                keys.append(d.name)
    return keys


def replay(
    config_path: str,
    data_path: str,
    *,
    fps: float | None = None,
    speed: float = 1.0,
    column: str = "action",
    episode: int | None = None,
    start: int | None = None,
    end: int | None = None,
    home_before: bool = True,
    home_after: bool = False,
    dry_run: bool = False,
    play_video: bool = False,
    cameras: list[str] | None = None,
    log_dir: str | None = None,
) -> None:
    """执行 replay 主流程。"""
    data_p = Path(data_path).expanduser().resolve()
    info: dict | None = None
    dcsv2_meta: dict | None = None
    episode_meta: dict | None = None
    dataset_type = "unknown"

    if _is_lerobot_dataset(data_p):
        dataset_type = "lerobot"
        info = _load_lerobot_info(data_p)
        if fps is None:
            fps = float(info.get("fps", 30.0))
            log.info(f"从 info.json 读取 fps={fps}")
        actions = load_actions_from_lerobot(data_p, episode=episode, column=column, start=start, end=end)
        # 获取目标 episode 的 chunk/file 信息（用于定位视频文件）
        if play_video and episode is not None:
            all_eps = _load_lerobot_episode_meta(data_p)
            episode_meta = next((e for e in all_eps if e["episode_index"] == episode), None)
    elif _is_dcsv2_dataset(data_p):
        dataset_type = "dcsv2"
        dcsv2_meta = _load_dcsv2_metadata(data_p)
        if fps is None:
            fps = float(dcsv2_meta.get("fps_config", dcsv2_meta.get("fps_actual", 30.0)))
            log.info(f"从 metadata.json 读取 fps={fps}")
        log.info(f"DataCollectionSystemV2 数据集: {dcsv2_meta.get('task_title', '')} "
            f"(episode={dcsv2_meta.get('episode_id', '?')}, "
            f"total_frames={dcsv2_meta.get('total_frames', '?')})")
        actions = load_actions_from_dcsv2(data_p, start=start, end=end)
    else:
        dataset_type = "parquet"
        if fps is None:
            fps = 30.0
        if play_video:
            log.warning(" --play-video 仅支持 LeRobot v3.0 / DCSv2 数据集目录，忽略")
            play_video = False
        actions = load_actions(str(data_p), column=column, start=start, end=end)

    n_frames = len(actions)
    dt = 1.0 / (fps * speed)

    if dry_run:
        log.info(f"[DRY RUN] n_frames={n_frames}, fps={fps}, speed={speed}x, dt={dt:.4f}s")
        for i, act in enumerate(actions[:5]):
            log.info(f"  [{i:4d}] dim={len(act)} values={[round(float(v), 4) for v in act[:8]]}")
        if play_video:
            if dataset_type == "lerobot" and info is not None:
                all_video_keys = _find_video_keys(info)
            elif dataset_type == "dcsv2":
                all_video_keys = _find_dcsv2_video_keys(data_p)
            else:
                all_video_keys = []
            log.info(f"  数据集视频: {all_video_keys}")
        return

    # 准备视频播放器
    video_player: VideoPlayer | None = None
    if play_video:
        # 收集所有可用视频 key
        if dataset_type == "lerobot" and info is not None:
            all_video_keys = _find_video_keys(info)
        elif dataset_type == "dcsv2":
            all_video_keys = _find_dcsv2_video_keys(data_p)
        else:
            all_video_keys = []

        # 用户指定 cameras 时做短名匹配（如 "top" 匹配 "observation.images.top"）
        if cameras:
            selected = []
            for cam in cameras:
                matched = [k for k in all_video_keys if cam in k]
                if matched:
                    selected.extend(matched)
                else:
                    log.warning(f" 未找到匹配 '{cam}' 的视频，可用: {all_video_keys}")
            video_keys = list(dict.fromkeys(selected))  # 去重保序
        else:
            video_keys = all_video_keys

        if video_keys:
            video_paths: dict[str, Path] = {}
            if dataset_type == "lerobot" and info is not None:
                chunk_idx = episode_meta["chunk_index"] if episode_meta else 0
                file_idx = episode_meta["file_index"] if episode_meta else 0
                for vk in video_keys:
                    vp = _resolve_video_path(data_p, info, vk, chunk_idx, file_idx)
                    if vp.exists():
                        short = vk.replace("observation.images.", "")
                        video_paths[short] = vp
                    else:
                        log.warning(f" 视频文件不存在: {vp}")
            elif dataset_type == "dcsv2":
                for vk in video_keys:
                    vp = data_p / vk / "video.mp4"
                    if vp.exists():
                        short = vk.replace("observation.image.", "")
                        video_paths[short] = vp
                    else:
                        log.warning(f" 视频文件不存在: {vp}")
            if video_paths:
                video_player = VideoPlayer(video_paths)

    # 准备记录目录
    if log_dir:
        tracking_dir = Path(log_dir).expanduser().resolve()
    else:
        tracking_dir = Path(data_path).expanduser().resolve()
        if tracking_dir.is_file():
            tracking_dir = tracking_dir.parent
        tracking_dir = tracking_dir / "replay_tracking" / datetime.now().strftime("%Y%m%d_%H%M%S")

    # 跟踪数据缓冲
    track_timestamps: list[float] = []
    track_targets: list[list[float]] = []
    track_actuals: list[list[float]] = []

    try:
        log.info("[1] 从配置创建机器人...")
        with BaseRobot.from_config(config_path) as robot:
            if robot.is_fault():
                log.info("[!] 检测到故障，clear_fault...")
                robot.clear_fault()
                time.sleep(1.5)
            robot.enable()
            if not robot.wait_until_operational(timeout_s=20.0):
                raise RuntimeError("机器人未能在超时时间内变为 operational")
            log.info(f"[2] 机器人 '{robot.name}' 已 operational (dof={robot.dof})")

            # 打印当前 PD 增益信息（辅助调试）
            if hasattr(robot, '_ctrls') and robot._ctrls is not None:
                for idx, tag in enumerate(("left", "right")):
                    gain = robot._ctrls[idx].get_gain()
                    kp = np.array(gain.kp())
                    kd = np.array(gain.kd())
                    log.info(f"  [{tag}] joint kp={kp}, kd={kd}")
                    log.info(f"  [{tag}] gripper kp={gain.gripper_kp:.3f}, kd={gain.gripper_kd:.3f}")

            # replay 前先回 Home，从已知安全位置出发
            if home_before:
                log.info("[3] 回 Home（避免从任意姿态直接跳到第一帧）...")
                robot.go_home()

            log.info(f"[4] 开始 replay: {n_frames} 帧, fps={fps}, speed={speed}x")
            log.info(f"    跟踪数据将保存到: {tracking_dir}")
            frame_count = 0
            log_interval = max(int(fps), 1)
            # start 偏移量，用于视频帧对齐
            frame_offset = start or 0

            for i, act in enumerate(actions):
                t_loop = time.perf_counter()
                values = [float(v) for v in act]
                try:
                    robot.act(Action(ActionSpace.JOINT_POSITION, values))
                except Exception as exc:
                    log.info(f"[!] 帧 {i} 执行失败: {exc}")
                    break
                frame_count += 1

                # 每帧记录目标和实际状态
                s = robot.observe()
                track_timestamps.append(t_loop)
                track_targets.append(values[:len(s.joint_positions)])
                track_actuals.append(list(s.joint_positions))

                if video_player and video_player.active:
                    if not video_player.show_frame(frame_offset + i):
                        log.info("[!] 用户按 q 退出视频播放")
                        break

                if i % log_interval == 0:
                    q = s.joint_positions
                    if len(q) >= 14:
                        err = [abs(values[j] - q[j]) for j in range(14)]
                        max_err_idx = int(np.argmax(err))
                        log.info(
                            f"  [{i:4d}/{n_frames}] "
                            f"lj0={q[0]:.4f} rj0={q[7]:.4f} lg={q[6]:.4f} rg={q[13]:.4f} "
                            f"| max_err={np.degrees(err[max_err_idx]):.2f}deg @{JOINT_NAMES_14[max_err_idx]}"
                        )
                    else:
                        log.info(f"  [{i:4d}/{n_frames}] j0={q[0]:.4f}")

                elapsed = time.perf_counter() - t_loop
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            log.info(f"[5] Replay 完成: {frame_count}/{n_frames} 帧")

            if home_after:
                log.info("[6] 回 Home...")
                robot.go_home()
                log.info("[7] Home 完成")

        # replay 结束后保存跟踪数据并画图
        if track_timestamps:
            log.info("[8] 保存跟踪数据与分析图...")
            save_tracking_log(tracking_dir, track_timestamps, track_targets, track_actuals)
            plot_tracking(tracking_dir, track_timestamps, track_targets, track_actuals)
    finally:
        if video_player is not None:
            video_player.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 parquet / LeRobot v3.0 数据集 replay ARX5 动作")
    parser.add_argument("config", help="YAML 配置文件路径")
    parser.add_argument("data_path", help="parquet 文件路径 或 LeRobot v3.0 数据集目录")
    parser.add_argument("--fps", type=float, default=None, help="回放帧率 (LeRobot 数据集默认从 info.json 读取，否则 30)")
    parser.add_argument("--speed", type=float, default=1.0, help="回放速度倍率 (默认 1.0)")
    parser.add_argument("--column", type=str, default="action", help="parquet 中的动作列名 (默认 action)")
    parser.add_argument("--episode", type=int, default=None, help="LeRobot 数据集: 指定 episode 索引 (默认全部)")
    parser.add_argument("--start", type=int, default=None, help="起始帧索引（含）")
    parser.add_argument("--end", type=int, default=None, help="结束帧索引（不含）")
    parser.add_argument("--no-home-before", action="store_true", help="跳过 replay 前的 go_home")
    parser.add_argument("--home-after", action="store_true", help="replay 结束后回 Home")
    parser.add_argument("--dry-run", action="store_true", help="只打印动作信息，不连接机器人")
    parser.add_argument("--play-video", action="store_true", help="同步播放数据集中的视频 (仅 LeRobot 数据集)")
    parser.add_argument("--cameras", nargs="*", default=None, help="指定播放的相机名 (如 top left_wrist)，默认播放全部")
    parser.add_argument("--log-dir", type=str, default=None, help="跟踪数据保存目录 (默认: 数据目录/replay_tracking/<时间戳>)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_run_logger(__file__, args.config)
    replay(
        config_path=args.config,
        data_path=args.data_path,
        fps=args.fps,
        speed=args.speed,
        column=args.column,
        episode=args.episode,
        start=args.start,
        end=args.end,
        home_before=not args.no_home_before,
        home_after=args.home_after,
        dry_run=args.dry_run,
        play_video=args.play_video,
        cameras=args.cameras,
        log_dir=args.log_dir,
    )
