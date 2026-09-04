from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """加载 YAML 配置文件（返回原始字典）。"""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path):
    """加载 YAML 并返回经过 Pydantic 验证的 SystemConfig。"""
    from Core.config_schema import SystemConfig

    return SystemConfig.from_yaml(path)


# ── 传感器数据 ────────────────────────────────────────────────

@dataclass(slots=True)
class SensorFrame:
    """传感器统一输出帧。所有传感器 (相机、触觉) 都输出此类型。"""
    sensor_name: str
    sensor_type: str
    timestamp: datetime = field(default_factory=utc_now)
    payload: dict[str, Any] = field(default_factory=dict)


# ── 机器人参数 ────────────────────────────────────────────────

@dataclass(slots=True)
class GripperParams:
    """夹爪硬件参数。"""
    max_width: float            # 最大开口宽度 (m)
    min_width: float = 0.0      # 最小开口宽度 (m)
    max_velocity: float = 0.0   # 最大运动速度 (m/s)
    max_force: float = 0.0      # 最大夹持力 (N)


@dataclass(slots=True)
class RobotParams:
    """机器人硬件参数 — 描述机器人的物理能力和极限。"""
    dof: int
    joint_position_min: list[float]         # 各关节最小位置 (rad)
    joint_position_max: list[float]         # 各关节最大位置 (rad)
    joint_velocity_max: list[float]         # 各关节最大速度 (rad/s)
    joint_acceleration_max: list[float]     # 各关节最大加速度 (rad/s²)
    joint_torque_max: list[float]           # 各关节最大力矩 (Nm)
    gripper: GripperParams | None = None    # 夹爪参数，无夹爪时为 None
    home_position: list[float] = field(default_factory=list)  # Home 位置 (rad)
    control_frequency_hz: float = 1000.0    # 实时控制频率 (Hz)


# ── 原子状态快照 ──────────────────────────────────────────────

@dataclass(slots=True)
class ArmState:
    """机械臂在某一时刻的完整状态快照。

    一次硬件读取填充所有字段，保证时间一致性。
    子类不支持的字段保留空列表。
    """
    timestamp: float                                            # perf_counter 时间戳
    joint_positions: list[float] = field(default_factory=list)  # rad
    joint_velocities: list[float] = field(default_factory=list) # rad/s
    joint_torques: list[float] = field(default_factory=list)    # Nm
    joint_external_torques: list[float] = field(default_factory=list)  # Nm
    joint_positions_desired: list[float] = field(default_factory=list) # rad
    eef_pose: list[float] = field(default_factory=list)         # [x,y,z,qw,qx,qy,qz]
    eef_velocity: list[float] = field(default_factory=list)     # [vx,vy,vz,wx,wy,wz]
    wrench_in_tcp: list[float] = field(default_factory=list)    # [fx,fy,fz,tx,ty,tz]
    wrench_in_world: list[float] = field(default_factory=list)  # [fx,fy,fz,tx,ty,tz]


# ── 统一动作表达 ──────────────────────────────────────────────

class ActionSpace(Enum):
    """动作空间枚举。"""
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE   = "joint_torque"
    CARTESIAN      = "cartesian"


@dataclass(slots=True)
class Action:
    """统一动作指令。

    space:  动作空间
    values: 动作值 (长度和语义由 space 决定)
    extra:  可选附加参数 (如 velocities, max_vel, wrench 等)

    示例::

        # 关节位置
        Action(ActionSpace.JOINT_POSITION, [0.0, -0.3, 0.0, 1.5, 0.0, 0.3, 0.0])

        # 关节位置 + 前馈速度
        Action(ActionSpace.JOINT_POSITION, target_q,
               extra={"velocities": target_dq, "max_vel": max_v, "max_acc": max_a})

        # 笛卡尔 + 力
        Action(ActionSpace.CARTESIAN, [x, y, z, qw, qx, qy, qz],
               extra={"wrench": [fx, fy, fz, tx, ty, tz]})
    """
    space: ActionSpace
    values: list[float]
    extra: dict[str, Any] = field(default_factory=dict)


# ── 推理系统统一输入 ──────────────────────────────────────────

@dataclass(slots=True)
class Observation:
    """推理系统的统一感知输入。

    聚合单一时刻的所有感知数据，作为推理模型的输入:

      images:  各相机/触觉传感器图像 {name: ndarray}
      state:   机器人状态向量 (维度和语义由上层定义)
      prompt:  语言指令 (VLA 等语言条件模型使用)
      extra:   扩展字段 (点云、音频、力传感器等未来输入)

    示例::

        Observation(
            images={"main_cam": img_bgr, "wrist_left": img_bgr},
            state=[q1, q2, ..., q7, gripper_width],
            prompt="pick up the red cup",
        )
    """
    images: dict[str, Any] = field(default_factory=dict)
    state: list[float] = field(default_factory=list)
    prompt: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=_time.perf_counter)
