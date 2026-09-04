"""声明式配置 Schema — 一个 Pydantic model 对应一段 YAML。

所有默认值、类型约束、枚举限制集中在此文件，消费端直接用属性访问，
不再需要 .get() + 手动类型转换。

用法::

    from Core.config_schema import SystemConfig

    config = SystemConfig.from_yaml("Config/arx5_example.yaml")
    # config.robot          → Arx5 / Arx5Bimanual / Flexiv / Aloha / UR config
    # config.cameras        → dict[str, CameraConfig]
    # config.inference      → InferenceConfig | None
    # config.tactile        → dict[str, TactileConfig]
"""
from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


# ══════════════════════════════════════════════════════════════
#  枚举
# ══════════════════════════════════════════════════════════════


class ActionSpaceEnum(str, Enum):
    joint_position = "joint_position"
    joint_velocity = "joint_velocity"
    joint_torque = "joint_torque"
    cartesian = "cartesian"


class GripperModeEnum(str, Enum):
    binary = "binary"
    width = "width"
    raw = "raw"


class PolicyFormatEnum(str, Enum):
    normal = "normal"
    canonical = "canonical"


# ══════════════════════════════════════════════════════════════
#  关节限位 (共享)
# ══════════════════════════════════════════════════════════════


class JointLimitsConfig(BaseModel):
    """关节限位配置 (单位: 度/弧度混合，根据字段名区分)。"""
    position_min_deg: list[float] | None = None
    position_max_deg: list[float] | None = None
    velocity_max: list[float] | None = None
    acceleration_max: list[float] | None = None
    torque_max: list[float] | None = None


# ══════════════════════════════════════════════════════════════
#  机器人配置
# ══════════════════════════════════════════════════════════════


# ── ARX5 单臂 ────────────────────────────────────────────────


class Arx5ControlConfig(BaseModel):
    """ARX5 控制参数。

    关节限位请在顶层 ``joint_limits`` 字段设置，而非在此处。
    """
    # ── 控制器基础参数 ────────────────────────────────────────
    background_send_recv: bool = True    # 后台线程收发（建议保持 True）
    frequency_hz: float = 500.0          # 控制循环频率
    controller_dt: float = 0.002         # 单步时间间隔（s）
    over_current_cnt_max: int = 1000     # 过流保护计数
    log_level: str = "INFO"              # SDK 日志级别
    # ── 夹爪参数 ─────────────────────────────────────────────
    enable_gripper: bool = False         # 是否启用夹爪
    flip_gripper_sign: bool = False      # 反转夹爪运动方向
    gripper_kp: float = 4.0             # 夹爪位置增益
    gripper_kd: float = 0.24            # 夹爪速度阻尼
    gripper_closed_slack: float = 0.01  # 闭合判定余量（m）
    gripper_width: float | None = None  # 夹爪行程覆盖（m，None=SDK默认）
    gripper_torque_max: float | None = None
    gripper_vel_max: float | None = None
    gripper_open_readout: float | None = None   # SDK gripper_open_readout 覆盖
    gripper_readout_sign: int | None = None      # 读数符号（±1）
    gripper_binary: bool = False                 # 推理时用二值夹爪控制
    gripper_binary_threshold: float = 0.04       # 二值阈值（m）
    gripper_binary_close: float = 0.0            # 闭合目标（m）
    home_position_deg: list[float] | None = None  # Home 位置（度）

    model_config = {"extra": "allow"}


class Arx5RobotConfig(BaseModel):
    """ARX5 单臂机器人配置。"""
    type: Literal["arx5"]
    model: str = "X5"
    interface: str = "can0"
    interface_name: str | None = None  # 兼容旧字段名
    canable_serial: str | None = None
    name: str | None = None
    dof: int = 6
    control: Arx5ControlConfig = Arx5ControlConfig()
    joint_limits: JointLimitsConfig | None = None

    @property
    def resolved_interface(self) -> str:
        """优先用 interface_name（兼容旧配置），否则用 interface。"""
        return self.interface_name or self.interface


# ── ARX5 双臂 ────────────────────────────────────────────────


class Arx5ArmEndpointConfig(BaseModel):
    """双臂中单臂的端点配置。"""
    model: str = "X5"
    interface: str = "can0"
    interface_name: str | None = None
    canable_serial: str | None = None
    urdf_path: str | None = None

    @property
    def resolved_interface(self) -> str:
        return self.interface_name or self.interface


class Arx5BimanualControlConfig(BaseModel):
    """ARX5 双臂控制参数。

    只声明 SDK 不知道的控制策略参数。硬件参数（关节限位、gripper_width 等）
    由 SDK 在 connect() 时提供，不在此声明。如需覆盖 SDK 硬件参数，
    通过 extra 字段传入（如 left_gripper_open_readout、gripper_width）。
    关节限位请在顶层 ``joint_limits`` 字段设置，而非在此处。
    """
    # ── 控制器基础参数 ────────────────────────────────────────
    background_send_recv: bool = True
    controller_dt: float = 0.002
    over_current_cnt_max: int = 1000
    log_level: str = "INFO"
    # 左右臂关节 PD 增益相对 SDK default_kp/default_kd 的独立缩放。
    left_joint_kp_scale: float = Field(default=1.0, gt=0.0, le=2.0)
    left_joint_kd_scale: float = Field(default=1.0, gt=0.0, le=2.0)
    right_joint_kp_scale: float = Field(default=1.0, gt=0.0, le=2.0)
    right_joint_kd_scale: float = Field(default=1.0, gt=0.0, le=2.0)
    # ── 夹爪参数 ─────────────────────────────────────────────
    enable_gripper: bool = True
    flip_gripper_sign: bool = False
    gripper_kp: float = 5.0           # SDK 默认 5.0；右臂可用 right_gripper_kp 单独覆盖
    gripper_kd: float = 0.2           # SDK 默认 0.2；右臂可用 right_gripper_kd 单独覆盖
    right_gripper_kp: float | None = None
    right_gripper_kd: float | None = None
    gripper_closed_slack: float = 0.01
    gripper_binary: bool = False
    gripper_binary_threshold: float = 0.04
    gripper_binary_close: float = 0.0
    home_position_deg_14: list[float] | None = None  # 14 维 Home 位置（度）
    # 双臂 Cartesian IK 必须成对成功，避免一侧冻结、另一侧继续运动。
    cartesian_ik_atomic: bool = True
    # 精确目标失败时，左右臂沿各自当前→目标位姿使用相同进度系数重试。
    cartesian_ik_backoff_factors: list[float] = Field(
        default_factory=lambda: [0.5, 0.25, 0.125]
    )

    model_config = {"extra": "allow"}


class Arx5BimanualRobotConfig(BaseModel):
    """ARX5 双臂机器人配置。"""
    type: Literal["arx5_bimanual"]
    name: str | None = None
    left_arm: Arx5ArmEndpointConfig = Arx5ArmEndpointConfig()
    right_arm: Arx5ArmEndpointConfig = Arx5ArmEndpointConfig(interface="can1")
    control: Arx5BimanualControlConfig = Arx5BimanualControlConfig()
    joint_limits: JointLimitsConfig | None = None


# ── Flexiv ────────────────────────────────────────────────────


class FlexivGripperConfig(BaseModel):
    """Flexiv 夹爪配置。"""
    name: str = ""
    max_width: float = 0.1
    min_width: float = 0.0
    max_velocity: float = 0.2
    max_force: float = 30.0
    default_velocity: float = 0.1
    default_force: float = 30.0


class FlexivControlConfig(BaseModel):
    """Flexiv 控制参数。"""
    default_mode: str = "NRT_JOINT_POSITION"
    frequency_hz: float = 1000.0
    home_position_deg: list[float] | None = None
    home_velocity_scale: int = 50


class FlexivRobotConfig(BaseModel):
    """Flexiv 机器人配置。"""
    type: Literal["flexiv"]
    serial_number: str
    name: str | None = None
    network_interface_whitelist: list[str] = Field(default_factory=list)
    verbose: bool = True
    lite: bool = False
    gripper: FlexivGripperConfig = FlexivGripperConfig()
    control: FlexivControlConfig = FlexivControlConfig()
    joint_limits: JointLimitsConfig | None = None


# ── Aloha / Piper 双臂 ───────────────────────────────────────


class AlohaControlConfig(BaseModel):
    """Aloha/Piper 双臂控制参数。"""
    frequency_hz: float = 30.0

    model_config = {"extra": "allow"}


class AlohaRobotConfig(BaseModel):
    """Aloha/Piper 双臂机器人配置。"""
    type: Literal["aloha"]
    name: str | None = None
    can_interface_left: str = "can0"
    can_interface_right: str = "can1"
    speed_percent: int = 20
    enable_timeout: float = 5.0
    gripper_max_m: float = 0.07
    gripper_min_left_m: float | None = None
    gripper_max_left_m: float | None = None
    gripper_min_right_m: float | None = None
    gripper_max_right_m: float | None = None
    home_gripper_left_m: float | None = None
    home_gripper_right_m: float | None = None
    gripper_sdk_scale: float = 10_000_000.0
    gripper_sdk_max: int | None = None
    gripper_write_scale_left: float = 1.0
    gripper_write_scale_right: float = 1.0
    gripper_sdk_max_left: int | None = None
    gripper_sdk_max_right: int | None = None
    gripper_effort_left: int = 1000
    gripper_effort_right: int = 1000
    gripper_read_scale_left: float = 1.0
    gripper_read_scale_right: float = 1.0
    home_gripper_m: float | None = None
    control: AlohaControlConfig = AlohaControlConfig()

    model_config = {"extra": "allow"}


# ── UR / Robotiq ─────────────────────────────────────────────


class URControlConfig(BaseModel):
    """UR 控制参数。"""
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    frequency_hz: float = Field(
        default=125.0,
        validation_alias=AliasChoices("frequency_hz", "hz"),
    )
    home_position_deg: list[float] | None = None
    home_speed: float = 0.5
    home_acceleration: float = 0.5
    joint_step_limit: float = 0.05
    servoj_lookahead: float = 0.1
    servoj_gain: float = 300.0
    wait_step_reached: bool = True
    step_reached_tolerance: float = 0.01
    step_reached_poll_s: float = 0.005
    step_reached_timeout_s: float | None = None


class URGripperConfig(BaseModel):
    """UR 常用 Robotiq 夹爪配置。"""
    enabled: bool = True
    name: str = "robotiq"
    port: int = 63352
    speed: int = 255
    force: int = 100
    deadband: int = 0
    max_width: float = 0.085
    min_width: float = 0.0
    max_velocity: float = 0.2
    max_force: float = 100.0
    open_raw: int = 0
    close_raw: int = 255
    invert_set_open_close: bool = True

    model_config = {"extra": "allow"}


class URRobotConfig(BaseModel):
    """UR 机械臂配置。"""
    type: Literal["ur"]
    robot_ip: str = Field(
        validation_alias=AliasChoices("robot_ip", "ip", "host", "address"),
    )
    name: str | None = None
    dof: int = 6
    control: URControlConfig = URControlConfig()
    gripper: URGripperConfig = URGripperConfig()
    joint_limits: JointLimitsConfig | None = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


# ── Discriminated Union ──────────────────────────────────────

RobotConfig = Annotated[
    Union[
        Arx5RobotConfig,
        Arx5BimanualRobotConfig,
        FlexivRobotConfig,
        AlohaRobotConfig,
        URRobotConfig,
    ],
    Field(discriminator="type"),
]


# ══════════════════════════════════════════════════════════════
#  传感器配置
# ══════════════════════════════════════════════════════════════


class StreamConfig(BaseModel):
    """视频流配置。"""
    width: int = 640
    height: int = 480
    fps: int = 30
    encoding: str = "bgr8"
    aligned_to: str | None = None


class CameraConfig(BaseModel):
    """相机配置。"""
    type: str = "realsense"
    serial_number: str | None = None
    mapped_key: str | None = None
    # 手眼标定角色：eye_in_hand 求 T_EC（cam→gripper），eye_to_hand 求 T_BC（cam→base）；
    # 留空表示该相机不参与标定。
    calibration_role: Literal["eye_in_hand", "eye_to_hand"] | None = None
    enable_depth: bool = False
    align_depth_to_color: bool = True
    streams: dict[str, StreamConfig] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class TactileConfig(BaseModel):
    """触觉传感器配置。"""
    type: str = "opencv"
    mapped_key: str | None = None
    device_path: str | int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


# ══════════════════════════════════════════════════════════════
#  推理配置
# ══════════════════════════════════════════════════════════════


class SmoothConfig(BaseModel):
    """Action chunk temporal smoothing config."""
    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = False
    overlap_steps: int = Field(
        default=8,
        validation_alias=AliasChoices("overlap_steps", "min_smooth_steps"),
    )


class AsyncInferenceConfig(BaseModel):
    """Asynchronous inference loop config."""
    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = False
    obs_fps: float | None = None
    max_latency_steps: int = 8


class FlowDaggerConfig(BaseModel):
    """Human-intervention collection and episode-boundary training settings."""

    enabled: bool = False
    shadow_mode: bool = False
    run_stage: Literal["demonstration", "baseline", "shadow", "closed_loop"] = (
        "demonstration"
    )
    expert_fps: float = Field(default=30.0, gt=0.0, le=60.0)
    train_poll_interval_s: float = Field(default=2.0, gt=0.0)
    heartbeat_interval_s: float = Field(default=1.0, gt=0.0, le=5.0)
    max_chunk_age_s: float = Field(default=1.5, gt=0.0, le=5.0)
    teach_kp_scale: float = Field(default=0.0, ge=0.0, le=0.2)
    teach_kd_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    teach_gripper_kp_scale: float = Field(default=0.0, ge=0.0, le=0.2)
    teach_gripper_kd_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    left_gripper_open_m: float = Field(default=0.07, ge=0.0)
    left_gripper_close_m: float = Field(default=0.0, ge=0.0)
    right_gripper_open_m: float = Field(default=0.07, ge=0.0)
    right_gripper_close_m: float = Field(default=0.0, ge=0.0)


class InferenceConfig(BaseModel):
    """推理参数配置。"""
    model_config = ConfigDict(populate_by_name=True)

    # ── 网络 ──────────────────────────────────────────────────
    server: str = "127.0.0.1:5555"       # 推理服务器地址 host:port
    recv_timeout_ms: int = 30000          # 等待推理结果超时（ms）
    jpeg_quality: int = 90                # 图像压缩质量（1-100）
    prompt: str = ""                       # VLA language prompt
    # ── 控制节奏 ──────────────────────────────────────────────
    fps: int = 30                         # 控制循环频率（Hz）
    n_execute: int = 100                  # 每个 chunk 执行的 action 帧数（越小越快响应新推理）
    latency_compensation: bool = True     # 根据推理耗时跳过过期 action
    go_home_exit_speed_percent: int | None = None  # 退出/Ctrl+C 回 Home 专用速度百分比
    # ── 动作映射 ──────────────────────────────────────────────
    action_space: ActionSpaceEnum = ActionSpaceEnum.joint_position
    # arm_dof: 动作向量中手臂关节的维度。None = 自动从 robot.dof 读取。
    # 仅当模型输出的手臂维度与机器人 dof 不一致时才需要填写。
    arm_dof: int | None = Field(
        default=None,
        validation_alias=AliasChoices("arm_dof", "arm_dim"),
    )
    # gripper_action_index: 动作向量中夹爪维度的下标。
    # None = 自动推断（有夹爪时为 robot.dof，无夹爪时不控制）。
    gripper_action_index: int | None = Field(
        default=None,
        validation_alias=AliasChoices("gripper_action_index", "gripper_index"),
    )
    policy_format: PolicyFormatEnum = PolicyFormatEnum.normal
    canonical_dim: int = 32
    gripper_threshold: float = 0.5        # 二值夹爪的开合判定阈值
    gripper_mode: GripperModeEnum = GripperModeEnum.binary
    # 仅控制推理 action 分发；初始化、复位和显式夹爪操作仍默认等待完成。
    gripper_wait_during_inference: bool = False
    hold_gripper_closed_during_inference: bool = False
    open_gripper_on_inference_end: bool = False
    smooth: SmoothConfig = Field(default_factory=SmoothConfig)
    async_inference: AsyncInferenceConfig = Field(
        default_factory=AsyncInferenceConfig,
        validation_alias=AliasChoices("async", "async_inference"),
    )
    flowdagger: FlowDaggerConfig = Field(default_factory=FlowDaggerConfig)
    # ── 起始位置 ──────────────────────────────────────────────
    enabled_cameras: list[str] | None = None   # 参与推理的图像传感器白名单，None=全部
    inference_home: list[float] | None = None  # 推理前移动到的起始关节角（rad）
    inference_home_gripper_width: float | None = Field(
        default=None,
        ge=0.0,
        description="推理开始前夹爪移动到的宽度（m）；None 表示不控制",
    )
    # ── 安全守卫 ──────────────────────────────────────────────
    # 每个 chunk 执行前，检查首帧 action 与当前关节角的最大偏差。
    # 若超过此阈值则拒绝执行并停止循环，防止模型输出导致机械臂跳变。
    max_joint_delta_rad: float = 0.5  # 关节角跳变阈值（rad，约 28.6°）
    # EEF 模式首帧安全检查: 平移和姿态变化超过阈值则拒绝 chunk。
    max_eef_delta_m: float = 0.10
    max_eef_rotation_delta_rad: float = 0.8
    # EEF 每帧下发限幅：参考当前硬件 EEF，把绝对目标裁成单步小目标；<=0 禁用。
    max_eef_xyz_step: float = 0.0
    # Ignore sub-millimetre Cartesian target noise before applying the step limit.
    eef_translation_deadband_m: float = Field(default=0.0, ge=0.0, le=0.01)
    max_eef_rot_step_deg: float = 0.0
    max_eef_gripper_step: float = 0.0
    # 每步 dispatch 里对命令关节角速度的上限（rad/s）。
    # 若相邻命令的 |Δq|/dt 超过此值，按 sign·v_max·dt 裁切；None = 禁用。
    # 仅对 joint_position 生效。
    max_joint_velocity_rad_s: float | None = 1.2
    # ── 视频录制 (ffmpeg) ─────────────────────────────────────
    # 推理过程中的视频通过 subprocess 调用系统 ffmpeg 编码（需安装 ffmpeg）。
    video_codec: str = "libx264"          # 编码器：libx264 / libx265 / h264_videotoolbox 等
    video_crf: int = 28                   # 质量（仅 x264/x265 生效）：18=高清 23=默认 28=较小 越大体积越小
    video_preset: str = "veryfast"        # 速度档位：ultrafast/superfast/veryfast/faster/fast/medium

    @model_validator(mode="after")
    def validate_policy_format(self) -> InferenceConfig:
        if self.canonical_dim <= 0:
            raise ValueError("canonical_dim must be positive")
        if (
            self.policy_format == PolicyFormatEnum.canonical
            and self.action_space != ActionSpaceEnum.cartesian
        ):
            raise ValueError("policy_format=canonical only supports action_space=cartesian")
        return self


# ══════════════════════════════════════════════════════════════
#  顶层配置
# ══════════════════════════════════════════════════════════════


class LoggingConfig(BaseModel):
    """日志配置。"""
    console_level: str = "INFO"
    file_level: str = "DEBUG"


class SystemConfig(BaseModel):
    """系统顶层配置 — 对应一个完整的 YAML 文件。"""
    robot: RobotConfig
    cameras: dict[str, CameraConfig] = Field(default_factory=dict)
    tactile: dict[str, TactileConfig] = Field(default_factory=dict)
    inference: InferenceConfig | None = None
    logging: LoggingConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> SystemConfig:
        """从 YAML 文件加载并验证配置。"""
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)
