from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from typing import Any

from Core import Action, ActionSpace, ArmState, GripperParams, RobotParams
from Core.config_schema import FlexivRobotConfig
from Robot.base import BaseRobot
from Robot.gripper import BaseGripper, GripperState

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  FlexivGripper — 独立的夹爪设备
# ══════════════════════════════════════════════════════════════


class FlexivGripper(BaseGripper):
    """飞夕机械臂配套夹爪驱动。

    通过 FlexivRobot.create_gripper() 创建，需要机器人的 RDK 模块
    和底层句柄来建立通信。
    """

    _ASYNC_COMMAND_SETTLE_S = 0.05

    def __init__(
        self,
        rdk_module: Any,
        robot_handle: Any,
        *,
        name: str = "",
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name)
        self._rdk = rdk_module
        self._robot_handle = robot_handle
        self._config = config or {}

        self._gripper: Any | None = None
        self._connected = False

        # 默认参数
        self._default_velocity: float = self._config.get("default_velocity", 0.2)
        self._default_force: float = self._config.get("default_force", 20.0)
        self._max_width: float = self._config.get("max_width", 0.085)

    def connect(self) -> None:
        if self._connected:
            logger.info("夹爪已连接: %s", self.name or "(默认)")
            return
        self._gripper = self._rdk.Gripper(self._robot_handle)
        if self.name:
            self._gripper.Enable(self.name)
            tool = self._rdk.Tool(self._robot_handle)
            tool.Switch(self.name)
        self._gripper.Move(
            self._max_width,
            self._default_velocity,
            self._default_force,
        )
        self._wait_done()
        self._connected = True
        logger.info(
            "夹爪已连接并打开到最大宽度: %s width=%.4fm",
            self.name or "(默认)", self._max_width,
        )

    def disconnect(self) -> None:
        self._connected = False
        self._gripper = None

    def observe(self) -> GripperState:
        self._require_connected()
        gs = self._gripper.states()
        return GripperState(
            width=gs.width,
            max_width=self._max_width,
            force=gs.force,
            is_moving=gs.is_moving,
        )

    def set(self, open: bool, *, wait: bool = True) -> None:
        self._require_connected()
        if open:
            self._gripper.Move(
                self._max_width,
                self._default_velocity,
                self._default_force,
            )
        else:
            self._gripper.Grasp(self._default_force)
        self._wait_after_command(wait)

    def move(
        self,
        width: float,
        *,
        velocity: float | None = None,
        force: float | None = None,
        wait: bool = True,
    ) -> None:
        self._require_connected()
        self._gripper.Move(
            width,
            velocity if velocity is not None else self._default_velocity,
            force if force is not None else self._default_force,
        )
        self._wait_after_command(wait)

    # ── 内部 ──

    def _wait_after_command(self, wait: bool) -> None:
        if wait:
            self._wait_done()
        else:
            time.sleep(self._ASYNC_COMMAND_SETTLE_S)

    def _require_connected(self) -> None:
        if not self._connected or self._gripper is None:
            raise RuntimeError("夹爪尚未连接，请先调用 connect()")

    def _wait_done(self, timeout_s: float = 10.0) -> None:
        """等待夹爪运动完成。"""
        time.sleep(0.3)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self._gripper.states().is_moving:
                return
            time.sleep(0.05)
        logger.warning("夹爪运动等待超时 (%.1fs)", timeout_s)


# ══════════════════════════════════════════════════════════════
#  FlexivRobot
# ══════════════════════════════════════════════════════════════


@BaseRobot.register("flexiv")
class FlexivRobot(BaseRobot):
    """飞夕机械臂驱动 (Rizon 系列)。

    原语层:
      observe()  → 一次 states() 调用，原子快照
      act()      → 根据 ActionSpace 分发到对应的 RT 接口

    任务层 (覆盖基类默认实现，使用 Flexiv Primitive 更高效):
      move_joint_position()  → ExecutePrimitive("MoveJ")
      move_eef()             → ExecutePrimitive("MoveL")
      go_home()              → MoveJ 到 YAML 配置的 home_position_deg
    """

    # ── Rizon4 默认硬件参数 ───────────────────────────────────

    _DEFAULT_PARAMS = RobotParams(
        dof=7,
        joint_position_min=[-2.7925, -2.2689, -2.9671, -1.8675, -2.9671, -1.3963, -2.9671],
        joint_position_max=[+2.7925, +2.2689, +2.9671, +2.6878, +2.9671, +4.5379, +2.9671],
        joint_velocity_max=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        joint_acceleration_max=[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        joint_torque_max=[100.0, 100.0, 50.0, 50.0, 20.0, 20.0, 20.0],
        gripper=GripperParams(
            max_width=0.085,
            min_width=0.0,
            max_velocity=0.2,
            max_force=20.0,
        ),
        home_position=[0.0, -0.3491, 0.0, 1.5708, 0.0, 0.3491, 0.0],
        control_frequency_hz=1000.0,
    )

    # ── RDK 模式映射 (Flexiv 特有，不暴露到基类) ──────────────

    _MODE_MAP: dict[str, str] = {
        "primitive_execution": "NRT_PRIMITIVE_EXECUTION",
        "plan_execution": "NRT_PLAN_EXECUTION",
        "joint_position": "NRT_JOINT_POSITION",
        "joint_impedance": "NRT_JOINT_IMPEDANCE",
        "cartesian_motion_force": "NRT_CARTESIAN_MOTION_FORCE",
        "rt_joint_position": "NRT_JOINT_POSITION",
        "rt_joint_torque": "NRT_JOINT_TORQUE",
        "rt_cartesian_motion_force": "NRT_CARTESIAN_MOTION_FORCE",
    }

    # ActionSpace → 所需的 RDK 模式
    _ACTION_MODE: dict[ActionSpace, str] = {
        ActionSpace.JOINT_POSITION: "NRT_JOINT_POSITION",
        ActionSpace.JOINT_VELOCITY: "NRT_JOINT_POSITION",
        ActionSpace.JOINT_TORQUE:   "NRT_JOINT_TORQUE",
        ActionSpace.CARTESIAN:      "NRT_CARTESIAN_MOTION_FORCE",
    }

    _MODE_SWITCH_POLL_INTERVAL = 0.1
    _MODE_SWITCH_TIMEOUT = 5.0

    # ================================================================
    #  初始化
    # ================================================================

    def __init__(
        self,
        serial_number: str,
        *,
        name: str | None = None,
        network_interface_whitelist: Sequence[str] | None = None,
        verbose: bool = True,
        lite: bool = False,
        robot_handle: Any | None = None,
        params: RobotParams | None = None,
        gripper_config: dict[str, Any] | None = None,
        control_config: dict[str, Any] | None = None,
    ) -> None:
        import flexivrdk as rdk_module  # type: ignore[no-redef]

        self.serial_number = serial_number
        self._rdk = rdk_module
        self._robot_handle = robot_handle
        self._network_interface_whitelist = list(network_interface_whitelist or [])
        self._verbose = verbose
        self._lite = lite

        # 延迟到 connect() 中初始化
        self._robot: Any | None = None
        self._connected: bool = False

        # 参数
        self._params = params if params is not None else self._DEFAULT_PARAMS

        # 配置 (来自 YAML)
        self._gripper_config = gripper_config or {}
        self._control_config = control_config or {}

        # send_joint_velocity 使用的上次调用时间戳
        self._last_vel_send_time: float | None = None

        super().__init__(
            name=name or serial_number,
            robot_type="flexiv",
            dof=self._params.dof,
        )

    # ── 从 config 创建 ──

    @classmethod
    def _from_config_dict(cls, robot_cfg: FlexivRobotConfig | dict[str, Any]) -> FlexivRobot:
        """从 typed config 创建 FlexivRobot。也兼容旧的 dict 调用路径。"""
        if isinstance(robot_cfg, dict):
            robot_cfg = FlexivRobotConfig.model_validate(robot_cfg)

        return cls(
            serial_number=robot_cfg.serial_number,
            name=robot_cfg.name,
            network_interface_whitelist=robot_cfg.network_interface_whitelist,
            verbose=robot_cfg.verbose,
            lite=robot_cfg.lite,
            params=cls._build_params(robot_cfg),
            gripper_config=robot_cfg.gripper.model_dump(),
            control_config=robot_cfg.control.model_dump(),
        )

    @classmethod
    def _build_params(cls, cfg: FlexivRobotConfig) -> RobotParams:
        """从 typed config 构建 RobotParams，缺省字段使用默认值。"""
        default = cls._DEFAULT_PARAMS
        g = cfg.gripper
        ctrl = cfg.control
        lim = cfg.joint_limits

        gripper = GripperParams(
            max_width=g.max_width,
            min_width=g.min_width,
            max_velocity=g.max_velocity,
            max_force=g.max_force,
        ) if default.gripper else None

        home_rad = (
            [math.radians(d) for d in ctrl.home_position_deg]
            if ctrl.home_position_deg is not None
            else list(default.home_position)
        )

        return RobotParams(
            dof=default.dof,
            joint_position_min=(
                [math.radians(d) for d in lim.position_min_deg]
                if lim and lim.position_min_deg else list(default.joint_position_min)
            ),
            joint_position_max=(
                [math.radians(d) for d in lim.position_max_deg]
                if lim and lim.position_max_deg else list(default.joint_position_max)
            ),
            joint_velocity_max=(
                list(lim.velocity_max) if lim and lim.velocity_max
                else list(default.joint_velocity_max)
            ),
            joint_acceleration_max=(
                list(lim.acceleration_max) if lim and lim.acceleration_max
                else list(default.joint_acceleration_max)
            ),
            joint_torque_max=(
                list(lim.torque_max) if lim and lim.torque_max
                else list(default.joint_torque_max)
            ),
            gripper=gripper,
            home_position=home_rad,
            control_frequency_hz=ctrl.frequency_hz,
        )

    # ================================================================
    #  生命周期
    # ================================================================

    def connect(self) -> None:
        if self._connected:
            return
        if self._robot_handle is not None:
            self._robot = self._robot_handle
        else:
            self._robot = self._rdk.Robot(
                self.serial_number,
                self._network_interface_whitelist,
                self._verbose,
                self._lite,
            )
        # 验证实际连接
        if not self._robot.connected():
            self._robot = None
            raise RuntimeError(
                f"无法连接到机器人 '{self.name}' (SN={self.serial_number})，请检查网络和电源"
            )
        # 从硬件读取实际 DOF
        info = self._robot.info()
        self.dof = getattr(info, "DoF", self._params.dof) or self._params.dof
        self._connected = True
        logger.info("已连接到 %s (DOF=%d)", self.name, self.dof)

    def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            if self._robot.operational():
                self._robot.Stop()
        except Exception:
            logger.warning("断开连接时停止运动失败", exc_info=True)
        self._connected = False
        self._robot = None
        logger.info("已断开 %s", self.name)

    def enable(self) -> None:
        self._require_connected()
        self._robot.Enable()

    def stop(self) -> None:
        self._require_connected()
        self._robot.Stop()

    def emergency_stop(self) -> None:
        self._require_connected()
        try:
            self._robot.Stop()
        except Exception:
            logger.error("紧急停止调用异常", exc_info=True)
        logger.warning("紧急停止已执行: %s", self.name)

    def clear_fault(self) -> None:
        self._require_connected()
        self._robot.ClearFault()

    def get_params(self) -> RobotParams:
        return self._params

    def is_connected(self) -> bool:
        if not self._connected or self._robot is None:
            return False
        return bool(self._robot.connected())

    def is_operational(self) -> bool:
        self._require_connected()
        return bool(self._robot.operational())

    def is_busy(self) -> bool:
        self._require_connected()
        return bool(self._robot.busy())

    def is_fault(self) -> bool:
        self._require_connected()
        return bool(self._robot.fault())

    # ================================================================
    #  核心原语: observe()
    # ================================================================

    def observe(self) -> ArmState:
        """原子状态快照 — 一次 states() 调用获取全部状态。

        Flexiv tcp_pose 四元数顺序: 标量在前 [x,y,z, qw,qx,qy,qz]。
        """
        self._require_connected()
        s = self._robot.states()
        return ArmState(
            timestamp=time.perf_counter(),
            joint_positions=_to_list(s.q),
            joint_velocities=_to_list(s.dq),
            joint_torques=_to_list(s.tau),
            joint_external_torques=_to_list(s.tau_ext),
            joint_positions_desired=_to_list(s.theta),
            eef_pose=_to_list(s.tcp_pose),
            eef_velocity=_to_list(s.tcp_vel),
            wrench_in_tcp=_to_list(s.ext_wrench_in_tcp),
            wrench_in_world=_to_list(s.ext_wrench_in_world),
        )

    # ================================================================
    #  核心原语: act()
    # ================================================================

    def act(self, action: Action, *, state: ArmState | None = None) -> None:
        """执行一步动作，根据 ActionSpace 分发到对应 RT 接口。

        自动切换到所需的 RDK 模式 (首次切换后缓存，后续无开销)。
        """
        self._require_connected()

        # 自动切换模式
        required_mode = self._ACTION_MODE.get(action.space)
        if required_mode is None:
            raise ValueError(f"不支持的动作空间: {action.space}")
        self._ensure_mode(required_mode)

        if action.space == ActionSpace.JOINT_POSITION:
            self._act_joint_position(action)

        elif action.space == ActionSpace.JOINT_VELOCITY:
            self._act_joint_velocity(action, state)

        elif action.space == ActionSpace.JOINT_TORQUE:
            self._act_joint_torque(action)

        elif action.space == ActionSpace.CARTESIAN:
            self._act_cartesian(action)

    def _act_joint_position(self, action: Action) -> None:
        self._validate_length("action.values", action.values)
        self._validate_joint_positions(action.values)
        vel = action.extra.get("velocities", [0.0] * self.dof)
        max_vel = action.extra.get("max_vel", list(self._params.joint_velocity_max))
        max_acc = action.extra.get("max_acc", list(self._params.joint_acceleration_max))
        self._robot.SendJointPosition(list(action.values), vel, max_vel, max_acc)

    def _act_joint_velocity(self, action: Action, state: ArmState | None) -> None:
        self._validate_length("action.values", action.values)
        self._validate_joint_velocities(action.values)
        # 使用实际时间间隔做位置积分，避免累积误差
        now = time.perf_counter()
        if self._last_vel_send_time is not None:
            actual_dt = now - self._last_vel_send_time
        else:
            actual_dt = 1.0 / self._params.control_frequency_hz
        self._last_vel_send_time = now

        vel = [float(v) for v in action.values]
        q = state.joint_positions if state is not None else _to_list(self._robot.states().q)
        target = [q[i] + vel[i] * actual_dt for i in range(self.dof)]
        zeros = [0.0] * self.dof
        self._robot.SendJointPosition(target, vel, zeros, zeros)

    def _act_joint_torque(self, action: Action) -> None:
        self._validate_length("action.values", action.values)
        self._validate_joint_torques(action.values)
        self._robot.SendJointTorque(
            [float(t) for t in action.values], False, [0.0] * self.dof,
        )

    def _act_cartesian(self, action: Action) -> None:
        self._validate_length("action.values", action.values, expected=7)
        wrench = action.extra.get("wrench", [0.0] * 6)
        self._robot.SendCartesianMotionForce(list(action.values), wrench)

    # ================================================================
    #  末端工具工厂
    # ================================================================

    def create_gripper(
        self, config: dict[str, Any] | None = None,
    ) -> FlexivGripper:
        """创建与本机器人关联的夹爪。

        夹爪通过机器人的 RDK 模块和底层句柄建立通信。
        返回后需调用 gripper.connect() 初始化。
        """
        self._require_connected()
        cfg = config if config is not None else self._gripper_config
        return FlexivGripper(
            self._rdk,
            self._robot,
            name=cfg.get("name", ""),
            config=cfg,
        )

    # ================================================================
    #  任务层覆盖 (使用 Flexiv Primitive，比默认 RT 循环更高效)
    # ================================================================

    def move_joint_position(
        self,
        positions: Sequence[float],
        *,
        velocity: float | None = None,
        timeout_s: float = 30.0,
    ) -> bool:
        self._require_connected()
        self._validate_length("positions", positions)
        self._validate_joint_positions(positions)

        self._ensure_mode("NRT_PRIMITIVE_EXECUTION")
        if velocity is not None:
            self._robot.SetVelocityScale(_to_scale(velocity))
        # Flexiv NRT_PRIMITIVE_EXECUTION MoveJ expects joint targets in degrees.
        # Keep the public/internal API in radians and convert only at the RDK boundary.
        jpos = self._rdk.JPos([math.degrees(v) for v in positions])
        self._robot.ExecutePrimitive("MoveJ", {"target": jpos}, True)
        return self._wait_primitive_done(timeout_s)

    def move_eef(
        self,
        position: Sequence[float],
        orientation: Sequence[float] | None = None,
        *,
        velocity: float | None = None,

        timeout_s: float = 30.0,
    ) -> bool:
        self._require_connected()
        self._validate_length("position", position, expected=3)

        self._ensure_mode("NRT_PRIMITIVE_EXECUTION")
        if velocity is not None:
            self._robot.SetVelocityScale(_to_scale(velocity))
        if orientation is None:
            # 保持当前姿态; Flexiv tcp_pose: [x,y,z, qw,qx,qy,qz]
            tcp = _to_list(self._robot.states().tcp_pose)
            assert len(tcp) >= 7, f"tcp_pose 长度异常: 期望 >= 7，实际 {len(tcp)}"
            orientation = _quat_to_rotvec(tcp[3], tcp[4], tcp[5], tcp[6])
        coord = self._rdk.Coord(
            list(position), list(orientation), ["world", "world"],
        )
        self._robot.ExecutePrimitive("MoveL", {"target": coord}, True)
        return self._wait_primitive_done(timeout_s)

    def go_home(self, *, velocity: float | None = None, timeout_s: float = 60.0) -> bool:
        home = self.get_params().home_position
        if not home:
            raise RuntimeError("未配置 Home 位置")
        if velocity is None:
            vel_scale = float(self._control_config.get("home_velocity_scale", 50))
            velocity = max(1.0, min(100.0, vel_scale)) / 100.0
        return self.move_joint_position(
            home,
            velocity=velocity,
            timeout_s=timeout_s,
        )

    # ================================================================
    #  Flexiv 特有方法 (不在基类中)
    # ================================================================

    def switch_mode(self, mode_name: str) -> None:
        """切换 RDK 控制模式 (Flexiv 特有)。

        通常不需要手动调用 — act() 会自动切换到所需模式。
        """
        self._require_connected()
        rdk_name = self._MODE_MAP.get(mode_name, mode_name)
        try:
            rdk_mode = getattr(self._rdk.Mode, rdk_name)
        except AttributeError as exc:
            available = ", ".join(sorted(self._rdk.Mode.__members__.keys()))
            raise ValueError(
                f"未知模式 '{mode_name}'。可用: {available}"
            ) from exc
        self._robot.SwitchMode(rdk_mode)
        self._wait_mode(rdk_name)

    @property
    def native_handle(self) -> Any:
        """底层 RDK Robot 句柄，供需要直接访问 SDK 的场景使用。"""
        self._require_connected()
        return self._robot

    # ── 内部工具 ──────────────────────────────────────────────

    def _require_connected(self) -> None:
        if not self._connected or self._robot is None:
            raise RuntimeError(
                f"机器人 '{self.name}' 尚未连接，请先调用 connect() "
                f"或使用 with 语句"
            )

    def _ensure_mode(self, target_mode: str) -> None:
        """确保当前处于目标模式，不一致时自动切换并等待生效。"""
        current = _enum_name(self._robot.mode())
        if current != target_mode:
            self.switch_mode(target_mode)

    def _wait_mode(self, target_mode: str) -> None:
        deadline = time.monotonic() + self._MODE_SWITCH_TIMEOUT
        while time.monotonic() < deadline:
            current = _enum_name(self._robot.mode())
            if current == target_mode:
                return
            time.sleep(self._MODE_SWITCH_POLL_INTERVAL)
        current = _enum_name(self._robot.mode())
        if current != target_mode:
            raise RuntimeError(
                f"模式切换超时: 期望 '{target_mode}'，当前 '{current}'"
            )

    def _wait_primitive_done(self, timeout_s: float = 30.0) -> bool:
        """等待当前 Primitive 完成 (仅 NRT_PRIMITIVE_EXECUTION 模式)。"""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ps = self._robot.primitive_states()
            if ps.get("reachedTarget", 0) == 1:
                return True
            time.sleep(0.1)
        return self._robot.primitive_states().get("reachedTarget", 0) == 1


# ── 模块级工具函数 ────────────────────────────────────────────

def _enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return [value]


def _to_scale(value: float) -> int:
    """将 [0.0, 1.0] 速度比例转换为 [1, 100] 整数百分比。"""
    return max(1, min(100, int(value * 100)))


def _quat_to_rotvec(qw: float, qx: float, qy: float, qz: float) -> list[float]:
    """四元数 (标量在前) 转旋转向量。

    参数顺序与 Flexiv RDK tcp_pose 的四元数部分一致: [qw, qx, qy, qz]。
    """
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    sin_half = math.sqrt(qx * qx + qy * qy + qz * qz)
    if sin_half < 1e-10:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(sin_half, qw)
    scale = angle / sin_half
    return [qx * scale, qy * scale, qz * scale]
