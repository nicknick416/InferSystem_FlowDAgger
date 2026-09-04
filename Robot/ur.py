from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from typing import Any

from Core import Action, ActionSpace, ArmState, GripperParams, RobotParams
from Core.config_schema import URRobotConfig
from Robot.base import BaseRobot
from Robot.gripper import BaseGripper, GripperState

logger = logging.getLogger(__name__)


class URRobotiqGripper(BaseGripper):
    """UR 机械臂常见 Robotiq 夹爪驱动。"""

    def __init__(
        self,
        robot_ip: str,
        *,
        name: str = "",
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name)
        self._robot_ip = robot_ip
        self._config = config or {}
        self._gripper: Any | None = None
        self._connected = False

        self._port = int(self._config.get("port", 63352))
        self._speed = int(self._config.get("speed", 255))
        self._force = int(self._config.get("force", 100))
        self._deadband = int(self._config.get("deadband", 0))
        self._max_width = float(self._config.get("max_width", 0.085))
        self._min_width = float(self._config.get("min_width", 0.0))

        # Robotiq 原始位姿刻度: 默认 0=打开, 255=闭合
        self._open_raw = int(self._config.get("open_raw", 0))
        self._close_raw = int(self._config.get("close_raw", 255))
        # For compatibility with policy outputs that use 0=open,1=close while
        # upper layers may call set(True)=open by contract.
        self._invert_set_open_close = bool(self._config.get("invert_set_open_close", True))
        self._last_raw: int | None = None

    def connect(self) -> None:
        if self._connected:
            return
        try:
            from robotiq_gripper import RobotiqGripper  # type: ignore
        except Exception:
            try:
                from ur_collector.robotiq_gripper import RobotiqGripper  # type: ignore
            except Exception:
                try:
                    from SDK.ur.robotiq_gripper import RobotiqGripper  # type: ignore
                except Exception as e:
                    raise ModuleNotFoundError(
                        "Cannot import RobotiqGripper. Install `robotiq_gripper`, or provide `ur_collector` / `SDK.ur` on PYTHONPATH."
                    ) from e

        self._gripper = RobotiqGripper()
        self._gripper.connect(self._robot_ip, port=self._port)

        # 兼容不同版本 activate() 签名
        try:
            self._gripper.activate(
                calibration_cache_path="/tmp/robotiq_gripper_calibration.json"
            )
        except TypeError:
            self._gripper.activate()

        self._connected = True
        try:
            self._last_raw = int(self._call_first("get_current_position", "getCurrentPosition", default=0))
        except Exception:
            self._last_raw = None

        logger.info("UR Robotiq 夹爪已连接: %s:%d", self._robot_ip, self._port)

    def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            if self._gripper is not None:
                self._call_first("disconnect", default=None)
        finally:
            self._connected = False
            self._gripper = None

    def observe(self) -> GripperState:
        self._require_connected()
        raw = int(self._call_first("get_current_position", "getCurrentPosition", default=self._open_raw))
        force = float(self._call_first("get_current_force", "getCurrentForce", default=0.0))
        is_moving = bool(self._call_first("is_moving", "isMoving", default=False))
        return GripperState(
            width=self._raw_to_width(raw),
            max_width=self._max_width,
            force=force,
            is_moving=is_moving,
        )

    def set(self, open: bool, *, wait: bool = True) -> None:
        self._require_connected()
        # Robotiq's backend owns its command completion semantics.
        _ = wait
        logical_open = (not open) if self._invert_set_open_close else open
        raw = self._open_raw if logical_open else self._close_raw
        self.move_raw(raw)

    def move(
        self,
        width: float,
        *,
        velocity: float | None = None,
        force: float | None = None,
        wait: bool = True,
    ) -> None:
        self._require_connected()
        # Robotiq's backend owns its command completion semantics.
        _ = wait
        raw = self._width_to_raw(width)
        speed = self._to_speed_cmd(velocity)
        cmd_force = self._to_force_cmd(force)
        self.move_raw(raw, speed=speed, force=cmd_force)

    def move_normalized(
        self,
        value: float,
        *,
        velocity: float | None = None,
        force: float | None = None,
    ) -> bool:
        """连续夹爪控制: 默认按 0=open, 1=close 解释。"""
        self._require_connected()
        value = float(max(0.0, min(1.0, value)))
        raw_f = self._open_raw + value * (self._close_raw - self._open_raw)
        raw = int(round(raw_f))
        if self._last_raw is not None and abs(raw - self._last_raw) < self._deadband:
            return False
        speed = self._to_speed_cmd(velocity)
        cmd_force = self._to_force_cmd(force)
        return self.move_raw(raw, speed=speed, force=cmd_force)

    def move_raw(self, raw: int, *, speed: int | None = None, force: int | None = None) -> bool:
        self._require_connected()
        cmd_speed = self._speed if speed is None else int(max(0, min(255, speed)))
        cmd_force = self._force if force is None else int(max(0, min(255, force)))
        result = self._call_first(
            "move",
            args=(int(raw),),
            kwargs={"speed": cmd_speed, "force": cmd_force},
            default=(False, int(raw)),
        )

        ok = False
        if isinstance(result, (tuple, list)) and len(result) >= 1:
            ok = bool(result[0])
        elif isinstance(result, bool):
            ok = result
        elif result is not None:
            ok = True

        if ok:
            self._last_raw = int(raw)
        return ok

    # ── 内部 ──────────────────────────────────────────────────

    def _require_connected(self) -> None:
        if not self._connected or self._gripper is None:
            raise RuntimeError("夹爪尚未连接，请先调用 connect()")

    def _call_first(
        self,
        *names: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        default: Any = None,
    ) -> Any:
        assert self._gripper is not None
        kwargs = kwargs or {}
        for name in names:
            fn = getattr(self._gripper, name, None)
            if callable(fn):
                return fn(*args, **kwargs)
        return default

    def _to_speed_cmd(self, velocity: float | None) -> int:
        if velocity is None:
            return self._speed
        v = float(velocity)
        if v <= 1.0:
            return int(max(0, min(255, round(v * 255.0))))
        return int(max(0, min(255, round(v))))

    def _to_force_cmd(self, force: float | None) -> int:
        if force is None:
            return self._force
        f = float(force)
        if f <= 1.0:
            return int(max(0, min(255, round(f * 255.0))))
        return int(max(0, min(255, round(f))))

    def _width_to_raw(self, width: float) -> int:
        width = float(max(self._min_width, min(self._max_width, width)))
        if abs(self._max_width - self._min_width) < 1e-9:
            return self._open_raw
        ratio = (self._max_width - width) / (self._max_width - self._min_width)
        raw = self._open_raw + ratio * (self._close_raw - self._open_raw)
        return int(round(max(min(self._open_raw, self._close_raw), min(max(self._open_raw, self._close_raw), raw))))

    def _raw_to_width(self, raw: int) -> float:
        span_raw = float(self._close_raw - self._open_raw)
        if abs(span_raw) < 1e-9:
            return self._max_width
        ratio = (raw - self._open_raw) / span_raw
        return float(self._max_width - ratio * (self._max_width - self._min_width))


@BaseRobot.register("ur")
class URRobot(BaseRobot):
    """UR 机械臂驱动 (基于 ur_rtde)。"""

    _DEFAULT_PARAMS = RobotParams(
        dof=6,
        joint_position_min=[-2.0 * math.pi] * 6,
        joint_position_max=[2.0 * math.pi] * 6,
        joint_velocity_max=[3.14] * 6,
        joint_acceleration_max=[6.0] * 6,
        joint_torque_max=[150.0] * 6,
        gripper=GripperParams(
            max_width=0.085,
            min_width=0.0,
            max_velocity=0.2,
            max_force=100.0,
        ),
        home_position=[3.14159, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
        control_frequency_hz=125.0,
    )

    def __init__(
        self,
        robot_ip: str,
        *,
        name: str | None = None,
        params: RobotParams | None = None,
        control_config: dict[str, Any] | None = None,
        gripper_config: dict[str, Any] | None = None,
        rtde_control_module: Any | None = None,
        rtde_receive_module: Any | None = None,
        rtde_control_handle: Any | None = None,
        rtde_receive_handle: Any | None = None,
    ) -> None:
        self.robot_ip = robot_ip
        self._params = params if params is not None else self._DEFAULT_PARAMS
        self._control_config = control_config or {}
        self._gripper_config = gripper_config or {}

        self._rtde_control_module = rtde_control_module
        self._rtde_receive_module = rtde_receive_module

        self._rtde_c: Any | None = rtde_control_handle
        self._rtde_r: Any | None = rtde_receive_handle

        self._connected = False
        self._last_cmd_q: list[float] | None = None
        self._last_vel_send_time: float | None = None

        super().__init__(
            name=name or robot_ip,
            robot_type="ur",
            dof=self._params.dof,
        )

    @classmethod
    def _from_config_dict(cls, robot_cfg: URRobotConfig | dict[str, Any]) -> URRobot:
        if isinstance(robot_cfg, dict):
            robot_cfg = URRobotConfig.model_validate(robot_cfg)

        ip = robot_cfg.robot_ip
        if not ip:
            raise ValueError("UR 配置缺少 robot.ip / robot.robot_ip / robot.host")

        return cls(
            robot_ip=ip,
            name=robot_cfg.name,
            params=cls._build_params(robot_cfg),
            control_config=robot_cfg.control.model_dump(exclude_none=True),
            gripper_config=robot_cfg.gripper.model_dump(exclude_none=True),
        )

    @classmethod
    def _build_params(cls, robot_cfg: URRobotConfig | dict[str, Any]) -> RobotParams:
        if isinstance(robot_cfg, dict):
            robot_cfg = URRobotConfig.model_validate(robot_cfg)

        default = cls._DEFAULT_PARAMS
        control_cfg = robot_cfg.control
        gripper_cfg = robot_cfg.gripper
        limits_cfg = robot_cfg.joint_limits

        dof = int(robot_cfg.dof)

        pos_min_deg = limits_cfg.position_min_deg if limits_cfg else None
        pos_max_deg = limits_cfg.position_max_deg if limits_cfg else None

        if dof != default.dof:
            joint_position_min = [-2.0 * math.pi] * dof
            joint_position_max = [2.0 * math.pi] * dof
            joint_velocity_max = [3.14] * dof
            joint_acceleration_max = [6.0] * dof
            joint_torque_max = [150.0] * dof
        else:
            joint_position_min = list(default.joint_position_min)
            joint_position_max = list(default.joint_position_max)
            joint_velocity_max = list(default.joint_velocity_max)
            joint_acceleration_max = list(default.joint_acceleration_max)
            joint_torque_max = list(default.joint_torque_max)

        if pos_min_deg is not None:
            joint_position_min = [math.radians(d) for d in pos_min_deg]
        if pos_max_deg is not None:
            joint_position_max = [math.radians(d) for d in pos_max_deg]

        home_deg = control_cfg.home_position_deg
        home_pos = [math.radians(d) for d in home_deg] if home_deg is not None else list(default.home_position)

        gripper_enabled = gripper_cfg.enabled
        gripper = None
        if gripper_enabled:
            base_gripper = default.gripper or GripperParams(max_width=0.085)
            gripper = GripperParams(
                max_width=float(gripper_cfg.max_width),
                min_width=float(gripper_cfg.min_width),
                max_velocity=float(gripper_cfg.max_velocity),
                max_force=float(gripper_cfg.max_force),
            )

        return RobotParams(
            dof=dof,
            joint_position_min=joint_position_min,
            joint_position_max=joint_position_max,
            joint_velocity_max=limits_cfg.velocity_max if limits_cfg and limits_cfg.velocity_max is not None else joint_velocity_max,
            joint_acceleration_max=limits_cfg.acceleration_max if limits_cfg and limits_cfg.acceleration_max is not None else joint_acceleration_max,
            joint_torque_max=limits_cfg.torque_max if limits_cfg and limits_cfg.torque_max is not None else joint_torque_max,
            gripper=gripper,
            home_position=home_pos,
            control_frequency_hz=float(control_cfg.frequency_hz),
        )

    # ── 生命周期 ──────────────────────────────────────────────

    def connect(self) -> None:
        if self._connected:
            return

        if self._rtde_c is None or self._rtde_r is None:
            if self._rtde_control_module is None:
                import rtde_control as _rtde_control  # type: ignore
                self._rtde_control_module = _rtde_control
            if self._rtde_receive_module is None:
                import rtde_receive as _rtde_receive  # type: ignore
                self._rtde_receive_module = _rtde_receive

            self._rtde_r = self._rtde_receive_module.RTDEReceiveInterface(self.robot_ip)
            self._rtde_c = self._rtde_control_module.RTDEControlInterface(self.robot_ip)

        self._connected = True

        # 以硬件返回自由度为准，保证校验长度一致
        q = _to_float_list(_safe_call(self._rtde_r, "getActualQ", default=[]))
        if q:
            hw_dof = len(q)
            if hw_dof != self.dof:
                logger.warning("配置 DOF=%d 与硬件 DOF=%d 不一致，采用硬件值", self.dof, hw_dof)
                self.dof = hw_dof
                self._params = _adapt_params_dof(self._params, hw_dof)

        logger.info("已连接 UR: %s (DOF=%d)", self.name, self.dof)

    def disconnect(self) -> None:
        if not self._connected:
            return

        try:
            self.stop()
        except Exception:
            logger.warning("断开前 stop 失败", exc_info=True)

        try:
            _safe_call(self._rtde_c, "disconnect", default=None)
        except Exception:
            pass
        try:
            _safe_call(self._rtde_r, "disconnect", default=None)
        except Exception:
            pass

        self._rtde_c = None
        self._rtde_r = None
        self._connected = False
        self._last_cmd_q = None
        self._last_vel_send_time = None
        logger.info("已断开 UR: %s", self.name)

    def enable(self) -> None:
        self._require_connected()
        # ur_rtde 无统一 "enable"，这里保证控制通道在线。
        if not bool(_safe_call(self._rtde_c, "isConnected", default=True)):
            _safe_call(self._rtde_c, "reconnect", default=None)
            _safe_call(self._rtde_c, "reuploadScript", default=None)
            time.sleep(0.1)

    def stop(self) -> None:
        self._require_connected()
        _safe_call(self._rtde_c, "servoStop", default=None)
        _safe_call(self._rtde_c, "speedStop", default=None)
        _safe_call(self._rtde_c, "stopScript", default=None)
        self._last_cmd_q = None
        self._last_vel_send_time = None

    def emergency_stop(self) -> None:
        self._require_connected()
        self.stop()
        logger.warning("UR 紧急停止已执行: %s", self.name)

    def clear_fault(self) -> None:
        self._require_connected()
        # UR 清故障通常依赖 dashboard server，这里保留 no-op。
        logger.warning("UR clear_fault() 需要 Dashboard 接口，当前为 no-op")

    def get_params(self) -> RobotParams:
        return self._params

    def is_connected(self) -> bool:
        if not self._connected or self._rtde_c is None:
            return False
        return bool(_safe_call(self._rtde_c, "isConnected", default=True))

    def is_operational(self) -> bool:
        self._require_connected()
        if self.is_fault():
            return False
        return self.is_connected()

    def is_busy(self) -> bool:
        self._require_connected()
        # ur_rtde: isSteady=True 时机器人已稳定到位
        steady = _safe_call(self._rtde_c, "isSteady", default=None)
        if steady is None:
            return False
        return not bool(steady)

    def is_fault(self) -> bool:
        self._require_connected()
        protective = bool(_safe_call(self._rtde_r, "isProtectiveStopped", default=False))
        emergency = bool(_safe_call(self._rtde_r, "isEmergencyStopped", default=False))
        return protective or emergency

    # ── 核心原语 ──────────────────────────────────────────────

    def observe(self) -> ArmState:
        self._require_connected()

        q = _to_float_list(_safe_call(self._rtde_r, "getActualQ", default=[]))
        dq = _to_float_list(_safe_call(self._rtde_r, "getActualQd", default=[]))
        tau = _to_float_list(_safe_call(self._rtde_r, "getActualCurrentAsTorque", default=[]))
        q_target = _to_float_list(_safe_call(self._rtde_r, "getTargetQ", default=[]))

        tcp_pose_rv = _to_float_list(_safe_call(self._rtde_r, "getActualTCPPose", default=[]))
        tcp_speed = _to_float_list(_safe_call(self._rtde_r, "getActualTCPSpeed", default=[]))
        tcp_force = _to_float_list(_safe_call(self._rtde_r, "getActualTCPForce", default=[]))

        eef_pose = []
        if len(tcp_pose_rv) >= 6:
            x, y, z, rx, ry, rz = tcp_pose_rv[:6]
            qw, qx, qy, qz = _rotvec_to_quat(rx, ry, rz)
            eef_pose = [x, y, z, qw, qx, qy, qz]

        return ArmState(
            timestamp=time.perf_counter(),
            joint_positions=q,
            joint_velocities=dq,
            joint_torques=tau,
            joint_external_torques=[],
            joint_positions_desired=q_target,
            eef_pose=eef_pose,
            eef_velocity=tcp_speed,
            wrench_in_tcp=tcp_force,
            wrench_in_world=[],
        )

    def act(self, action: Action, *, state: ArmState | None = None) -> None:
        self._require_connected()

        if action.space == ActionSpace.JOINT_POSITION:
            self._act_joint_position(action, state)
            return

        if action.space == ActionSpace.JOINT_VELOCITY:
            self._act_joint_velocity(action, state)
            return

        if action.space == ActionSpace.JOINT_TORQUE:
            self._act_joint_torque(action)
            return

        if action.space == ActionSpace.CARTESIAN:
            self._act_cartesian(action)
            return

        raise ValueError(f"不支持的动作空间: {action.space}")

    def _act_joint_position(self, action: Action, state: ArmState | None) -> None:
        self._validate_length("action.values", action.values)
        self._validate_joint_positions(action.values)
        target = [float(v) for v in action.values]
        self._send_servoj(target, state=state)

    def servo_joint_positions(
        self,
        target_q: Sequence[float],
        *,
        state: ArmState | None = None,
    ) -> None:
        """Stream one absolute joint-position target via servoJ without extra step limiting.

        Client-side code can perform its own joint-step limiting before calling this
        method. This keeps the behavior aligned with ur_act_client.py, where the
        client computes cmd_q and the robot wrapper sends it directly.
        """
        self._require_connected()
        target = [float(v) for v in target_q]
        self._validate_length("target_q", target)
        self._validate_joint_positions(target)
        self._send_servoj_direct(target, state=state)

    def _act_joint_velocity(self, action: Action, state: ArmState | None) -> None:
        self._validate_length("action.values", action.values)
        self._validate_joint_velocities(action.values)

        now = time.perf_counter()
        if self._last_vel_send_time is not None:
            dt = now - self._last_vel_send_time
        else:
            dt = 1.0 / max(self._params.control_frequency_hz, 1.0)
        self._last_vel_send_time = now

        if state is not None and len(state.joint_positions) == self.dof:
            q_now = [float(v) for v in state.joint_positions]
        else:
            q_now = _to_float_list(_safe_call(self._rtde_r, "getActualQ", default=[]))
            self._validate_length("current_q", q_now)

        target = [q_now[i] + float(action.values[i]) * dt for i in range(self.dof)]
        self._send_servoj(target, state=state)

    def _act_joint_torque(self, action: Action) -> None:
        self._validate_length("action.values", action.values)
        self._validate_joint_torques(action.values)

        torques = [float(v) for v in action.values]
        # 不同 ur_rtde 版本方法名不同，按可用能力调用
        if _has_callable(self._rtde_c, "directTorque"):
            self._rtde_c.directTorque(torques)
            return
        if _has_callable(self._rtde_c, "setJointTorques"):
            self._rtde_c.setJointTorques(torques)
            return
        raise NotImplementedError("当前 ur_rtde 版本不支持关节力矩控制")

    def _act_cartesian(self, action: Action) -> None:
        self._validate_length("action.values", action.values, expected=7)
        x, y, z, qw, qx, qy, qz = [float(v) for v in action.values]
        rx, ry, rz = _quat_to_rotvec(qw, qx, qy, qz)
        pose_rv = [x, y, z, rx, ry, rz]

        dt = 1.0 / max(self._params.control_frequency_hz, 1.0)
        lookahead = float(self._control_config.get("servoj_lookahead", 0.1))
        gain = float(self._control_config.get("servoj_gain", 300.0))

        if _has_callable(self._rtde_c, "servoL"):
            self._rtde_c.servoL(pose_rv, 0.0, 0.0, dt, lookahead, gain)
            return

        # fallback: moveL 非实时，但保持接口可用
        if _has_callable(self._rtde_c, "moveL"):
            speed = float(action.extra.get("speed", 0.25))
            acc = float(action.extra.get("acceleration", 1.2))
            self._rtde_c.moveL(pose_rv, speed, acc, True)
            return

        raise NotImplementedError("当前 ur_rtde 版本不支持笛卡尔控制")

    def go_home(self, *, velocity: float | None = None, timeout_s: float = 60.0) -> bool:
        """UR 专用回 Home: moveJ + reconnect/reupload + 重试。"""
        del timeout_s  # moveJ 自身阻塞，不再单独使用 timeout

        self._require_connected()
        home = self.get_params().home_position
        self._validate_length("home_position", home)

        speed = float(self._control_config.get("home_speed", 0.5))
        acceleration = float(self._control_config.get("home_acceleration", 0.5))
        if velocity is not None:
            speed = float(velocity)

        moved_home = False
        for attempt in (1, 2):
            try:
                if not bool(_safe_call(self._rtde_c, "isConnected", default=True)):
                    _safe_call(self._rtde_c, "reconnect", default=None)
                    time.sleep(0.1)

                _safe_call(self._rtde_c, "reuploadScript", default=None)
                time.sleep(0.2)

                if _has_callable(self._rtde_c, "moveJ"):
                    self._rtde_c.moveJ(list(home), speed=speed, acceleration=acceleration)
                    moved_home = True
                    break

                # 兜底: 没有 moveJ 时退回基类实现
                return super().go_home(velocity=velocity, timeout_s=60.0)

            except Exception as e:
                logger.warning("Home attempt %d failed: %s", attempt, e)
                time.sleep(0.2)

        if not moved_home:
            logger.warning("Return home failed after retries.")

        return moved_home

    def _send_servoj(self, target_q: list[float], *, state: ArmState | None = None) -> None:
        self._validate_length("target_q", target_q)

        # Align with ur_act_client behavior: ensure control channel is ready
        # before sending each servoJ command.
        if not bool(_safe_call(self._rtde_c, "isConnected", default=True)):
            _safe_call(self._rtde_c, "reconnect", default=None)
            _safe_call(self._rtde_c, "reuploadScript", default=None)
            time.sleep(0.1)

        if self._last_cmd_q is None:
            if state is not None and len(state.joint_positions) == self.dof:
                self._last_cmd_q = [float(v) for v in state.joint_positions]
            else:
                self._last_cmd_q = _to_float_list(_safe_call(self._rtde_r, "getActualQ", default=[]))

        self._validate_length("_last_cmd_q", self._last_cmd_q)

        step_limit = float(self._control_config.get("joint_step_limit", 0.05))
        if step_limit > 0:
            cmd_q: list[float] = []
            for prev, target in zip(self._last_cmd_q, target_q):
                dq = max(-step_limit, min(step_limit, target - prev))
                cmd_q.append(prev + dq)
        else:
            cmd_q = list(target_q)

        dt = 1.0 / max(self._params.control_frequency_hz, 1.0)
        lookahead = float(self._control_config.get("servoj_lookahead", 0.1))
        gain = float(self._control_config.get("servoj_gain", 300.0))

        self._rtde_c.servoJ(cmd_q, 0.0, 0.0, dt, lookahead, gain)
        self._last_cmd_q = cmd_q

        # Optional blocking wait: ensure each servoJ step is largely reached
        # before returning to upper loop, so a chunk is consumed more strictly
        # in sequence.
        if bool(self._control_config.get("wait_step_reached", True)):
            self._wait_until_joint_near_target(cmd_q)

    def _send_servoj_direct(self, target_q: list[float], *, state: ArmState | None = None) -> None:
        self._validate_length("target_q", target_q)

        if not bool(_safe_call(self._rtde_c, "isConnected", default=True)):
            _safe_call(self._rtde_c, "reconnect", default=None)
            _safe_call(self._rtde_c, "reuploadScript", default=None)
            time.sleep(0.1)

        if self._last_cmd_q is None:
            if state is not None and len(state.joint_positions) == self.dof:
                self._last_cmd_q = [float(v) for v in state.joint_positions]
            else:
                self._last_cmd_q = _to_float_list(_safe_call(self._rtde_r, "getActualQ", default=[]))
        self._validate_length("_last_cmd_q", self._last_cmd_q)

        dt = 1.0 / max(self._params.control_frequency_hz, 1.0)
        lookahead = float(self._control_config.get("servoj_lookahead", 0.1))
        gain = float(self._control_config.get("servoj_gain", 300.0))

        cmd_q = list(target_q)
        self._rtde_c.servoJ(cmd_q, 0.0, 0.0, dt, lookahead, gain)
        self._last_cmd_q = cmd_q

        if bool(self._control_config.get("wait_step_reached", True)):
            self._wait_until_joint_near_target(cmd_q)

    def _wait_until_joint_near_target(self, target_q: list[float]) -> None:
        tol = float(self._control_config.get("step_reached_tolerance", 0.01))
        poll_s = float(self._control_config.get("step_reached_poll_s", 0.005))

        # Default behavior: strictly wait until target is reached.
        # Optional timeout can be enabled by setting step_reached_timeout_s > 0.
        timeout_cfg = self._control_config.get("step_reached_timeout_s", None)
        timeout_s: float | None
        if timeout_cfg is None:
            timeout_s = None
        else:
            timeout_s = float(timeout_cfg)
            if timeout_s <= 0:
                timeout_s = None

        if tol <= 0:
            return

        deadline = (time.perf_counter() + timeout_s) if timeout_s is not None else None
        while True:
            q_now = _to_float_list(_safe_call(self._rtde_r, "getActualQ", default=[]))
            if len(q_now) == self.dof:
                max_err = max(abs(a - b) for a, b in zip(q_now, target_q))
                if max_err <= tol:
                    return

            if deadline is not None and time.perf_counter() >= deadline:
                return

            time.sleep(poll_s)


    # ── 末端工具工厂 ──────────────────────────────────────────

    def create_gripper(self, config: dict[str, Any] | None = None) -> URRobotiqGripper | None:
        self._require_connected()
        cfg = config if config is not None else self._gripper_config
        if not cfg.get("enabled", True):
            return None
        return URRobotiqGripper(
            self.robot_ip,
            name=cfg.get("name", "robotiq"),
            config=cfg,
        )

    # ── 内部 ──────────────────────────────────────────────────

    def _require_connected(self) -> None:
        if not self._connected or self._rtde_c is None or self._rtde_r is None:
            raise RuntimeError(
                f"机器人 '{self.name}' 尚未连接，请先调用 connect() 或使用 with 语句"
            )


# ── 模块级工具函数 ────────────────────────────────────────────


def _has_callable(obj: Any, name: str) -> bool:
    return callable(getattr(obj, name, None))


def _safe_call(obj: Any, name: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    fn = getattr(obj, name, None)
    if not callable(fn):
        return default
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def _to_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        return [float(v) for v in list(value)]
    except Exception:
        return []


def _adapt_params_dof(params: RobotParams, dof: int) -> RobotParams:
    def _resize(seq: list[float], fill: float) -> list[float]:
        out = list(seq[:dof])
        while len(out) < dof:
            out.append(fill)
        return out

    return RobotParams(
        dof=dof,
        joint_position_min=_resize(params.joint_position_min, -2.0 * math.pi),
        joint_position_max=_resize(params.joint_position_max, 2.0 * math.pi),
        joint_velocity_max=_resize(params.joint_velocity_max, 3.14),
        joint_acceleration_max=_resize(params.joint_acceleration_max, 6.0),
        joint_torque_max=_resize(params.joint_torque_max, 150.0),
        gripper=params.gripper,
        home_position=_resize(params.home_position, 0.0),
        control_frequency_hz=params.control_frequency_hz,
    )


def _quat_to_rotvec(qw: float, qx: float, qy: float, qz: float) -> list[float]:
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    sin_half = math.sqrt(qx * qx + qy * qy + qz * qz)
    if sin_half < 1e-10:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(sin_half, qw)
    scale = angle / sin_half
    return [qx * scale, qy * scale, qz * scale]


def _rotvec_to_quat(rx: float, ry: float, rz: float) -> list[float]:
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    half = 0.5 * angle
    s = math.sin(half) / angle
    return [math.cos(half), rx * s, ry * s, rz * s]
