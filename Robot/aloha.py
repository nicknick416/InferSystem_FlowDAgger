from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from Core import Action, ActionSpace, ArmState, RobotParams
from Core.config_schema import AlohaRobotConfig
from Robot.base import BaseRobot

logger = logging.getLogger(__name__)

_DEG_TO_RAD = math.pi / 180.0
_RAD_TO_DEG = 180.0 / math.pi


@dataclass(slots=True)
class _PiperArm:
    name: str
    can_interface: str
    piper: Any | None = None


@BaseRobot.register("aloha")
class AlohaRobot(BaseRobot):
    """Lightweight dual-arm Piper/Aloha driver.

    This driver intentionally bypasses LeRobot and camera handling.  It only
    owns the robot primitive path: connect, observe arm state, and send arm
    actions.  RGB and tactile images are handled by SensorManager from the
    top-level YAML sections.

    Joint state/action layout is 14 values in SI-ish units:
        [left_j1..j6 rad, left_gripper m,
         right_j1..j6 rad, right_gripper m]

    Cartesian action layout is the canonical bimanual pose format:
        [left_x,y,z,qw,qx,qy,qz,left_gripper,
         right_x,y,z,qw,qx,qy,qz,right_gripper]
    """

    def __init__(
        self,
        *,
        can_interface_left: str = "can0",
        can_interface_right: str = "can1",
        speed_percent: int = 20,
        enable_timeout: float = 5.0,
        name: str = "aloha",
        control_frequency_hz: float = 30.0,
        gripper_max_m: float = 0.07,
        gripper_min_left_m: float | None = None,
        gripper_max_left_m: float | None = None,
        gripper_min_right_m: float | None = None,
        gripper_max_right_m: float | None = None,
        home_gripper_left_m: float | None = None,
        home_gripper_right_m: float | None = None,
        gripper_sdk_scale: float = 10_000_000.0,
        gripper_sdk_max: int | None = None,
        gripper_write_scale_left: float = 1.0,
        gripper_write_scale_right: float = 1.0,
        gripper_sdk_max_left: int | None = None,
        gripper_sdk_max_right: int | None = None,
        gripper_effort_left: int = 1000,
        gripper_effort_right: int = 1000,
        gripper_read_scale_left: float = 1.0,
        gripper_read_scale_right: float = 1.0,
        home_gripper_m: float | None = None,
    ) -> None:
        super().__init__(name=name, robot_type="aloha", dof=14)
        self._left = _PiperArm("left", can_interface_left)
        self._right = _PiperArm("right", can_interface_right)
        self._speed_percent = max(1, min(100, int(speed_percent)))
        self._enable_timeout = float(enable_timeout)
        self._gripper_sdk_scale = float(gripper_sdk_scale) if gripper_sdk_scale else 10_000_000.0
        self._gripper_sdk_max = gripper_sdk_max
        self._gripper_write_scale = {
            "left": float(gripper_write_scale_left or 1.0),
            "right": float(gripper_write_scale_right or 1.0),
        }
        self._gripper_sdk_max_by_arm = {
            "left": gripper_sdk_max if gripper_sdk_max_left is None else gripper_sdk_max_left,
            "right": gripper_sdk_max if gripper_sdk_max_right is None else gripper_sdk_max_right,
        }
        self._gripper_effort = {
            "left": max(0, min(5000, int(gripper_effort_left))),
            "right": max(0, min(5000, int(gripper_effort_right))),
        }
        left_min = 0.0 if gripper_min_left_m is None else float(gripper_min_left_m)
        left_max = float(gripper_max_m if gripper_max_left_m is None else gripper_max_left_m)
        right_min = 0.0 if gripper_min_right_m is None else float(gripper_min_right_m)
        right_max = float(gripper_max_m if gripper_max_right_m is None else gripper_max_right_m)
        self._gripper_limits = {
            "left": (min(left_min, left_max), max(left_min, left_max)),
            "right": (min(right_min, right_max), max(right_min, right_max)),
        }
        self._gripper_read_scale = {
            "left": float(gripper_read_scale_left or 1.0),
            "right": float(gripper_read_scale_right or 1.0),
        }
        default_home = gripper_max_m if home_gripper_m is None else float(home_gripper_m)
        self._home_gripper = {
            "left": default_home if home_gripper_left_m is None else float(home_gripper_left_m),
            "right": default_home if home_gripper_right_m is None else float(home_gripper_right_m),
        }
        self._last_logged_gripper_sdk: dict[str, int] = {}
        self._connected = False
        self._fault = False
        self._params = RobotParams(
            dof=14,
            joint_position_min=[
                -165.0 * _DEG_TO_RAD, -105.0 * _DEG_TO_RAD,
                -165.0 * _DEG_TO_RAD, -155.0 * _DEG_TO_RAD,
                -165.0 * _DEG_TO_RAD, -165.0 * _DEG_TO_RAD, self._gripper_limits["left"][0],
                -165.0 * _DEG_TO_RAD, -105.0 * _DEG_TO_RAD,
                -165.0 * _DEG_TO_RAD, -155.0 * _DEG_TO_RAD,
                -165.0 * _DEG_TO_RAD, -165.0 * _DEG_TO_RAD, self._gripper_limits["right"][0],
            ],
            joint_position_max=[
                165.0 * _DEG_TO_RAD, 105.0 * _DEG_TO_RAD,
                165.0 * _DEG_TO_RAD, 0.0,
                165.0 * _DEG_TO_RAD, 165.0 * _DEG_TO_RAD, self._gripper_limits["left"][1],
                165.0 * _DEG_TO_RAD, 105.0 * _DEG_TO_RAD,
                165.0 * _DEG_TO_RAD, 0.0,
                165.0 * _DEG_TO_RAD, 165.0 * _DEG_TO_RAD, self._gripper_limits["right"][1],
            ],
            joint_velocity_max=[math.radians(360.0)] * 14,
            joint_acceleration_max=[math.radians(720.0)] * 14,
            joint_torque_max=[100.0] * 14,
            home_position=[0.0] * 14,
            control_frequency_hz=control_frequency_hz,
        )

    @classmethod
    def _from_config_dict(cls, robot_cfg: AlohaRobotConfig | dict[str, Any]) -> AlohaRobot:
        if isinstance(robot_cfg, dict):
            robot_cfg = AlohaRobotConfig.model_validate(robot_cfg)
        control_cfg = robot_cfg.control
        return cls(
            can_interface_left=robot_cfg.can_interface_left,
            can_interface_right=robot_cfg.can_interface_right,
            speed_percent=robot_cfg.speed_percent,
            enable_timeout=robot_cfg.enable_timeout,
            name=robot_cfg.name or "aloha",
            control_frequency_hz=control_cfg.frequency_hz,
            gripper_max_m=robot_cfg.gripper_max_m,
            gripper_min_left_m=robot_cfg.gripper_min_left_m,
            gripper_max_left_m=robot_cfg.gripper_max_left_m,
            gripper_min_right_m=robot_cfg.gripper_min_right_m,
            gripper_max_right_m=robot_cfg.gripper_max_right_m,
            home_gripper_left_m=robot_cfg.home_gripper_left_m,
            home_gripper_right_m=robot_cfg.home_gripper_right_m,
            gripper_sdk_scale=robot_cfg.gripper_sdk_scale,
            gripper_sdk_max=robot_cfg.gripper_sdk_max,
            gripper_write_scale_left=robot_cfg.gripper_write_scale_left,
            gripper_write_scale_right=robot_cfg.gripper_write_scale_right,
            gripper_sdk_max_left=robot_cfg.gripper_sdk_max_left,
            gripper_sdk_max_right=robot_cfg.gripper_sdk_max_right,
            gripper_effort_left=robot_cfg.gripper_effort_left,
            gripper_effort_right=robot_cfg.gripper_effort_right,
            gripper_read_scale_left=robot_cfg.gripper_read_scale_left,
            gripper_read_scale_right=robot_cfg.gripper_read_scale_right,
            home_gripper_m=robot_cfg.home_gripper_m,
        )

    def connect(self) -> None:
        if self._connected:
            return
        try:
            from piper_sdk import C_PiperInterface_V2 as PiperInterface
        except Exception as exc:
            try:
                from piper_sdk import C_PiperInterface as PiperInterface
            except Exception:
                raise ImportError(
                    "piper_sdk==0.6.1 is required for aloha. Install it in the "
                    "infersystem environment before connecting hardware."
                ) from exc

        try:
            for arm in (self._left, self._right):
                arm.piper = PiperInterface(arm.can_interface)
                arm.piper.ConnectPort()
                self._enable_arm(arm)
                self._clear_gripper_fault(arm)
                self._set_move_mode(arm, 0x01)
                logger.info(
                    "Aloha %s arm connected on %s, speed=%d%%, gripper_effort=%d",
                    arm.name,
                    arm.can_interface,
                    self._speed_percent,
                    self._gripper_effort.get(arm.name, 1000),
                )
            self._connected = True
            self._fault = False
        except Exception:
            self._fault = True
            self.disconnect()
            raise

    def disconnect(self) -> None:
        for arm in (self._left, self._right):
            piper = arm.piper
            if piper is None:
                continue
            try:
                disconnect = getattr(piper, "DisconnectPort", None)
                if callable(disconnect):
                    disconnect()
            except Exception:
                logger.warning("Error disconnecting %s arm", arm.name, exc_info=True)
            finally:
                arm.piper = None
        self._connected = False

    def enable(self) -> None:
        for arm in (self._left, self._right):
            if arm.piper is not None:
                self._enable_arm(arm)
        self._fault = False

    def stop(self) -> None:
        for arm in (self._left, self._right):
            if arm.piper is None:
                continue
            try:
                arm.piper.MotionCtrl_2(0x01, 0x01, 1, 0x00)
            except Exception:
                logger.warning("Stop failed for %s arm", arm.name, exc_info=True)

    def emergency_stop(self) -> None:
        self._fault = True
        for arm in (self._left, self._right):
            if arm.piper is None:
                continue
            try:
                arm.piper.EmergencyStop(0x01)
            except Exception:
                logger.warning("Emergency stop failed for %s arm", arm.name, exc_info=True)

    def clear_fault(self) -> None:
        for arm in (self._left, self._right):
            if arm.piper is None:
                continue
            try:
                arm.piper.EmergencyStop(0x02)
            except Exception:
                logger.debug("Fault clear failed for %s arm", arm.name, exc_info=True)
        self._fault = False

    def get_params(self) -> RobotParams:
        return self._params

    def is_connected(self) -> bool:
        return self._connected

    def is_operational(self) -> bool:
        return self._connected and not self._fault

    def is_busy(self) -> bool:
        return False

    def is_fault(self) -> bool:
        return self._fault

    def observe(self) -> ArmState:
        self._require_connected()
        positions = self._read_arm_state(self._left) + self._read_arm_state(self._right)
        eef_pose = self._read_eef_pose(self._left) + self._read_eef_pose(self._right)
        return ArmState(
            timestamp=time.perf_counter(),
            joint_positions=positions,
            eef_pose=eef_pose,
        )

    def act(self, action: Action, *, state: ArmState | None = None) -> None:
        self._require_connected()
        if action.space is ActionSpace.JOINT_POSITION:
            if len(action.values) != 14:
                raise ValueError(f"aloha joint action must be 14 values, got {len(action.values)}")
            values = self._clamp_positions(action.values)
            self._send_arm_action(self._left, values[:7])
            self._send_arm_action(self._right, values[7:14])
        elif action.space is ActionSpace.CARTESIAN:
            self._act_cartesian(action.values, state=state)
        else:
            raise NotImplementedError(f"aloha does not support {action.space.value} actions")

    def go_home(self, *, velocity: float | None = None, timeout_s: float = 30.0) -> bool:
        left_home = self._clamp_gripper_value(self._home_gripper["left"], "left")
        right_home = self._clamp_gripper_value(self._home_gripper["right"], "right")
        home_values = [0.0] * 6 + [left_home] + [0.0] * 6 + [right_home]
        home = Action(ActionSpace.JOINT_POSITION, home_values)
        dt = 1.0 / self._params.control_frequency_hz
        deadline = time.monotonic() + timeout_s
        commanded_once = False
        old_speed = self._speed_percent
        if velocity is not None:
            self._speed_percent = max(1, min(100, int(round(float(velocity)))))
        try:
            while time.monotonic() < deadline:
                state = self.observe()
                arm_positions = state.joint_positions[:6] + state.joint_positions[7:13]
                at_home = all(abs(v) < math.radians(2.0) for v in arm_positions)
                if at_home and commanded_once:
                    return True
                self.act(home)
                commanded_once = True
                time.sleep(dt)
            return False
        finally:
            self._speed_percent = old_speed

    def _enable_arm(self, arm: _PiperArm) -> None:
        assert arm.piper is not None
        deadline = time.monotonic() + self._enable_timeout
        while True:
            try:
                if arm.piper.EnablePiper():
                    return
            except Exception:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"EnablePiper timed out for {arm.name} arm on {arm.can_interface}"
                )
            time.sleep(0.01)

    def _require_connected(self) -> None:
        if not self._connected or self._left.piper is None or self._right.piper is None:
            raise RuntimeError("AlohaRobot not connected")

    def _clear_gripper_fault(self, arm: _PiperArm) -> None:
        assert arm.piper is not None
        try:
            arm.piper.GripperCtrl(0, self._gripper_effort.get(arm.name, 1000), 0x02, 0)
            time.sleep(0.01)
            logger.info("Aloha %s gripper fault clear sent", arm.name)
        except Exception:
            logger.warning("Aloha %s gripper fault clear failed", arm.name, exc_info=True)

    def _read_arm_state(self, arm: _PiperArm) -> list[float]:
        assert arm.piper is not None
        joint_state = arm.piper.GetArmJointMsgs().joint_state
        joints_rad = [
            float(joint_state.joint_1) / 1000.0 * _DEG_TO_RAD,
            float(joint_state.joint_2) / 1000.0 * _DEG_TO_RAD,
            float(joint_state.joint_3) / 1000.0 * _DEG_TO_RAD,
            float(joint_state.joint_4) / 1000.0 * _DEG_TO_RAD,
            float(joint_state.joint_5) / 1000.0 * _DEG_TO_RAD,
            float(joint_state.joint_6) / 1000.0 * _DEG_TO_RAD,
        ]
        gripper_m = 0.0
        try:
            gripper = arm.piper.GetArmGripperMsgs()
            raw_gripper_m = float(gripper.gripper_state.grippers_angle) / self._gripper_sdk_scale
            gripper_m = raw_gripper_m * self._gripper_read_scale.get(arm.name, 1.0)
            gripper_m = self._clamp_gripper_value(gripper_m, arm.name)
        except Exception:
            logger.debug("Unable to read %s gripper state", arm.name, exc_info=True)
        return joints_rad + [gripper_m]

    def _send_arm_action(self, arm: _PiperArm, values: list[float]) -> None:
        assert arm.piper is not None
        self._set_move_mode(arm, 0x01)
        joints_sdk = [int(round(v * _RAD_TO_DEG * 1000.0)) for v in values[:6]]
        gripper_m = self._clamp_gripper_value(values[6], arm.name)
        gripper_sdk = self._gripper_to_sdk(gripper_m, arm.name)
        arm.piper.JointCtrl(*joints_sdk)
        self._log_gripper_command(arm, gripper_m, gripper_sdk)
        arm.piper.GripperCtrl(gripper_sdk, self._gripper_effort.get(arm.name, 1000), 0x01, 0)

    def _act_cartesian(self, values: Sequence[float], *, state: ArmState | None = None) -> None:
        n = len(values)
        if n not in (14, 16):
            raise ValueError(f"aloha cartesian action must be 14 or 16 values, got {n}")
        if n == 16:
            left_pose = values[:7]
            right_pose = values[8:15]
            left_gripper = self._policy_gripper_to_encoder(float(values[7]), "left")
            right_gripper = self._policy_gripper_to_encoder(float(values[15]), "right")
        else:
            pose_state = state if state is not None else self.observe()
            if len(pose_state.joint_positions) < 14:
                raise ValueError("aloha cartesian 14D action requires current gripper state")
            left_pose, left_gripper = values[:7], float(pose_state.joint_positions[6])
            right_pose, right_gripper = values[7:14], float(pose_state.joint_positions[13])
        self._send_arm_cartesian(self._left, left_pose, left_gripper)
        self._send_arm_cartesian(self._right, right_pose, right_gripper)

    def _policy_gripper_to_encoder(self, value: float, arm_name: str) -> float:
        return self._clamp_gripper_value(float(value), arm_name)

    def _send_arm_cartesian(self, arm: _PiperArm, pose7: Sequence[float], gripper_m: float) -> None:
        assert arm.piper is not None
        if len(pose7) != 7:
            raise ValueError(f"aloha cartesian pose must be pose7, got {len(pose7)} values")
        self._set_endpose_mode(arm)
        roll, pitch, yaw = _quat_to_euler_xyz(pose7[3:7])
        arm.piper.EndPoseCtrl(
            int(round(float(pose7[0]) * 1_000_000.0)),
            int(round(float(pose7[1]) * 1_000_000.0)),
            int(round(float(pose7[2]) * 1_000_000.0)),
            int(round(roll * _RAD_TO_DEG * 1000.0)),
            int(round(pitch * _RAD_TO_DEG * 1000.0)),
            int(round(yaw * _RAD_TO_DEG * 1000.0)),
        )
        gripper_m = self._clamp_gripper_value(gripper_m, arm.name)
        gripper_sdk = self._gripper_to_sdk(gripper_m, arm.name)
        self._log_gripper_command(arm, gripper_m, gripper_sdk)
        arm.piper.GripperCtrl(gripper_sdk, self._gripper_effort.get(arm.name, 1000), 0x01, 0)

    def _log_gripper_command(self, arm: _PiperArm, gripper_m: float, gripper_sdk: int) -> None:
        prev = self._last_logged_gripper_sdk.get(arm.name)
        if prev is None or abs(int(gripper_sdk) - int(prev)) >= 1000:
            logger.info(
                "Aloha %s gripper command: %.5fm -> sdk=%d",
                arm.name,
                float(gripper_m),
                int(gripper_sdk),
            )
            self._last_logged_gripper_sdk[arm.name] = int(gripper_sdk)

    def _gripper_to_sdk(self, gripper_m: float, arm_name: str) -> int:
        gripper_m = self._clamp_gripper_value(gripper_m, arm_name)
        scale = self._gripper_sdk_scale * self._gripper_write_scale.get(arm_name, 1.0)
        sdk = int(round(float(gripper_m) * scale))
        sdk_max = self._gripper_sdk_max_by_arm.get(arm_name, self._gripper_sdk_max)
        if sdk_max is not None and sdk >= 0:
            sdk = min(int(sdk_max), sdk)
        return sdk

    def _clamp_gripper_value(self, gripper_m: float, arm_name: str) -> float:
        lo, hi = self._gripper_limits.get(arm_name, (0.0, self._params.joint_position_max[6]))
        return max(lo, min(hi, float(gripper_m)))

    def _set_endpose_mode(self, arm: _PiperArm) -> None:
        assert arm.piper is not None
        motion_ctrl_1 = getattr(arm.piper, "MotionCtrl_1", None)
        if callable(motion_ctrl_1):
            motion_ctrl_1(0x00, 0x00, 0x00)
        arm.piper.MotionCtrl_2(0x01, 0x00, self._speed_percent, 0x00)

    def _set_move_mode(self, arm: _PiperArm, move_mode: int) -> None:
        assert arm.piper is not None
        arm.piper.MotionCtrl_2(0x01, move_mode, self._speed_percent, 0x00)

    def _clamp_positions(self, values: list[float]) -> list[float]:
        mins = self._params.joint_position_min
        maxs = self._params.joint_position_max
        clamped: list[float] = []
        for i, value in enumerate(values):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"aloha action value {i} is not finite: {value!r}")
            clamped.append(max(mins[i], min(maxs[i], numeric)))
        return clamped

    def _read_eef_pose(self, arm: _PiperArm) -> list[float]:
        assert arm.piper is not None
        try:
            msg = arm.piper.GetArmEndPoseMsgs()
            pose = getattr(msg, "end_pose", msg)
            x = _get_pose_axis(pose, "X_axis") / 1_000_000.0
            y = _get_pose_axis(pose, "Y_axis") / 1_000_000.0
            z = _get_pose_axis(pose, "Z_axis") / 1_000_000.0
            roll = math.radians(_get_pose_axis(pose, "RX_axis") / 1000.0)
            pitch = math.radians(_get_pose_axis(pose, "RY_axis") / 1000.0)
            yaw = math.radians(_get_pose_axis(pose, "RZ_axis") / 1000.0)
            return [x, y, z] + _euler_xyz_to_quat(roll, pitch, yaw)
        except Exception:
            logger.debug("Unable to read %s eef pose", arm.name, exc_info=True)
            return []


def _get_pose_axis(pose: Any, name: str) -> float:
    return float(getattr(pose, name))


def _euler_xyz_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    return [qw, qx, qy, qz]


def _quat_to_euler_xyz(quat: Sequence[float]) -> tuple[float, float, float]:
    qw, qx, qy, qz = [float(v) for v in quat]
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        return 0.0, 0.0, 0.0
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw
