from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from Core import Action, ActionSpace, ArmState, RobotParams
from Core.config_schema import Arx5BimanualRobotConfig
from Robot.arx5 import (
    GRIPPER_CLOSED_SLACK_DEFAULT_M,
    _pose6d_to_pose7,
    _pose7_to_pose6d,
    apply_robot_config_overrides,
    infer_gripper_sdk_sign,
)
from Robot.base import BaseRobot

logger = logging.getLogger(__name__)

_INIT_TIMEOUT_S = 30.0
_RESET_TIMEOUT_S = 30.0
_GET_STATE_TIMEOUT_S = 0.5


@BaseRobot.register("arx5_bimanual")
class Arx5BimanualRobot(BaseRobot):
    """ARX5 双臂驱动，统一暴露 14 维 joint_position:
    [left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]
    """

    def __init__(
        self,
        *,
        left_model: str,
        left_interface: str,
        right_model: str,
        right_interface: str,
        left_urdf_path: str | Path | None = None,
        right_urdf_path: str | Path | None = None,
        name: str | None = None,
        use_background_send_recv: bool = True,
        log_level: str = "INFO",
        ctrl_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name or "arx5_bimanual", robot_type="arx5_bimanual", dof=14)
        self._left_model = left_model
        self._left_interface = left_interface
        self._right_model = right_model
        self._right_interface = right_interface
        self._left_urdf_path = _resolve_configured_urdf(left_urdf_path)
        self._right_urdf_path = _resolve_configured_urdf(right_urdf_path)
        self._use_background_send_recv = use_background_send_recv
        self._log_level_name = log_level.upper()
        self._ctrl_cfg = ctrl_cfg or {}

        self._joint_kp_scale = (
            float(self._ctrl_cfg.get("left_joint_kp_scale", 1.0)),
            float(self._ctrl_cfg.get("right_joint_kp_scale", 1.0)),
        )
        self._joint_kd_scale = (
            float(self._ctrl_cfg.get("left_joint_kd_scale", 1.0)),
            float(self._ctrl_cfg.get("right_joint_kd_scale", 1.0)),
        )

        self._enable_gripper = bool(self._ctrl_cfg.get("enable_gripper", True))
        self._flip_gripper_sign = bool(self._ctrl_cfg.get("flip_gripper_sign", False))
        self._gripper_command_offset_m = float(
            self._ctrl_cfg.get("gripper_command_offset_m", 0.0)
        )
        # (left_kp, right_kp) / (left_kd, right_kd)，右臂可单独覆盖；默认值与 SDK 一致
        kp = float(self._ctrl_cfg.get("gripper_kp", 5.0))
        kd = float(self._ctrl_cfg.get("gripper_kd", 0.2))
        self._gripper_kp = (kp, float(self._ctrl_cfg.get("right_gripper_kp", kp)))
        self._gripper_kd = (kd, float(self._ctrl_cfg.get("right_gripper_kd", kd)))

        self._arx5: Any | None = None
        self._ctrls: tuple[Any, Any] | None = None  # (left_ctrl, right_ctrl)
        self._solvers: tuple[Any, Any] | None = None  # (left_solver, right_solver) 用于 CARTESIAN IK
        self._last_joint_position_target: list[float] = []
        self._teach_mode = False
        self._teach_gain_snapshot: list[dict[str, Any]] | None = None
        self._connected = False
        self._gripper_sdk_sign: tuple[int, int] = (1, 1)
        # 二值夹爪: action 位置值经阈值判断后 snap 到 close/open 极限
        self._gripper_binary = bool(self._ctrl_cfg.get("gripper_binary", False))
        self._gripper_binary_threshold = float(self._ctrl_cfg.get("gripper_binary_threshold", 0.04))
        self._gripper_binary_close = float(self._ctrl_cfg.get("gripper_binary_close", -0.0))
        self._cartesian_ik_atomic = bool(self._ctrl_cfg.get("cartesian_ik_atomic", True))
        self._cartesian_ik_backoff_factors = tuple(
            float(factor)
            for factor in self._ctrl_cfg.get(
                "cartesian_ik_backoff_factors", [0.5, 0.25, 0.125]
            )
            if 0.0 < float(factor) < 1.0
        )

        # 占位参数 — connect() 后由 _sync_params_from_sdk 用 SDK 真实值覆盖
        home_deg = self._ctrl_cfg.get("home_position_deg_14") or [0.0] * 14
        self._params = RobotParams(
            dof=14,
            joint_position_min=[-math.pi] * 14,
            joint_position_max=[math.pi] * 14,
            joint_velocity_max=[5.0] * 14,
            joint_acceleration_max=[3.0] * 14,
            joint_torque_max=[30.0] * 14,
            gripper=None,
            home_position=[math.radians(v) for v in home_deg],
            control_frequency_hz=1.0 / float(self._ctrl_cfg.get("controller_dt", 0.002)),
        )

    # ── 工厂 ──────────────────────────────────────────────────

    @classmethod
    def _from_config_dict(cls, robot_cfg: Arx5BimanualRobotConfig | dict[str, Any]) -> Arx5BimanualRobot:
        """从 typed config 创建实例。也兼容旧的 dict 调用路径。"""
        if isinstance(robot_cfg, dict):
            robot_cfg = Arx5BimanualRobotConfig.model_validate(robot_cfg)

        ctrl = robot_cfg.control
        merged = ctrl.model_dump(exclude_none=True)
        if robot_cfg.joint_limits:
            merged["joint_limits"] = robot_cfg.joint_limits.model_dump(exclude_none=True)

        return cls(
            left_model=robot_cfg.left_arm.model,
            left_interface=robot_cfg.left_arm.resolved_interface,
            right_model=robot_cfg.right_arm.model,
            right_interface=robot_cfg.right_arm.resolved_interface,
            left_urdf_path=robot_cfg.left_arm.urdf_path,
            right_urdf_path=robot_cfg.right_arm.urdf_path,
            name=robot_cfg.name,
            use_background_send_recv=ctrl.background_send_recv,
            log_level=ctrl.log_level,
            ctrl_cfg=merged,
        )

    # ── 生命周期 ──────────────────────────────────────────────

    def connect(self) -> None:
        """延迟导入 arx5_interface，依次创建左右臂控制器并从 SDK 同步参数。"""
        if self._connected:
            return
        if self._left_interface == self._right_interface:
            raise ValueError("left/right arm interface must be different")

        import arx5_interface as arx5

        self._arx5 = arx5
        left_ctrl = self._init_one(
            self._left_model,
            self._left_interface,
            arm_side="left",
            urdf_path=self._left_urdf_path,
        )
        right_ctrl = self._init_one(
            self._right_model,
            self._right_interface,
            arm_side="right",
            urdf_path=self._right_urdf_path,
        )
        self._ctrls = (left_ctrl, right_ctrl)
        self._solvers = self._create_solvers()
        self._set_log_level(self._log_level_name)
        self._sync_params_from_sdk()
        self._apply_joint_gains()
        self._apply_gripper_gains()
        self._connected = True
        import atexit
        atexit.register(self.disconnect)
        logger.info(
            "已连接 ARX5 双臂: left=%s@%s right=%s@%s",
            self._left_model, self._left_interface,
            self._right_model, self._right_interface,
        )

    def disconnect(self) -> None:
        """安全断开：先尝试双臂回零（防止松弛坠落），再进入被动态。"""
        if not self._connected:
            return
        self._connected = False
        if self._ctrls is not None:
            # 先尝试回零
            for i, tag in enumerate(("left", "right")):
                try:
                    logger.info("ARX5 %s arm 断开前回零 ...", tag)
                    self._ctrls[i].reset_to_home()
                except Exception:
                    logger.warning("ARX5 %s arm 断开前回零失败", tag, exc_info=True)
            # 再进入阻尼态
            for i, tag in enumerate(("left", "right")):
                try:
                    self._ctrls[i].set_to_damping()
                except Exception:
                    logger.warning("ARX5 %s arm set_to_damping 失败", tag, exc_info=True)
        self._ctrls = None
        self._solvers = None
        self._arx5 = None
        logger.info("已断开 ARX5 双臂: %s", self.name)

    def enable(self) -> None:
        """ARX5 SDK 无显式 enable 接口，连接后即就绪。"""
        self._require_connected()

    def stop(self) -> None:
        """双臂同时进入阻尼态。"""
        self._require_connected()
        for ctrl in self._ctrls:
            ctrl.set_to_damping()

    def emergency_stop(self) -> None:
        """紧急停止：同 stop()，SDK 无更高优先级的停止方式。"""
        self.stop()
        logger.warning("ARX5 双臂紧急停止已执行 (set_to_damping)")

    def clear_fault(self) -> None:
        """SDK 无故障清除接口，尝试 go_home 恢复。"""
        self._require_connected()
        self.go_home()

    def get_params(self) -> RobotParams:
        return self._params

    def is_connected(self) -> bool:
        return self._connected and self._ctrls is not None

    def is_operational(self) -> bool:
        return self.is_connected()

    def is_busy(self) -> bool:
        return False

    def is_fault(self) -> bool:
        return False

    @property
    def in_teach_mode(self) -> bool:
        return self._teach_mode

    def enter_teach_mode(
        self,
        *,
        kp_scale: float = 0.0,
        kd_scale: float = 1.0,
        gripper_kp_scale: float = 0.0,
        gripper_kd_scale: float = 1.0,
    ) -> None:
        """Make both arms and integrated grippers hand-guidable.

        The current joint state is installed as the command before gains are
        reduced.  Gripper proportional gain is normally set to zero as well,
        allowing its width to be demonstrated directly by hand while retaining
        damping.  All gains are restored transactionally on exit.
        """
        self._require_connected()
        if self._teach_mode:
            return
        if not 0.0 <= kp_scale <= 0.2:
            raise ValueError("teach kp_scale must be in [0, 0.2]")
        if not 0.0 <= kd_scale <= 2.0:
            raise ValueError("teach kd_scale must be in [0, 2]")
        if not 0.0 <= gripper_kp_scale <= 0.2:
            raise ValueError("teach gripper_kp_scale must be in [0, 0.2]")
        if not 0.0 <= gripper_kd_scale <= 2.0:
            raise ValueError("teach gripper_kd_scale must be in [0, 2]")

        state = self.observe()
        self._act_joint_position(list(state.joint_positions))
        snapshots: list[dict[str, Any]] = []
        for ctrl in self._ctrls:
            gain = ctrl.get_gain()
            snapshots.append({
                "kp": np.asarray(gain.kp(), dtype=np.float64).copy(),
                "kd": np.asarray(gain.kd(), dtype=np.float64).copy(),
                "gripper_kp": float(gain.gripper_kp),
                "gripper_kd": float(gain.gripper_kd),
            })
        try:
            for ctrl, snapshot in zip(self._ctrls, snapshots):
                gain = ctrl.get_gain()
                gain.kp()[:] = snapshot["kp"] * kp_scale
                gain.kd()[:] = snapshot["kd"] * kd_scale
                gain.gripper_kp = snapshot["gripper_kp"] * gripper_kp_scale
                gain.gripper_kd = snapshot["gripper_kd"] * gripper_kd_scale
                ctrl.set_gain(gain)
        except Exception:
            logger.exception("进入拖动示教失败，正在回滚双臂增益")
            for ctrl, snapshot in zip(self._ctrls, snapshots):
                try:
                    gain = ctrl.get_gain()
                    gain.kp()[:] = snapshot["kp"]
                    gain.kd()[:] = snapshot["kd"]
                    gain.gripper_kp = snapshot["gripper_kp"]
                    gain.gripper_kd = snapshot["gripper_kd"]
                    ctrl.set_gain(gain)
                except Exception:
                    logger.critical("拖动示教增益回滚失败", exc_info=True)
            self._act_joint_position(list(self.observe().joint_positions))
            raise
        self._teach_gain_snapshot = snapshots
        self._teach_mode = True
        logger.warning(
            "ARX5 双臂进入拖动示教: arm kp/kd=%.3f/%.3f gripper kp/kd=%.3f/%.3f",
            kp_scale,
            kd_scale,
            gripper_kp_scale,
            gripper_kd_scale,
        )

    def exit_teach_mode(self) -> ArmState:
        """Hold the measured pose before restoring normal arm gains."""
        self._require_connected()
        state = self.observe()
        self._act_joint_position(list(state.joint_positions))
        if self._teach_mode and self._teach_gain_snapshot is not None:
            teach_gains = []
            for ctrl in self._ctrls:
                gain = ctrl.get_gain()
                teach_gains.append({
                    "kp": np.asarray(gain.kp(), dtype=np.float64).copy(),
                    "kd": np.asarray(gain.kd(), dtype=np.float64).copy(),
                    "gripper_kp": float(gain.gripper_kp),
                    "gripper_kd": float(gain.gripper_kd),
                })
            try:
                for ctrl, snapshot in zip(self._ctrls, self._teach_gain_snapshot):
                    gain = ctrl.get_gain()
                    gain.kp()[:] = snapshot["kp"]
                    gain.kd()[:] = snapshot["kd"]
                    gain.gripper_kp = snapshot["gripper_kp"]
                    gain.gripper_kd = snapshot["gripper_kd"]
                    ctrl.set_gain(gain)
            except Exception:
                logger.exception(
                    "退出拖动示教恢复增益失败，正在将双臂回滚到拖动阻尼"
                )
                for ctrl, teach_gain in zip(self._ctrls, teach_gains):
                    try:
                        gain = ctrl.get_gain()
                        gain.kp()[:] = teach_gain["kp"]
                        gain.kd()[:] = teach_gain["kd"]
                        gain.gripper_kp = teach_gain["gripper_kp"]
                        gain.gripper_kd = teach_gain["gripper_kd"]
                        ctrl.set_gain(gain)
                    except Exception:
                        logger.critical("拖动阻尼增益回滚失败", exc_info=True)
                self._act_joint_position(list(self.observe().joint_positions))
                # Keep the mode/snapshot intact so callers cannot resume policy
                # control after a half-restored gain transaction.
                raise
        self._teach_mode = False
        self._teach_gain_snapshot = None
        self._act_joint_position(list(state.joint_positions))
        logger.info("ARX5 双臂退出拖动示教并保持实时位置")
        return state

    def set_teach_gripper(self, side: str, width_m: float) -> None:
        """Change one integrated gripper while holding measured arm joints."""
        if not self._teach_mode:
            raise RuntimeError("teach gripper commands require teach mode")
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        state = self.observe()
        target = list(state.joint_positions)
        target[6 if side == "left" else 13] = float(width_m)
        self._act_joint_position(target)

    @property
    def last_joint_position_target(self) -> list[float]:
        """最近一次实际交给 SDK 的 14 维关节位置目标（诊断用途）。"""
        return list(self._last_joint_position_target)

    # ── 原语 ──────────────────────────────────────────────────

    def observe(self) -> ArmState:
        """并行读取左右臂关节状态，合并为 14 维向量。

        注意：两臂读取存在微小时间差（约 <2ms），非严格原子快照。
        """
        self._require_connected()
        left_js, right_js = _run_dual(
            lambda: self._ctrls[0].get_joint_state(),
            lambda: self._ctrls[1].get_joint_state(),
            timeout=_GET_STATE_TIMEOUT_S,
            err_msg="get_joint_state timeout: ARX5 arm communication blocked",
        )
        q14 = _merge_arm_field(left_js, right_js, "pos", "gripper_pos")
        dq14 = _merge_arm_field(left_js, right_js, "vel", "gripper_vel")
        tau14 = _merge_arm_field(left_js, right_js, "torque", "gripper_torque")
        ls, rs = self._gripper_sdk_sign
        q14[6] *= ls
        q14[13] *= rs
        dq14[6] *= ls
        dq14[13] *= rs
        tau14[6] *= ls
        tau14[13] *= rs
        try:
            left_eef, right_eef = _run_dual(
                lambda: self._ctrls[0].get_eef_state(),
                lambda: self._ctrls[1].get_eef_state(),
                timeout=_GET_STATE_TIMEOUT_S,
                err_msg="get_eef_state timeout: ARX5 arm communication blocked",
            )
            eef_pose = (
                _pose6d_to_pose7(left_eef.pose_6d().tolist())
                + _pose6d_to_pose7(right_eef.pose_6d().tolist())
            )
        except Exception:
            logger.debug("ARX5 双臂 get_eef_state() 异常，eef_pose 置空", exc_info=True)
            eef_pose = []
        return ArmState(
            timestamp=time.perf_counter(),
            joint_positions=q14,
            joint_velocities=dq14,
            joint_torques=tau14,
            joint_external_torques=[],
            joint_positions_desired=[],
            eef_pose=eef_pose,
            eef_velocity=[],
            wrench_in_tcp=[],
            wrench_in_world=[],
        )

    def act(self, action: Action, *, state: ArmState | None = None) -> bool:
        """根据 ActionSpace 分发：JOINT_POSITION 直接下发，CARTESIAN 通过 IK 解算后下发。"""
        self._require_connected()
        try:
            if action.space == ActionSpace.JOINT_POSITION:
                self._act_joint_position(action.values)
                return True
            elif action.space == ActionSpace.CARTESIAN:
                return self._act_cartesian(action.values, state=state)
            else:
                logger.warning("ARX5 双臂不支持动作空间 %s，已跳过", action.space.value)
                return False
        except Exception:
            logger.error("ARX5 双臂 act() 异常，已跳过本步", exc_info=True)
            return False

    def move_joint_position(
        self,
        positions: Sequence[float],
        *,
        velocity: float | None = None,
        tolerance: float = 0.01,
        timeout_s: float = 30.0,
    ) -> bool:
        """smoothstep 插值规划移动到目标 14 维关节位置（含 gripper）。

        以 SDK controller_dt 对应的频率为插值步进频率，使用 smoothstep (3t²−2t³) 插值，
        起止速度为零，避免阶跃冲击。距离很近时直接下发目标，由 PD 控制收敛。
        duration 由纯关节（排除 gripper index 6/13）的最大位移 / velocity 决定。
        """
        self._require_connected()
        self._validate_length("positions", positions)
        target = np.asarray(positions, dtype=np.float64)
        self._clip_14d(target)

        state = self.observe()
        start = np.asarray(state.joint_positions, dtype=np.float64)

        joint_indices = [i for i in range(14) if i not in (6, 13)]
        max_disp = max(abs(float(target[i] - start[i])) for i in joint_indices)
        if max_disp < tolerance:
            return True

        dt = 1.0 / self._params.control_frequency_hz

        # 距离足够小，直接下发目标让 PD 收敛
        direct_threshold = 0.02  # rad
        if max_disp < direct_threshold:
            hold_steps = max(int(round(0.3 / dt)), 1)
            next_time = time.perf_counter()
            for _ in range(hold_steps):
                self.act(Action(ActionSpace.JOINT_POSITION, target.tolist()))
                next_time += dt
                self._rt_sleep_until(next_time)
            final = self.observe()
            return all(abs(final.joint_positions[i] - positions[i]) < tolerance for i in joint_indices)

        vel = velocity if velocity is not None else 1.0
        duration = max_disp / vel
        steps = max(int(round(duration / dt)), 1)

        diff = target - start
        next_time = time.perf_counter()
        for i in range(1, steps + 1):
            t = i / steps
            alpha = t * t * (3.0 - 2.0 * t)  # smoothstep
            cmd = start + diff * alpha
            self.act(Action(ActionSpace.JOINT_POSITION, cmd.tolist()))
            next_time += dt
            self._rt_sleep_until(next_time)

        final = self.observe()
        return all(abs(final.joint_positions[i] - positions[i]) < tolerance for i in joint_indices)

    def go_home(self, *, velocity: float | None = None, timeout_s: float = 60.0) -> bool:
        """双臂并行 reset_to_home，阻塞直到完成或超时。"""
        self._require_connected()
        _run_dual(
            lambda: self._ctrls[0].reset_to_home(),
            lambda: self._ctrls[1].reset_to_home(),
            timeout=min(_RESET_TIMEOUT_S, timeout_s),
            err_msg="ARX5 bimanual reset_to_home timeout",
        )
        # reset_to_home 会将 gain 重置为 SDK 默认值，需要重新应用全部配置增益。
        self._apply_joint_gains()
        self._apply_gripper_gains()
        return True

    # ── 动作实现 ──────────────────────────────────────────────

    def _act_joint_position(self, values: list[float]) -> None:
        """下发 14 维关节位置指令。

        接受 12 维（纯关节，gripper 保持当前值）或 14 维（含 gripper）。
        流程: 维度补全 → 限位校验 → gripper flip/clamp → 分发左右臂。
        """
        v = np.asarray(values, dtype=np.float64)
        if v.shape[0] not in (12, 14):
            logger.warning("ARX5 双臂 joint_position 期望 12 或 14 维，实际 %d，已跳过", v.shape[0])
            return

        # 12 维补全：读取当前 gripper 值填入 index 6 和 13
        if v.shape[0] == 12:
            q_now = np.asarray(self.observe().joint_positions, dtype=np.float64)
            q_now[:6] = v[:6]
            q_now[7:13] = v[6:12]
            v = q_now

        self._clip_14d(v)

        # 二值 snap: action 位置值 -> 极限位置（close=-0.01 / open=max），让电机顶到底施最大力
        if self._gripper_binary and self._enable_gripper:
            lg_max = self._params.joint_position_max[6]
            rg_max = self._params.joint_position_max[13]
            v[6]  = lg_max if float(v[6])  >= self._gripper_binary_threshold else self._gripper_binary_close
            v[13] = rg_max if float(v[13]) >= self._gripper_binary_threshold else self._gripper_binary_close

        # gripper: 上层为规范开合量（张开为正，与 gripper_width 同向）；下发 SDK 需乘各臂 sign
        ls, rs = self._gripper_sdk_sign
        left_g = float(v[6]) + self._gripper_command_offset_m
        right_g = float(v[13]) + self._gripper_command_offset_m
        if not self._enable_gripper:
            q_now = self.observe().joint_positions
            left_g, right_g = float(q_now[6]), float(q_now[13])
        elif self._flip_gripper_sign:
            left_g, right_g = -left_g, -right_g
        left_g = float(np.clip(left_g, self._params.joint_position_min[6], self._params.joint_position_max[6]))
        right_g = float(np.clip(right_g, self._params.joint_position_min[13], self._params.joint_position_max[13]))
        left_sdk = ls * left_g
        right_sdk = rs * right_g

        left_cmd = self._arx5.JointState(6)
        left_cmd.pos()[:] = v[:6]
        left_cmd.gripper_pos = left_sdk
        self._ctrls[0].set_joint_cmd(left_cmd)

        right_cmd = self._arx5.JointState(6)
        right_cmd.pos()[:] = v[7:13]
        right_cmd.gripper_pos = right_sdk
        self._ctrls[1].set_joint_cmd(right_cmd)

        # v 此时已经过 IK、限位和夹爪语义处理，正是本周期的发布目标。
        self._last_joint_position_target = v.tolist()

        if not self._use_background_send_recv:
            self._ctrls[0].send_recv_once()
            self._ctrls[1].send_recv_once()

    def _act_cartesian(self, values: Sequence[float], *, state: ArmState | None = None) -> bool:
        """笛卡尔空间控制：对左右臂分别 IK 解算后下发关节位置。

        接受 16 维 [left_x,y,z,qw,qx,qy,qz, left_gripper, right_x,y,z,qw,qx,qy,qz, right_gripper]
        或 14 维 [left_pose_7d, right_pose_7d]（gripper 保持当前值）。
        原子模式下精确目标失败会以共同进度系数缩步重试；仍失败时双臂和
        夹爪主动保持观测状态，并向上层返回 False 以触发丢弃旧 chunk。
        """
        n = len(values)
        if n not in (14, 16):
            logger.warning("CARTESIAN action 期望 14 或 16 维，实际 %d，已跳过", n)
            return False

        obs = state if state is not None else self.observe()
        q14 = obs.joint_positions

        if n == 16:
            left_pose, left_grip = values[:7], float(values[7])
            right_pose, right_grip = values[8:15], float(values[15])
        else:
            left_pose, left_grip = values[:7], float(q14[6])
            right_pose, right_grip = values[7:14], float(q14[13])

        left_6d = _pose7_to_pose6d(left_pose)
        right_6d = _pose7_to_pose6d(right_pose)

        left_q = np.asarray(q14[:6], dtype=np.float64)
        right_q = np.asarray(q14[7:13], dtype=np.float64)

        l_status, l_sol = self._solvers[0].multi_trial_ik(left_6d, left_q, 5)
        r_status, r_sol = self._solvers[1].multi_trial_ik(right_6d, right_q, 5)

        if l_status == 0 and r_status == 0:
            self._send_cartesian_joint_targets(l_sol, left_grip, r_sol, right_grip)
            return True

        if l_status != 0:
            self._log_ik_failure("左", 0, l_status, left_q, l_sol)
        if r_status != 0:
            self._log_ik_failure("右", 1, r_status, right_q, r_sol)

        if not self._cartesian_ik_atomic:
            left_target = left_q if l_status != 0 else l_sol
            right_target = right_q if r_status != 0 else r_sol
            self._send_cartesian_joint_targets(
                left_target, left_grip, right_target, right_grip
            )
            return True

        if len(obs.eef_pose) >= 14:
            current_left_pose = obs.eef_pose[:7]
            current_right_pose = obs.eef_pose[7:14]
            for factor in self._cartesian_ik_backoff_factors:
                retry_left_pose = _interpolate_pose7(
                    current_left_pose, left_pose, factor
                )
                retry_right_pose = _interpolate_pose7(
                    current_right_pose, right_pose, factor
                )
                retry_l_status, retry_l_sol = self._solvers[0].multi_trial_ik(
                    _pose7_to_pose6d(retry_left_pose), left_q, 5
                )
                retry_r_status, retry_r_sol = self._solvers[1].multi_trial_ik(
                    _pose7_to_pose6d(retry_right_pose), right_q, 5
                )
                if retry_l_status == 0 and retry_r_status == 0:
                    retry_left_grip = float(q14[6]) + factor * (
                        left_grip - float(q14[6])
                    )
                    retry_right_grip = float(q14[13]) + factor * (
                        right_grip - float(q14[13])
                    )
                    logger.warning(
                        "双臂 IK 精确目标失败，使用共同缩步 factor=%.3f 恢复",
                        factor,
                    )
                    self._send_cartesian_joint_targets(
                        retry_l_sol,
                        retry_left_grip,
                        retry_r_sol,
                        retry_right_grip,
                    )
                    return True

        # 原子模式下任一侧最终失败，两侧和夹爪都主动保持观测位置。
        self._hold_current_state(q14)
        return False

    def _send_cartesian_joint_targets(
        self,
        left_q: Sequence[float],
        left_grip: float,
        right_q: Sequence[float],
        right_grip: float,
    ) -> None:
        target_14 = (
            np.asarray(left_q, dtype=np.float64).tolist()
            + [float(left_grip)]
            + np.asarray(right_q, dtype=np.float64).tolist()
            + [float(right_grip)]
        )
        self._act_joint_position(target_14)

    def _hold_current_state(self, q14: Sequence[float]) -> None:
        """直接保持观测状态，绕过 gripper offset/二值化等 action 语义。"""
        left_cmd = self._arx5.JointState(6)
        left_cmd.pos()[:] = np.asarray(q14[:6], dtype=np.float64)
        left_cmd.gripper_pos = self._gripper_sdk_sign[0] * float(q14[6])
        self._ctrls[0].set_joint_cmd(left_cmd)

        right_cmd = self._arx5.JointState(6)
        right_cmd.pos()[:] = np.asarray(q14[7:13], dtype=np.float64)
        right_cmd.gripper_pos = self._gripper_sdk_sign[1] * float(q14[13])
        self._ctrls[1].set_joint_cmd(right_cmd)

        if not self._use_background_send_recv:
            self._ctrls[0].send_recv_once()
            self._ctrls[1].send_recv_once()

    def _log_ik_failure(
        self,
        side: str,
        arm_index: int,
        status: int,
        q_current: Sequence[float],
        q_candidate: Sequence[float],
    ) -> None:
        offset = 0 if arm_index == 0 else 7
        q_min = self._params.joint_position_min[offset:offset + 6]
        q_max = self._params.joint_position_max[offset:offset + 6]
        name = self._solvers[arm_index].get_ik_status_name(status)
        logger.warning(
            "%s臂 IK 解算失败: %s (status=%d); q_current=%s; "
            "q_candidate=%s; q_min=%s; q_max=%s",
            side,
            name,
            status,
            np.asarray(q_current, dtype=np.float64).tolist(),
            np.asarray(q_candidate, dtype=np.float64).tolist(),
            q_min,
            q_max,
        )

    # ── 内部工具 ──────────────────────────────────────────────

    def _create_solvers(self) -> tuple[Any, Any]:
        """从左右臂 RobotConfig 创建 Arx5Solver（用于 CARTESIAN IK 解算）。"""
        solvers = []
        for ctrl in self._ctrls:
            rc = ctrl.get_robot_config()
            solvers.append(self._arx5.Arx5Solver(
                rc.urdf_path, rc.joint_dof,
                rc.joint_pos_min, rc.joint_pos_max,
                rc.base_link_name, rc.eef_link_name,
                rc.gravity_vector,
            ))
        return tuple(solvers)

    def _init_one(
        self,
        model: str,
        interface_name: str,
        *,
        arm_side: str | None = None,
        urdf_path: Path | None = None,
    ) -> Any:
        """创建单臂 Arx5JointController（带超时保护，防止 CAN 通信阻塞卡死）。"""
        robot_cfg = self._arx5.RobotConfigFactory.get_instance().get_config(model)
        ctrl_cfg = self._arx5.ControllerConfigFactory.get_instance().get_config("joint_controller", robot_cfg.joint_dof)
        ctrl_cfg.background_send_recv = bool(self._use_background_send_recv)
        if "controller_dt" in self._ctrl_cfg:
            ctrl_cfg.controller_dt = float(self._ctrl_cfg["controller_dt"])
        if "over_current_cnt_max" in self._ctrl_cfg:
            ctrl_cfg.over_current_cnt_max = int(self._ctrl_cfg["over_current_cnt_max"])
        elif int(ctrl_cfg.over_current_cnt_max) < 1000:
            # 提高默认阈值，降低因瞬态过流导致的误触发通信阻塞
            ctrl_cfg.over_current_cnt_max = 1000

        apply_robot_config_overrides(robot_cfg, self._ctrl_cfg, arm_side=arm_side)
        if urdf_path is not None:
            robot_cfg.urdf_path = str(urdf_path)
            logger.info("ARX5 %s arm 使用自定义 URDF: %s", arm_side, urdf_path)
        else:
            logger.info(
                "ARX5 %s arm 未配置 urdf_path，使用 SDK 默认 URDF: %s",
                arm_side,
                robot_cfg.urdf_path,
            )

        result: list[Any] = [None]
        error: list[BaseException | None] = [None]

        def _create() -> None:
            try:
                result[0] = self._arx5.Arx5JointController(robot_cfg, ctrl_cfg, interface_name)
            except BaseException as exc:
                error[0] = exc

        t = threading.Thread(target=_create, daemon=True)
        t.start()
        t.join(timeout=_INIT_TIMEOUT_S)
        if t.is_alive():
            raise RuntimeError(f"Arx5JointController init timed out after {_INIT_TIMEOUT_S}s on {interface_name}")
        if error[0] is not None:
            raise error[0]
        return result[0]

    def _sync_params_from_sdk(self) -> None:
        """用 SDK 实际硬件参数构建 RobotParams — SDK 是硬件参数的唯一真实来源。

        gripper 范围: [-slack, gripper_width]，slack 由配置提供（Python 层概念，SDK 无此字段）。
        """
        left_rc = self._ctrls[0].get_robot_config()
        right_rc = self._ctrls[1].get_robot_config()
        left_cc = self._ctrls[0].get_controller_config()
        right_cc = self._ctrls[1].get_controller_config()

        ls = infer_gripper_sdk_sign(left_rc, self._ctrl_cfg, arm="left")
        rs = infer_gripper_sdk_sign(right_rc, self._ctrl_cfg, arm="right")
        self._gripper_sdk_sign = (ls, rs)
        logger.info("ARX5 双臂 gripper readout_sign: left=%d right=%d", ls, rs)

        def _arm_limits(rc: Any) -> tuple[list[float], list[float], list[float], list[float], float]:
            return (
                np.asarray(rc.joint_pos_min, dtype=float).tolist(),
                np.asarray(rc.joint_pos_max, dtype=float).tolist(),
                np.asarray(rc.joint_vel_max, dtype=float).tolist(),
                np.asarray(rc.joint_torque_max, dtype=float).tolist(),
                abs(float(rc.gripper_width)),
            )

        l_min, l_max, l_vel, l_tau, l_gw = _arm_limits(left_rc)
        r_min, r_max, r_vel, r_tau, r_gw = _arm_limits(right_rc)
        # SDK gripper_width is a nominal mechanism limit.  A calibrated
        # per-arm maximum may be lower to avoid driving into the physical stop.
        l_gw = min(l_gw, float(self._ctrl_cfg.get("left_gripper_max", l_gw)))
        r_gw = min(r_gw, float(self._ctrl_cfg.get("right_gripper_max", r_gw)))
        slack = float(self._ctrl_cfg.get("gripper_closed_slack", GRIPPER_CLOSED_SLACK_DEFAULT_M))
        dt = max(float(left_cc.controller_dt), float(right_cc.controller_dt))

        self._params = RobotParams(
            dof=14,
            joint_position_min=l_min + [-slack] + r_min + [-slack],
            joint_position_max=l_max + [l_gw] + r_max + [r_gw],
            joint_velocity_max=l_vel + [1.0] + r_vel + [1.0],
            joint_acceleration_max=[3.0] * 14,
            joint_torque_max=l_tau + [1.0] + r_tau + [1.0],
            gripper=None,
            home_position=self._params.home_position or [0.0] * 14,
            control_frequency_hz=1.0 / dt,
        )

    def _apply_gripper_gains(self) -> None:
        """为左右臂设置 gripper PD 增益：禁用时清零，启用时补齐 SDK 默认值为 0 的情况。"""
        for i, (kp, kd) in enumerate(zip(self._gripper_kp, self._gripper_kd)):
            gain = self._ctrls[i].get_gain()
            if not self._enable_gripper:
                gain.gripper_kp = 0.0
                gain.gripper_kd = 0.0
            else:
                gain.gripper_kp = kp
                gain.gripper_kd = kd
            self._ctrls[i].set_gain(gain)

    def _apply_joint_gains(self) -> None:
        """基于 SDK 默认值独立缩放左右臂六关节 PD 增益。

        每次都从 ControllerConfig.default_kp/default_kd 计算，避免重复调用时
        在已经缩放的 gain 上继续累乘。reset_to_home() 恢复 SDK 默认值后也会
        调用本方法，因此推理阶段的有效增益与 YAML 始终一致。
        """
        for i, side in enumerate(("left", "right")):
            ctrl = self._ctrls[i]
            controller_cfg = ctrl.get_controller_config()
            base_kp = np.asarray(controller_cfg.default_kp, dtype=np.float64)
            base_kd = np.asarray(controller_cfg.default_kd, dtype=np.float64)
            if base_kp.shape != (6,) or base_kd.shape != (6,):
                raise ValueError(
                    f"ARX5 {side} arm SDK default gain shape invalid: "
                    f"kp={base_kp.shape} kd={base_kd.shape}"
                )
            effective_kp = base_kp * self._joint_kp_scale[i]
            effective_kd = base_kd * self._joint_kd_scale[i]
            gain = ctrl.get_gain()
            gain.kp()[:] = effective_kp
            gain.kd()[:] = effective_kd
            ctrl.set_gain(gain)
            logger.info(
                "ARX5 %s arm joint gains: kp_scale=%.3f kd_scale=%.3f "
                "kp=%s kd=%s",
                side,
                self._joint_kp_scale[i],
                self._joint_kd_scale[i],
                effective_kp.tolist(),
                effective_kd.tolist(),
            )

    def _set_log_level(self, level_name: str) -> None:
        if self._arx5 is None or self._ctrls is None:
            return
        level = getattr(self._arx5.LogLevel, level_name, self._arx5.LogLevel.INFO)
        for ctrl in self._ctrls:
            ctrl.set_log_level(level)

    def _clip_14d(self, v: np.ndarray) -> None:
        """将 14 维位置裁切到限位范围内（推理输出可能略越界）。"""
        p = self._params
        lo = np.asarray(p.joint_position_min, dtype=np.float64)
        hi = np.asarray(p.joint_position_max, dtype=np.float64)
        mask = (v < lo) | (v > hi)
        if mask.any():
            for i in np.where(mask)[0]:
                side = "left" if int(i) < 7 else "right"
                kind = "gripper" if int(i) in (6, 13) else f"joint_{int(i) % 7}"
                logger.warning(
                    "ARX5 双臂 %s %s 位置 %.4f 超限 [%.4f, %.4f]，已裁切",
                    side, kind, float(v[i]), float(lo[i]), float(hi[i]),
                )
            np.clip(v, lo, hi, out=v)

    def _require_connected(self) -> None:
        if not self.is_connected():
            raise RuntimeError(f"机器人 '{self.name}' 尚未连接，请先调用 connect() 或使用 with 语句")


# ── 模块级工具 ────────────────────────────────────────────────


def _interpolate_pose7(
    current: Sequence[float], target: Sequence[float], factor: float
) -> list[float]:
    """沿 current→target 插值 pose7；位置线性插值，姿态走最短四元数路径。"""
    alpha = float(factor)
    current_xyz = np.asarray(current[:3], dtype=np.float64)
    target_xyz = np.asarray(target[:3], dtype=np.float64)
    xyz = current_xyz + alpha * (target_xyz - current_xyz)

    current_q = np.asarray(current[3:7], dtype=np.float64)
    target_q = np.asarray(target[3:7], dtype=np.float64)
    current_q /= max(float(np.linalg.norm(current_q)), 1e-12)
    target_q /= max(float(np.linalg.norm(target_q)), 1e-12)
    dot = float(np.dot(current_q, target_q))
    if dot < 0.0:
        target_q = -target_q
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        quat = current_q + alpha * (target_q - current_q)
        quat /= max(float(np.linalg.norm(quat)), 1e-12)
    else:
        angle = math.acos(dot)
        sin_angle = math.sin(angle)
        quat = (
            math.sin((1.0 - alpha) * angle) / sin_angle * current_q
            + math.sin(alpha * angle) / sin_angle * target_q
        )
    return xyz.tolist() + quat.tolist()


def _resolve_configured_urdf(value: str | Path | None) -> Path | None:
    """解析显式 URDF；未配置时返回 None，让 ARX SDK 使用模型默认值。"""
    if value is None or not str(value).strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ARX5 URDF 文件不存在: {path}")
    return path


def _run_dual(fn_left, fn_right, *, timeout: float, err_msg: str) -> tuple[Any, Any]:
    """并行执行左右臂操作，带超时和异常传播。任一臂超时或异常则 raise。"""
    results: list[Any] = [None, None]
    errors: list[BaseException | None] = [None, None]

    def _wrap(idx, fn):
        try:
            results[idx] = fn()
        except BaseException as exc:
            errors[idx] = exc

    tl = threading.Thread(target=_wrap, args=(0, fn_left), daemon=True)
    tr = threading.Thread(target=_wrap, args=(1, fn_right), daemon=True)
    tl.start()
    tr.start()
    tl.join(timeout=timeout)
    tr.join(timeout=timeout)
    if tl.is_alive() or tr.is_alive():
        raise RuntimeError(err_msg)
    for e in errors:
        if e is not None:
            raise e
    return results[0], results[1]


def _merge_arm_field(left_js: Any, right_js: Any, vec_attr: str, gripper_attr: str) -> list[float]:
    """合并左右臂 JointState 的某个字段为 14 维列表。

    vec_attr: JointState 上的方法名（如 "pos"），返回 6 维 ndarray。
    gripper_attr: JointState 上的属性名（如 "gripper_pos"），返回标量。
    """
    left_vec = np.asarray(getattr(left_js, vec_attr)(), dtype=float).tolist()
    right_vec = np.asarray(getattr(right_js, vec_attr)(), dtype=float).tolist()
    return left_vec + [float(getattr(left_js, gripper_attr))] + right_vec + [float(getattr(right_js, gripper_attr))]
