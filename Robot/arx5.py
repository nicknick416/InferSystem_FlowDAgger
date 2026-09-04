from __future__ import annotations

import logging
import math
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from Core import Action, ActionSpace, ArmState, RobotParams
from Core.config_schema import Arx5RobotConfig
from Robot.base import BaseRobot

# 优先使用现场安装的 ARX5 SDK；不存在时才回退到仓库 third_party。
_ARX5_SYSTEM_PYTHON = Path.home() / "arx5-sdk" / "python"
_ARX5_VENDORED_PYTHON = Path(__file__).resolve().parents[1] / "third_party" / "arx5-sdk" / "python"
_ARX5_SDK_PYTHON = str(
    _ARX5_SYSTEM_PYTHON if _ARX5_SYSTEM_PYTHON.is_dir() else _ARX5_VENDORED_PYTHON
)
if _ARX5_SDK_PYTHON not in sys.path:
    sys.path.insert(0, _ARX5_SDK_PYTHON)

logger = logging.getLogger(__name__)

_GET_STATE_TIMEOUT_S = 0.5

# 夹爪夹紧时编码器读数可能略低于 0 (m)。校验与 clip 默认允许的负向余量；可用 YAML `gripper_closed_slack` 覆盖。
GRIPPER_CLOSED_SLACK_DEFAULT_M = 0.01
_INIT_TIMEOUT_S = 30.0


# ══════════════════════════════════════════════════════════════
#  共享工具（供 arx5_bimanual 复用）
# ══════════════════════════════════════════════════════════════


def call_with_timeout(fn, *, timeout: float, err_msg: str) -> Any:
    """在子线程中执行 fn()，超时则 raise RuntimeError。

    用于保护可能阻塞在 C++ 层的 SDK 调用，确保主线程能响应 Ctrl-C。
    """
    result: list[Any] = [None]
    error: list[BaseException | None] = [None]

    def _run():
        try:
            result[0] = fn()
        except BaseException as exc:
            error[0] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise RuntimeError(err_msg)
    if error[0] is not None:
        raise error[0]
    return result[0]


def apply_robot_config_overrides(
    robot_cfg: Any,
    ctrl_cfg: dict[str, Any],
    *,
    arm_side: str | None = None,
) -> None:
    """将 YAML 中的关节限位 / gripper 配置覆盖到 SDK robot_config（须在控制器实例化前调用）。

    arm_side: 双臂时传入 \"left\" / \"right\"，以便使用 left_gripper_open_readout 等键；单臂保持 None。
    """
    # Some ARX wheels retain the absolute URDF path from the machine where the
    # wheel was built.  Fall back to the vendored SDK model when that path is
    # not valid on the runtime machine.
    urdf_path = Path(str(robot_cfg.urdf_path)).expanduser()
    if not urdf_path.is_file():
        vendored_urdf = (
            Path(__file__).resolve().parents[1]
            / "third_party"
            / "arx5-sdk"
            / "models"
            / f"{urdf_path.name or 'X5.urdf'}"
        )
        if vendored_urdf.is_file():
            robot_cfg.urdf_path = str(vendored_urdf)

    lim = ctrl_cfg.get("joint_limits", {})
    if "velocity_max" in lim:
        robot_cfg.joint_vel_max = np.asarray(lim["velocity_max"], dtype=np.float64)
    if "torque_max" in lim:
        robot_cfg.joint_torque_max = np.asarray(lim["torque_max"], dtype=np.float64)
    if "position_min_deg" in lim:
        robot_cfg.joint_pos_min = np.asarray([math.radians(float(v)) for v in lim["position_min_deg"]], dtype=np.float64)
    if "position_max_deg" in lim:
        robot_cfg.joint_pos_max = np.asarray([math.radians(float(v)) for v in lim["position_max_deg"]], dtype=np.float64)
    if "gripper_width" in ctrl_cfg:
        robot_cfg.gripper_width = float(ctrl_cfg["gripper_width"])
    if arm_side == "left" and "left_gripper_open_readout" in ctrl_cfg:
        robot_cfg.gripper_open_readout = float(ctrl_cfg["left_gripper_open_readout"])
    elif arm_side == "right" and "right_gripper_open_readout" in ctrl_cfg:
        robot_cfg.gripper_open_readout = float(ctrl_cfg["right_gripper_open_readout"])
    if "gripper_torque_max" in ctrl_cfg:
        robot_cfg.gripper_torque_max = float(ctrl_cfg["gripper_torque_max"])
    if "gripper_vel_max" in ctrl_cfg:
        robot_cfg.gripper_vel_max = float(ctrl_cfg["gripper_vel_max"])
    if "gripper_open_readout" in ctrl_cfg and arm_side is None:
        robot_cfg.gripper_open_readout = float(ctrl_cfg["gripper_open_readout"])


def infer_gripper_sdk_sign(_robot_cfg: Any, ctrl_cfg: dict[str, Any], *, arm: str | None = None) -> int:
    """Python 层对 SDK `JointState.gripper_pos` 的额外符号（默认不重映射）。

    arx5-sdk 已用 `robot_config.gripper_open_readout`（可为负）把电机角换算到米、并期望读数在
    约 [0, gripper_width]。再根据 `gripper_open_readout < 0` 在 Python 里乘 -1 会**二次取反**，
    导致张开仍为负且指令与 SDK  clip 不一致。

    因此此处**不再**根据 open_readout 推断符号；仅当配置显式要求与 SDK 输出相反时设置:
    `gripper_readout_sign` / `left_gripper_readout_sign` / `right_gripper_readout_sign` 为 ±1。

    - 观测: canonical = sign * raw_sdk
    - 下发: raw_sdk = sign * canonical
    """
    if arm == "left" and "left_gripper_readout_sign" in ctrl_cfg:
        return int(ctrl_cfg["left_gripper_readout_sign"])
    if arm == "right" and "right_gripper_readout_sign" in ctrl_cfg:
        return int(ctrl_cfg["right_gripper_readout_sign"])
    if arm is None and "gripper_readout_sign" in ctrl_cfg:
        return int(ctrl_cfg["gripper_readout_sign"])
    return 1


def apply_gripper_gain(ctrl: Any, enable: bool, kp: float, kd: float) -> None:
    """设置 gripper PD 增益：禁用时清零，启用时补齐 SDK 默认值为 0 的情况。"""
    gain = ctrl.get_gain()
    if not enable:
        gain.gripper_kp = 0.0
        gain.gripper_kd = 0.0
    else:
        gain.gripper_kp = kp
        gain.gripper_kd = kd
    ctrl.set_gain(gain)


def create_controller(
    arx5_module: Any,
    model: str,
    interface_name: str,
    ctrl_cfg: dict[str, Any],
    *,
    use_background_send_recv: bool = True,
) -> Any:
    """创建 Arx5JointController（带超时保护和 over_current_cnt_max 安全默认值）。"""
    robot_cfg = arx5_module.RobotConfigFactory.get_instance().get_config(model)
    sdk_ctrl_cfg = arx5_module.ControllerConfigFactory.get_instance().get_config("joint_controller", robot_cfg.joint_dof)
    sdk_ctrl_cfg.background_send_recv = bool(use_background_send_recv)
    if "controller_dt" in ctrl_cfg:
        sdk_ctrl_cfg.controller_dt = float(ctrl_cfg["controller_dt"])
    if "over_current_cnt_max" in ctrl_cfg:
        sdk_ctrl_cfg.over_current_cnt_max = int(ctrl_cfg["over_current_cnt_max"])
    elif int(sdk_ctrl_cfg.over_current_cnt_max) < 1000:
        sdk_ctrl_cfg.over_current_cnt_max = 1000

    apply_robot_config_overrides(robot_cfg, ctrl_cfg)

    controller = call_with_timeout(
        lambda: arx5_module.Arx5JointController(robot_cfg, sdk_ctrl_cfg, interface_name),
        timeout=_INIT_TIMEOUT_S,
        err_msg=f"Arx5JointController init timed out after {_INIT_TIMEOUT_S}s on {interface_name}",
    )
    return controller, robot_cfg, sdk_ctrl_cfg


def js_to_7d(js: Any, gripper_sdk_sign: int = 1) -> tuple[list[float], list[float], list[float]]:
    """从 SDK JointState 提取 7 维 (6 关节 + gripper) 的 pos / vel / torque。

    gripper 分量按 gripper_sdk_sign 转为规范方向（张开为正，与 gripper_width 同向）。
    """
    s = int(gripper_sdk_sign)
    q = np.asarray(js.pos(), dtype=float).tolist() + [s * float(js.gripper_pos)]
    dq = np.asarray(js.vel(), dtype=float).tolist() + [s * float(js.gripper_vel)]
    tau = np.asarray(js.torque(), dtype=float).tolist() + [s * float(js.gripper_torque)]
    return q, dq, tau


# ══════════════════════════════════════════════════════════════
#  Arx5Robot — 单臂
# ══════════════════════════════════════════════════════════════


@BaseRobot.register("arx5")
class Arx5Robot(BaseRobot):
    """ARX5 单臂驱动 (基于 arx5-interface Python SDK)。

    支持 JOINT_POSITION 和 CARTESIAN（通过 IK 解算后下发关节位置）。
    observe() 返回 7 维 [q0..q5, gripper_pos]。
    """

    def __init__(
        self,
        *,
        model: str,
        interface_name: str,
        name: str | None = None,
        use_background_send_recv: bool = True,
        log_level: str = "INFO",
        params: RobotParams,
        ctrl_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name or f"arx5_{model}", robot_type="arx5", dof=params.dof)
        self._model = model
        self._interface_name = interface_name
        self._use_background_send_recv = use_background_send_recv
        self._log_level_name = log_level.upper()
        self._params = params
        self._ctrl_cfg = ctrl_cfg or {}

        self._enable_gripper = bool(self._ctrl_cfg.get("enable_gripper", False))
        self._flip_gripper_sign = bool(self._ctrl_cfg.get("flip_gripper_sign", False))
        self._gripper_kp = float(self._ctrl_cfg.get("gripper_kp", 4.0))
        self._gripper_kd = float(self._ctrl_cfg.get("gripper_kd", 0.24))

        self._arx5: Any | None = None
        self._controller: Any | None = None
        self._solver: Any | None = None  # Arx5Solver，用于 CARTESIAN IK
        self._connected = False
        self._gripper_sdk_sign: int = 1  # 仅当 YAML 设 gripper_readout_sign 时在 connect 后更新
        _slack = float(self._ctrl_cfg.get("gripper_closed_slack", GRIPPER_CLOSED_SLACK_DEFAULT_M))
        self._gripper_closed_slack_m = _slack
        _user_gmin = float(self._ctrl_cfg.get("gripper_min", -_slack))
        self._gripper_canonical_min = min(_user_gmin, -_slack)

    # ── 工厂 ──────────────────────────────────────────────────

    @classmethod
    def _from_config_dict(cls, robot_cfg: Arx5RobotConfig | dict[str, Any]) -> Arx5Robot:
        """从 typed config 创建实例。也兼容旧的 dict 调用路径。"""
        if isinstance(robot_cfg, dict):
            robot_cfg = Arx5RobotConfig.model_validate(robot_cfg)

        ctrl = robot_cfg.control
        merged = ctrl.model_dump(exclude_none=True)
        if robot_cfg.joint_limits:
            merged["joint_limits"] = robot_cfg.joint_limits.model_dump(exclude_none=True)

        return cls(
            model=robot_cfg.model,
            interface_name=robot_cfg.resolved_interface,
            name=robot_cfg.name,
            use_background_send_recv=ctrl.background_send_recv,
            log_level=ctrl.log_level,
            params=cls._build_params(robot_cfg),
            ctrl_cfg=merged,
        )

    @classmethod
    def _build_params(cls, cfg: Arx5RobotConfig) -> RobotParams:
        """从 typed config 构建初始 RobotParams（connect 后会被 SDK 实际值覆盖）。"""
        dof = cfg.dof
        ctrl = cfg.control
        lim = cfg.joint_limits  # 关节限位只从顶层读取，control 段不再重复
        return RobotParams(
            dof=dof,
            joint_position_min=[math.radians(v) for v in (lim.position_min_deg if lim and lim.position_min_deg else [-180.0] * dof)],
            joint_position_max=[math.radians(v) for v in (lim.position_max_deg if lim and lim.position_max_deg else [180.0] * dof)],
            joint_velocity_max=list(lim.velocity_max if lim and lim.velocity_max else [2.0] * dof),
            joint_acceleration_max=list(lim.acceleration_max if lim and lim.acceleration_max else [3.0] * dof),
            joint_torque_max=list(lim.torque_max if lim and lim.torque_max else [30.0] * dof),
            gripper=None,
            home_position=[math.radians(v) for v in (ctrl.home_position_deg or [0.0] * dof)],
            control_frequency_hz=ctrl.frequency_hz,
        )

    # ── 生命周期 ──────────────────────────────────────────────

    def connect(self) -> None:
        """延迟导入 arx5_interface，创建控制器并从 SDK 同步实际硬件参数。"""
        if self._connected:
            return
        import arx5_interface as arx5

        controller, robot_cfg, sdk_ctrl_cfg = create_controller(
            arx5, self._model, self._interface_name, self._ctrl_cfg,
            use_background_send_recv=self._use_background_send_recv,
        )
        self._controller = controller
        self._arx5 = arx5
        self.dof = int(robot_cfg.joint_dof)
        self._solver = arx5.Arx5Solver(
            robot_cfg.urdf_path, robot_cfg.joint_dof,
            robot_cfg.joint_pos_min, robot_cfg.joint_pos_max,
            robot_cfg.base_link_name, robot_cfg.eef_link_name,
            robot_cfg.gravity_vector,
        )
        self._sync_params_from_sdk(robot_cfg, sdk_ctrl_cfg)
        self._set_log_level(self._log_level_name)
        apply_gripper_gain(self._controller, self._enable_gripper, self._gripper_kp, self._gripper_kd)
        self._connected = True
        import atexit
        atexit.register(self.disconnect)
        logger.info("已连接 ARX5: model=%s interface=%s dof=%d", self._model, self._interface_name, self.dof)

    def disconnect(self) -> None:
        """安全断开：先尝试回零（防止松弛坠落），再进入被动态。"""
        if not self._connected:
            return
        self._connected = False
        if self._controller is not None:
            try:
                logger.info("ARX5 断开前回零 ...")
                self._controller.reset_to_home()
            except Exception:
                logger.warning("ARX5 断开前回零失败（将直接进入阻尼态）", exc_info=True)
            try:
                self._controller.set_to_damping()
            except Exception:
                logger.warning("ARX5 set_to_damping 失败", exc_info=True)
        self._controller = None
        self._solver = None
        self._arx5 = None
        logger.info("已断开 ARX5: %s", self.name)

    def enable(self) -> None:
        self._require_connected()

    def stop(self) -> None:
        self._require_connected()
        self._controller.set_to_damping()

    def emergency_stop(self) -> None:
        self._require_connected()
        self._controller.set_to_damping()
        logger.warning("ARX5 紧急停止已执行 (set_to_damping)")

    def clear_fault(self) -> None:
        self._require_connected()
        self._controller.reset_to_home()

    def get_params(self) -> RobotParams:
        return self._params

    def is_connected(self) -> bool:
        return self._connected and self._controller is not None

    def is_operational(self) -> bool:
        return self.is_connected()

    def is_busy(self) -> bool:
        return False

    def is_fault(self) -> bool:
        return False

    # ── 原语 ──────────────────────────────────────────────────

    def observe(self) -> ArmState:
        """读取关节状态，返回 7 维 [q0..q5, gripper]（带超时保护）。"""
        self._require_connected()
        js = call_with_timeout(
            lambda: self._controller.get_joint_state(),
            timeout=_GET_STATE_TIMEOUT_S,
            err_msg="get_joint_state timeout: ARX5 arm communication blocked",
        )
        q, dq, tau = js_to_7d(js, self._gripper_sdk_sign)
        try:
            eef = self._controller.get_eef_state()
            eef_pose = _pose6d_to_pose7(eef.pose_6d().tolist())
        except Exception:
            logger.debug("ARX5 get_eef_state() 异常，eef_pose 置空", exc_info=True)
            eef_pose = []
        return ArmState(
            timestamp=time.perf_counter(),
            joint_positions=q, joint_velocities=dq, joint_torques=tau,
            joint_external_torques=[], joint_positions_desired=[],
            eef_pose=eef_pose, eef_velocity=[],
            wrench_in_tcp=[], wrench_in_world=[],
        )

    def act(self, action: Action, *, state: ArmState | None = None) -> None:
        self._require_connected()
        try:
            if action.space == ActionSpace.JOINT_POSITION:
                self._act_joint_position(action.values)
            elif action.space == ActionSpace.CARTESIAN:
                self._act_cartesian(action.values, state=state)
            else:
                logger.warning("ARX5 不支持动作空间 %s，已跳过", action.space.value)
        except Exception:
            logger.error("ARX5 act() 异常，已跳过本步", exc_info=True)

    def move_joint_position(
        self,
        positions: Sequence[float],
        *,
        velocity: float | None = None,
        tolerance: float = 0.01,
        timeout_s: float = 30.0,
    ) -> bool:
        """smoothstep 插值规划移动到目标关节位置（6 维纯关节，gripper 保持当前值）。

        以 YAML control.frequency_hz 为插值步进频率，使用 smoothstep (3t²−2t³) 插值，
        起止速度为零，避免阶跃冲击。距离很近时直接下发目标，由 PD 控制收敛。
        duration = max_displacement / velocity。
        """
        self._require_connected()
        self._validate_length("positions", positions)
        positions = self._clip_joint_positions(list(positions))

        state = self.observe()
        start = state.joint_positions[:self.dof]
        target = list(positions)
        gripper_now = state.joint_positions[self.dof]

        max_disp = max(abs(t - s) for t, s in zip(target, start))
        if max_disp < tolerance:
            return True

        # 距离足够小，直接下发目标让 PD 收敛
        direct_threshold = 0.05  # rad
        if max_disp < direct_threshold:
            self.act(Action(ActionSpace.JOINT_POSITION, target + [gripper_now]))
            dt = 1.0 / self._params.control_frequency_hz
            # 持续下发一小段时间确保 SDK 后台线程执行
            hold_steps = max(int(round(0.3 / dt)), 1)
            next_time = time.perf_counter()
            for _ in range(hold_steps):
                self.act(Action(ActionSpace.JOINT_POSITION, target + [gripper_now]))
                next_time += dt
                self._rt_sleep_until(next_time)
            final = self.observe()
            return all(abs(a - b) < tolerance for a, b in zip(final.joint_positions[:self.dof], target))

        vel = velocity if velocity is not None else 1.0
        duration = max_disp / vel
        dt = 1.0 / self._params.control_frequency_hz
        steps = max(int(round(duration / dt)), 1)

        next_time = time.perf_counter()
        for i in range(1, steps + 1):
            t = i / steps
            alpha = t * t * (3.0 - 2.0 * t)  # smoothstep
            cmd = [s + (tgt - s) * alpha for s, tgt in zip(start, target)]
            cmd.append(gripper_now)
            self.act(Action(ActionSpace.JOINT_POSITION, cmd))
            next_time += dt
            self._rt_sleep_until(next_time)

        final = self.observe()
        return all(abs(a - b) < tolerance for a, b in zip(final.joint_positions[:self.dof], target))

    def go_home(self, *, velocity: float | None = None, timeout_s: float = 60.0) -> bool:
        self._require_connected()
        self._controller.reset_to_home()
        return True

    # ── 动作实现 ──────────────────────────────────────────────

    def _act_joint_position(self, target_q: Sequence[float]) -> None:
        """接受 dof 维（纯关节）或 dof+1 维（末位为 gripper 目标）。"""
        n = len(target_q)
        if n not in (self.dof, self.dof + 1):
            logger.warning("action.values 长度期望 %d 或 %d，实际 %d，已跳过", self.dof, self.dof + 1, n)
            return

        q_only = self._clip_joint_positions(list(target_q[:self.dof]))

        cmd = self._arx5.JointState(self.dof)
        cmd.pos()[:] = np.asarray(q_only, dtype=np.float64)

        if n == self.dof + 1 and self._enable_gripper:
            # 上层/策略使用规范开合量 g_canonical ∈ [0, width]（张开为正）
            g = float(target_q[self.dof])
            if self._flip_gripper_sign:
                g = -g
            g = float(np.clip(g, self._gripper_canonical_min, self._gripper_width))
            cmd.gripper_pos = self._gripper_sdk_sign * g

        self._controller.set_joint_cmd(cmd)
        if not self._use_background_send_recv:
            self._controller.send_recv_once()

    def _act_cartesian(self, values: Sequence[float], *, state: ArmState | None = None) -> None:
        """笛卡尔空间控制：通过 IK 解算后下发关节位置。

        接受 7 维 [x,y,z,qw,qx,qy,qz] 或 8 维（末位为 gripper 目标）。
        IK 解算失败时记录警告并保持当前关节位置，不中断控制循环。
        """
        n = len(values)
        if n not in (7, 8):
            logger.warning("CARTESIAN action 期望 7 或 8 维，实际 %d，已跳过", n)
            return

        pose_6d = _pose7_to_pose6d(values[:7])
        obs = state if state is not None else self.observe()
        q_current = np.asarray(obs.joint_positions[:self.dof], dtype=np.float64)
        ik_status, q_sol = self._solver.multi_trial_ik(pose_6d, q_current, 5)
        if ik_status != 0:
            status_name = self._solver.get_ik_status_name(ik_status)
            logger.warning("IK 解算失败: %s (status=%d)，保持当前关节位置", status_name, ik_status)
            return

        target = q_sol.tolist()
        if n == 8:
            target.append(float(values[7]))
        self._act_joint_position(target)

    # ── 内部工具 ──────────────────────────────────────────────

    def _set_log_level(self, level_name: str) -> None:
        if self._arx5 is None or self._controller is None:
            return
        self._controller.set_log_level(getattr(self._arx5.LogLevel, level_name, self._arx5.LogLevel.INFO))

    def _sync_params_from_sdk(self, robot_cfg: Any, ctrl_cfg: Any) -> None:
        """用 SDK 返回的实际硬件参数覆盖 YAML 初始值。"""
        self._params = RobotParams(
            dof=int(robot_cfg.joint_dof),
            joint_position_min=np.asarray(robot_cfg.joint_pos_min, dtype=float).tolist(),
            joint_position_max=np.asarray(robot_cfg.joint_pos_max, dtype=float).tolist(),
            joint_velocity_max=np.asarray(robot_cfg.joint_vel_max, dtype=float).tolist(),
            joint_acceleration_max=[3.0] * int(robot_cfg.joint_dof),
            joint_torque_max=np.asarray(robot_cfg.joint_torque_max, dtype=float).tolist(),
            gripper=None,
            home_position=self._params.home_position,
            control_frequency_hz=1.0 / float(ctrl_cfg.controller_dt),
        )
        self._gripper_width = float(self._ctrl_cfg.get("gripper_max", abs(float(robot_cfg.gripper_width))))
        self._gripper_sdk_sign = infer_gripper_sdk_sign(robot_cfg, self._ctrl_cfg)
        _slack = float(self._ctrl_cfg.get("gripper_closed_slack", GRIPPER_CLOSED_SLACK_DEFAULT_M))
        self._gripper_closed_slack_m = _slack
        _user_gmin = float(self._ctrl_cfg.get("gripper_min", -_slack))
        self._gripper_canonical_min = min(_user_gmin, -_slack)
        logger.info(
            "ARX5 gripper: width=%.4f readout_sign=%d clip_min=%.5f (默认信任 SDK 米制读数；符号仅来自 gripper_readout_sign)",
            self._gripper_width,
            self._gripper_sdk_sign,
            self._gripper_canonical_min,
        )

    def _clip_joint_positions(self, positions: list[float]) -> list[float]:
        """将关节位置裁切到限位范围内（推理输出可能略越界）。"""
        params = self.get_params()
        for i, (pos, lo, hi) in enumerate(zip(
            positions, params.joint_position_min, params.joint_position_max,
        )):
            if not (lo <= pos <= hi):
                logger.warning(
                    "ARX5 joint_%d 位置 %.4f 超限 [%.4f, %.4f]，已裁切",
                    i, pos, lo, hi,
                )
                positions[i] = max(lo, min(pos, hi))
        return positions

    def _require_connected(self) -> None:
        if not self._connected or self._controller is None:
            raise RuntimeError(f"机器人 '{self.name}' 尚未连接，请先调用 connect() 或使用 with 语句")


# ── 内部工具 ─────────────────────────────────────────────────


def _pose6d_to_pose7(pose_6d: list[float]) -> list[float]:
    """ARX SDK ``[x,y,z,roll,pitch,yaw]`` → quaternion pose.

    The SDK's ``pose_6d`` orientation is ZYX roll-pitch-yaw, not an
    axis-angle rotation vector.  Keep this conversion consistent with the
    ARX data collector and ``Arx5Solver`` Cartesian target convention.
    """
    if len(pose_6d) < 6:
        return []
    x, y, z, roll, pitch, yaw = (float(v) for v in pose_6d[:6])
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return [
        x,
        y,
        z,
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _pose7_to_pose6d(pose_7: Sequence[float]) -> np.ndarray:
    """Quaternion pose → ARX SDK ``[x,y,z,roll,pitch,yaw]``."""
    x, y, z = float(pose_7[0]), float(pose_7[1]), float(pose_7[2])
    qw, qx, qy, qz = float(pose_7[3]), float(pose_7[4]), float(pose_7[5]), float(pose_7[6])
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    else:
        qw, qx, qy, qz = (v / norm for v in (qw, qx, qy, qz))
    roll = math.atan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    sin_pitch = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return np.array([x, y, z, roll, pitch, yaw], dtype=np.float64)
