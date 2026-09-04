"""推理输出 → 机器人控制的桥接层。

推理模型输出原始向量 (如 [j1..j7, gripper])，
ActionDispatcher 按配置拆分并分发给机器人手臂和夹爪。

用法::

    from Inference.dispatch import ActionDispatcher, build_state_vector

    dispatcher = ActionDispatcher.from_config(robot, gripper, config)

    # 控制循环
    for action_vec in actions:
        dispatcher.dispatch(action_vec)

    # 构建推理输入状态
    state = build_state_vector(arm_state, gripper_width, gripper_max_width)
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from Core import Action, ActionSpace, ArmState
from Core.config_schema import InferenceConfig, SystemConfig
from Inference.action_processing import canonicalize_action_values
from Robot.base import BaseRobot
from Robot.gripper import BaseGripper

logger = logging.getLogger(__name__)


class ActionDispatcher:
    """将推理输出的原始向量分发给机器人和夹爪。

    推理模型输出的 action 通常是一个扁平向量，如:
      [j1, j2, j3, j4, j5, j6, j7, gripper]  (8 维)

    ActionDispatcher 根据配置把它拆分并分发:
      - 前 arm_dim 个维度 → robot.act(Action)
      - gripper_index 位置的维度 → gripper 控制

    配置 (YAML inference 段)::

        inference:
          arm_dof: 7                  # 手臂关节维度（默认自动从 robot.dof 读取）
          gripper_action_index: 7     # 夹爪在 action 向量中的下标（默认自动推断）
          gripper_threshold: 0.5      # 二值夹爪阈值
          gripper_mode: binary        # 夹爪控制模式:
                                      #   binary      — 阈值后开合
                                      #   width       — action 值作为目标宽度 (m)
          gripper_wait_during_inference: false
                                      # 推理期间是否等待每条夹爪命令完成
          hold_gripper_closed_during_inference: false
                                      # true 时忽略模型的夹爪指令
          action_space: joint_position
    """

    _SPACE_MAP: dict[str, ActionSpace] = {
        "joint_position": ActionSpace.JOINT_POSITION,
        "joint_velocity": ActionSpace.JOINT_VELOCITY,
        "joint_torque": ActionSpace.JOINT_TORQUE,
        "cartesian": ActionSpace.CARTESIAN,
    }
    _GRIPPER_WIDTH_HYSTERESIS_M = 0.001

    def __init__(
        self,
        robot: BaseRobot,
        gripper: BaseGripper | None = None,
        *,
        arm_dim: int | None = None,
        gripper_index: int = -1,
        gripper_threshold: float = 0.5,
        gripper_mode: str = "binary",
        gripper_wait_during_inference: bool = False,
        hold_gripper_closed_during_inference: bool = False,
        action_space: ActionSpace = ActionSpace.JOINT_POSITION,
        max_joint_velocity_rad_s: float | None = None,
        max_eef_xyz_step: float = 0.0,
        eef_translation_deadband_m: float = 0.0,
        max_eef_rot_step_deg: float = 0.0,
        max_eef_gripper_step: float = 0.0,
    ) -> None:
        self._robot = robot
        self._gripper = gripper
        self._arm_dim = arm_dim if arm_dim is not None else robot.dof
        self._gripper_index = gripper_index
        self._gripper_threshold = gripper_threshold
        self._gripper_mode = gripper_mode
        self._gripper_wait_during_inference = gripper_wait_during_inference
        self._hold_gripper_closed_during_inference = (
            hold_gripper_closed_during_inference
        )
        self._action_space = action_space
        self._last_gripper_open: bool | None = None
        self._last_gripper_width_cmd: float | None = None
        self._gripper_max_width: float | None = None  # width 模式下缓存
        self.last_canonical_action_values: list[float] | None = None
        self.last_robot_action_values: list[float] | None = None
        self.last_dispatch_succeeded: bool | None = None
        self.last_limit_events: dict[str, int] = {}
        self._max_eef_xyz_step = max(0.0, float(max_eef_xyz_step or 0.0))
        self._eef_translation_deadband_m = max(
            0.0, float(eef_translation_deadband_m or 0.0)
        )
        self._max_eef_rot_step_rad = max(0.0, math.radians(float(max_eef_rot_step_deg or 0.0)))
        self._max_eef_gripper_step = max(0.0, float(max_eef_gripper_step or 0.0))
        # 速度裁切
        self._v_max = max_joint_velocity_rad_s
        self._last_arm_cmd: list[float] | None = None
        self._last_dispatch_t: float | None = None

    @classmethod
    def from_config(
        cls,
        robot: BaseRobot,
        gripper: BaseGripper | None,
        config: SystemConfig | dict[str, Any],
    ) -> ActionDispatcher:
        """从 SystemConfig 或原始 dict 创建 dispatcher。"""
        if isinstance(config, dict):
            config = SystemConfig.model_validate(config)

        infer_cfg = config.inference or InferenceConfig()

        action_space = cls._SPACE_MAP.get(
            infer_cfg.action_space.value, ActionSpace.JOINT_POSITION,
        )

        return cls(
            robot=robot,
            gripper=gripper,
            arm_dim=infer_cfg.arm_dof if infer_cfg.arm_dof is not None else robot.dof,
            gripper_index=(
                infer_cfg.gripper_action_index
                if infer_cfg.gripper_action_index is not None
                else (robot.dof if gripper is not None else -1)
            ),
            gripper_threshold=infer_cfg.gripper_threshold,
            gripper_mode=infer_cfg.gripper_mode.value,
            gripper_wait_during_inference=(
                infer_cfg.gripper_wait_during_inference
            ),
            hold_gripper_closed_during_inference=(
                infer_cfg.hold_gripper_closed_during_inference
            ),
            action_space=action_space,
            max_joint_velocity_rad_s=infer_cfg.max_joint_velocity_rad_s,
            max_eef_xyz_step=infer_cfg.max_eef_xyz_step,
            eef_translation_deadband_m=infer_cfg.eef_translation_deadband_m,
            max_eef_rot_step_deg=infer_cfg.max_eef_rot_step_deg,
            max_eef_gripper_step=infer_cfg.max_eef_gripper_step,
        )

    def reset_velocity_tracking(self) -> None:
        """清空手臂速度裁切和夹爪命令去重的参考点。

        在相位切换（如 go_home 之后、插值到 inference_home 之前、开始新
        episode 之前）调用。否则 `_last_arm_cmd` 会停留在旧 chunk 的最后
        命令——而 go_home 绕过 dispatcher 直接走硬件层，dispatcher 感知
        不到机器人已经被动过了——下一次 dispatch 会把命令贴在旧位置附近
        缓慢爬行，看上去就是机器人"往回跑"。
        """
        self._last_arm_cmd = None
        self._last_dispatch_t = None
        self._last_gripper_width_cmd = None

    def dispatch(self, action_values: list[float], *, state: ArmState | None = None) -> bool:
        """将推理输出分发给机器人和夹爪。

        Args:
            action_values: 推理模型输出的原始向量
            state: 可选已有机器人观测，传给 robot.act() 避免重复读取。
        """
        action_values = canonicalize_action_values(action_values, self._action_space)
        self.last_canonical_action_values = list(action_values)
        self.last_limit_events = {}

        # 手臂控制
        arm_values = self._robot_action_values(action_values)

        unclamped_arm_values = list(arm_values)
        arm_values = self._clamp_cartesian_step(arm_values, state)
        if len(arm_values) == len(unclamped_arm_values) and not np.allclose(
            arm_values, unclamped_arm_values, rtol=0.0, atol=1e-9
        ):
            self.last_limit_events["eef_step_clamps"] = 1

        # ── 速度裁切：|Δq|/dt ≤ v_max（仅 joint_position）──
        # 参考点为 "上一次实际下发的命令" 而非机器人观测，避免和硬件跟踪
        # 误差形成正反馈。首帧（_last_arm_cmd=None）跳过裁切。
        now = time.perf_counter()
        if (
            self._v_max is not None
            and self._action_space == ActionSpace.JOINT_POSITION
            and self._last_arm_cmd is not None
            and self._last_dispatch_t is not None
        ):
            # 把 dt 夹到 50ms 以内，防止长时间暂停后放开一大步
            dt = min(max(now - self._last_dispatch_t, 1e-3), 0.05)
            max_step = self._v_max * dt
            n_clipped = 0
            for i, (prev, tgt) in enumerate(
                zip(self._last_arm_cmd, arm_values)
            ):
                delta = tgt - prev
                if abs(delta) > max_step:
                    arm_values[i] = prev + math.copysign(max_step, delta)
                    n_clipped += 1
            if n_clipped:
                self.last_limit_events["velocity_clamped_joints"] = n_clipped
                logger.debug(
                    "velocity clamp: %d/%d joints @ %.2f rad/s (dt=%.1fms)",
                    n_clipped, self._arm_dim, self._v_max, dt * 1000,
                )

        action = Action(self._action_space, arm_values)
        robot_result = self._robot.act(action, state=state)
        if robot_result is False:
            self.last_dispatch_succeeded = False
            self.last_limit_events["action_rejections"] = 1
            return False

        self.last_dispatch_succeeded = True
        self.last_robot_action_values = list(arm_values)
        self._last_arm_cmd = arm_values
        self._last_dispatch_t = now

        # 夹爪控制
        if (
            self._gripper is not None
            and not self._hold_gripper_closed_during_inference
            and 0 <= self._gripper_index < len(action_values)
        ):
            gripper_val = float(action_values[self._gripper_index])
            if self._gripper_mode == "width":
                self._dispatch_gripper_width(gripper_val)
            else:
                gripper_open = gripper_val >= self._gripper_threshold
                if gripper_open != self._last_gripper_open:
                    self._dispatch_gripper(gripper_open)
                    self._last_gripper_open = gripper_open
                    logger.info(
                        "[夹爪] val=%.3f → %s (%s)",
                        gripper_val, "打开" if gripper_open else "关闭",
                        self._gripper_mode,
                    )
        return True


    def _clamp_cartesian_step(
        self,
        arm_values: list[float],
        state: ArmState | None,
    ) -> list[float]:
        """Clamp absolute Cartesian targets to a per-publish EEF step."""
        if self._action_space != ActionSpace.CARTESIAN:
            return arm_values
        if (
            self._max_eef_xyz_step <= 0
            and self._max_eef_rot_step_rad <= 0
            and self._max_eef_gripper_step <= 0
        ):
            return arm_values
        if state is None or len(getattr(state, "eef_pose", []) or []) < 7:
            return arm_values

        values = list(arm_values)
        current_eef = [float(v) for v in state.eef_pose]
        current_joints = [float(v) for v in getattr(state, "joint_positions", [])]

        if len(values) == 16 and len(current_eef) >= 14:
            left_gripper = current_joints[6] if len(current_joints) > 6 else values[7]
            right_gripper = current_joints[13] if len(current_joints) > 13 else values[15]
            if self._last_arm_cmd is not None and len(self._last_arm_cmd) == 16:
                # Gripper feedback can be quantized/scaled differently from the command.
                # Use the last published command only for gripper step limiting, while
                # EEF pose limiting still uses the live robot state above.
                left_gripper = self._last_arm_cmd[7]
                right_gripper = self._last_arm_cmd[15]
            left = self._clamp_single_cartesian_target(
                current_eef[:7],
                left_gripper,
                values[:8],
            )
            right = self._clamp_single_cartesian_target(
                current_eef[7:14],
                right_gripper,
                values[8:16],
            )
            return left + right
        if len(values) == 8:
            gripper = current_joints[6] if len(current_joints) > 6 else values[7]
            if self._last_arm_cmd is not None and len(self._last_arm_cmd) == 8:
                gripper = self._last_arm_cmd[7]
            return self._clamp_single_cartesian_target(
                current_eef[:7],
                gripper,
                values,
            )
        if len(values) == 14 and len(current_eef) >= 14:
            left_pose = self._clamp_pose7(current_eef[:7], values[:7])
            right_pose = self._clamp_pose7(current_eef[7:14], values[7:14])
            return left_pose + right_pose
        if len(values) == 7:
            return self._clamp_pose7(current_eef[:7], values)
        return values

    def _clamp_single_cartesian_target(
        self,
        current_pose7: Sequence[float],
        current_gripper: float,
        target8: Sequence[float],
    ) -> list[float]:
        out = self._clamp_pose7(current_pose7, target8[:7]) + [float(target8[7])]
        if self._max_eef_gripper_step > 0:
            out[7] = _clamp_scalar_step(
                float(current_gripper),
                out[7],
                self._max_eef_gripper_step,
            )
        return out

    def _clamp_pose7(
        self,
        current_pose7: Sequence[float],
        target_pose7: Sequence[float],
    ) -> list[float]:
        cur = [float(v) for v in current_pose7]
        tgt = [float(v) for v in target_pose7]
        out = list(tgt[:7])
        if (
            self._eef_translation_deadband_m > 0
            and np.linalg.norm(np.asarray(tgt[:3]) - np.asarray(cur[:3]))
            < self._eef_translation_deadband_m
        ):
            out[:3] = cur[:3]
        if self._max_eef_xyz_step > 0:
            out[:3] = _clamp_vector_step(cur[:3], out[:3], self._max_eef_xyz_step)
        if self._max_eef_rot_step_rad > 0:
            out[3:7] = _clamp_quaternion_step(cur[3:7], tgt[3:7], self._max_eef_rot_step_rad)
        return out

    def _dispatch_gripper(self, open: bool) -> None:
        """根据 gripper_mode 执行夹爪动作。

        binary — 调 set()，由夹爪实现决定具体行为。
        """
        self._gripper.set(
            open,
            wait=self._gripper_wait_during_inference,
        )

    def _dispatch_gripper_width(self, width: float) -> None:
        """Move gripper to a target width in meters."""
        max_width = self._get_gripper_max_width()
        target = max(0.0, min(float(width), max_width))
        if (
            self._last_gripper_width_cmd is not None
            and abs(target - self._last_gripper_width_cmd)
            < self._GRIPPER_WIDTH_HYSTERESIS_M
        ):
            return
        self._gripper.move(
            target,
            wait=self._gripper_wait_during_inference,
        )
        self._last_gripper_width_cmd = target
        logger.info(
            "[夹爪] width_cmd=%.4fm (raw=%.4f, max=%.4fm)",
            target, width, max_width,
        )

    def _robot_action_values(self, action_values: list[float]) -> list[float]:
        """Extract the values that should be sent to robot.act()."""
        action_values = self._convert_embedded_gripper_action_values(action_values)
        if self._action_space == ActionSpace.CARTESIAN:
            if (
                self._gripper is not None
                and 0 <= self._gripper_index < len(action_values)
            ):
                return (
                    list(action_values[:self._gripper_index])
                    + list(action_values[self._gripper_index + 1:])
                )
            return list(action_values)
        return list(action_values[:self._arm_dim])

    def _get_gripper_max_width(self) -> float:
        """获取夹爪最大宽度，首次调用时从硬件读取并缓存。"""
        if self._gripper_max_width is None:
            try:
                state = self._gripper.observe()
                if state.max_width > 0:
                    self._gripper_max_width = state.max_width
                    logger.info("[夹爪] max_width 缓存: %.4f m", self._gripper_max_width)
            except Exception:
                logger.warning("[夹爪] 读取 max_width 失败，回退到默认值 0.08m", exc_info=True)
        return self._gripper_max_width or 0.08  # 读取失败时回退到 8cm

    def _convert_embedded_gripper_action_values(self, action_values: list[float]) -> list[float]:
        """Apply gripper_mode to action channels owned by robot.act()."""
        if self._gripper is not None or self._gripper_index >= 0:
            return action_values

        indices = _embedded_gripper_indices(action_values, self._action_space, self._arm_dim)
        if not indices:
            return action_values

        out = list(action_values)
        max_widths = self._embedded_gripper_max_widths(indices)
        for idx, max_width in zip(indices, max_widths):
            out[idx] = _action_gripper_scalar(
                out[idx],
                mode=self._gripper_mode,
                threshold=self._gripper_threshold,
                max_width=max_width,
            )
        return out

    def _embedded_gripper_max_widths(self, indices: list[int]) -> list[float]:
        try:
            params = self._robot.get_params()
            max_values = list(params.joint_position_max)
        except Exception:
            logger.debug("读取 robot params 失败，内嵌夹爪宽度回退到 0.08m", exc_info=True)
            max_values = []

        # Cartesian bimanual action layout is:
        # [left_pose7, left_gripper, right_pose7, right_gripper].
        # RobotParams still stores gripper limits in joint layout [6, 13].
        if self._action_space == ActionSpace.CARTESIAN and indices == [7, 15]:
            mapped_indices = [6, 13]
        elif self._action_space == ActionSpace.CARTESIAN and indices == [7]:
            mapped_indices = [6]
        else:
            mapped_indices = indices

        widths: list[float] = []
        for idx in mapped_indices:
            width = max_values[idx] if 0 <= idx < len(max_values) else 0.0
            if width <= 0:
                width = 0.08
            widths.append(float(width))
        return widths


def build_state_vector(
    arm_state: ArmState,
    gripper_width: float = 0.0,
    gripper_max_width: float = 1.0,
    *,
    gripper_mode: str = "binary",
    gripper_threshold: float = 0.5,
) -> list[float]:
    """构建推理需要的状态向量。

    默认格式: [q1..qN, gripper]
    binary 模式下 gripper 为 0/1，width 模式下 gripper 为真实宽度 (m)。

    Args:
        arm_state: 机器人原子状态快照
        gripper_width: 当前夹爪宽度 (m)
        gripper_max_width: 夹爪最大宽度 (m)，用于归一化

    Returns:
        状态向量 (list[float])
    """
    state = list(arm_state.joint_positions)
    state.append(
        _state_gripper_scalar(
            gripper_width,
            mode=gripper_mode,
            threshold=gripper_threshold,
            max_width=gripper_max_width,
        )
    )
    return state


def build_policy_state_vector(
    arm_state: ArmState,
    *,
    action_space: Any = ActionSpace.JOINT_POSITION,
    gripper_width: float = 0.0,
    gripper_max_width: float = 1.0,
    gripper_mode: str = "binary",
    gripper_threshold: float = 0.5,
    arm_dof: int | None = None,
    gripper_action_index: int | None = None,
    embedded_gripper_max_widths: Sequence[float] | None = None,
) -> list[float]:
    """Build the state vector using the same pose convention as the policy action.

    Joint policies receive:
      [q0..qN, gripper]

    Cartesian/eef policies receive rot6d EEF state:
      single arm: [x,y,z,R00,R10,R20,R01,R11,R21,gripper]
      bimanual:   [left_10, right_10]
    """
    if getattr(action_space, "value", action_space) != ActionSpace.CARTESIAN.value:
        return build_joint_policy_state_vector(
            arm_state,
            gripper_width=gripper_width,
            gripper_max_width=gripper_max_width,
            gripper_mode=gripper_mode,
            gripper_threshold=gripper_threshold,
            arm_dof=arm_dof,
            gripper_action_index=gripper_action_index,
            embedded_gripper_max_widths=embedded_gripper_max_widths,
        )
    return build_eef_state_vector(
        arm_state,
        gripper_width=gripper_width,
        gripper_max_width=gripper_max_width,
        gripper_mode=gripper_mode,
        gripper_threshold=gripper_threshold,
        gripper_action_index=gripper_action_index,
        embedded_gripper_max_widths=embedded_gripper_max_widths,
    )


def build_joint_policy_state_vector(
    arm_state: ArmState,
    *,
    gripper_width: float = 0.0,
    gripper_max_width: float = 1.0,
    gripper_mode: str = "binary",
    gripper_threshold: float = 0.5,
    arm_dof: int | None = None,
    gripper_action_index: int | None = None,
    embedded_gripper_max_widths: Sequence[float] | None = None,
) -> list[float]:
    """Build joint policy state with external or embedded gripper semantics."""
    joints = [float(v) for v in arm_state.joint_positions]
    gripper = _state_gripper_scalar(
        gripper_width,
        mode=gripper_mode,
        threshold=gripper_threshold,
        max_width=gripper_max_width,
    )
    if gripper_action_index is not None and gripper_action_index >= 0:
        base = joints[:arm_dof] if arm_dof is not None else joints
        idx = min(gripper_action_index, len(base))
        return base[:idx] + [gripper] + base[idx:]

    indices = _embedded_gripper_indices(
        joints,
        ActionSpace.JOINT_POSITION,
        arm_dof or len(joints),
    )
    if not indices:
        return build_state_vector(
            arm_state,
            gripper_width,
            gripper_max_width,
            gripper_mode=gripper_mode,
            gripper_threshold=gripper_threshold,
        )

    out = list(joints)
    max_widths = list(embedded_gripper_max_widths or [])
    for pos, idx in enumerate(indices):
        max_width = max_widths[pos] if pos < len(max_widths) else gripper_max_width
        out[idx] = _state_gripper_scalar(
            out[idx],
            mode=gripper_mode,
            threshold=gripper_threshold,
            max_width=max_width,
        )
    return out


def build_eef_state_vector(
    arm_state: ArmState,
    *,
    gripper_width: float = 0.0,
    gripper_max_width: float = 1.0,
    gripper_mode: str = "binary",
    gripper_threshold: float = 0.5,
    gripper_action_index: int | None = None,
    embedded_gripper_max_widths: Sequence[float] | None = None,
) -> list[float]:
    """Build Cartesian/eef policy state in rot6d format.

    `ArmState.eef_pose` is canonical pose7:
      single arm: [x,y,z,qw,qx,qy,qz]
      bimanual:   [left_pose7, right_pose7]
    """
    eef_pose = [float(v) for v in arm_state.eef_pose]
    max_widths = list(embedded_gripper_max_widths or [])
    if len(eef_pose) == 7:
        if gripper_action_index is not None and gripper_action_index < 0 and len(arm_state.joint_positions) >= 7:
            gripper_source = float(arm_state.joint_positions[6])
            max_width = max_widths[0] if max_widths else gripper_max_width
        else:
            gripper_source = gripper_width
            max_width = gripper_max_width
        gripper = _state_gripper_scalar(
            gripper_source,
            mode=gripper_mode,
            threshold=gripper_threshold,
            max_width=max_width,
        )
        return _pose7_to_rot6d_state(eef_pose, gripper)
    if len(eef_pose) == 14:
        left_width, right_width = _bimanual_gripper_state(arm_state)
        left_max = max_widths[0] if len(max_widths) > 0 else gripper_max_width
        right_max = max_widths[1] if len(max_widths) > 1 else gripper_max_width
        left_gripper = _state_gripper_scalar(
            left_width,
            mode=gripper_mode,
            threshold=gripper_threshold,
            max_width=left_max,
        )
        right_gripper = _state_gripper_scalar(
            right_width,
            mode=gripper_mode,
            threshold=gripper_threshold,
            max_width=right_max,
        )
        return (
            _pose7_to_rot6d_state(eef_pose[:7], left_gripper)
            + _pose7_to_rot6d_state(eef_pose[7:14], right_gripper)
        )
    raise ValueError(
        "cartesian/eef state requires eef_pose length 7 or 14, "
        f"got {len(eef_pose)}"
    )


def _pose7_to_rot6d_state(pose7: Sequence[float], gripper: float) -> list[float]:
    if len(pose7) != 7:
        raise ValueError(f"pose7 state requires 7 values, got {len(pose7)}")
    x, y, z = [float(v) for v in pose7[:3]]
    rot6d = quaternion_to_rotation6d(pose7[3:7])
    return [x, y, z] + rot6d + [float(gripper)]


def quaternion_to_rotation6d(quat: Sequence[float]) -> list[float]:
    """Convert [qw,qx,qy,qz] to column-major rot6d.

    The returned layout is the first two columns of the rotation matrix:
    [R00,R10,R20,R01,R11,R21].
    """
    if len(quat) != 4:
        raise ValueError(f"quaternion requires 4 values, got {len(quat)}")
    qw, qx, qy, qz = _normalize_quaternion(quat)
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r10 = 2.0 * (qx * qy + qz * qw)
    r20 = 2.0 * (qx * qz - qy * qw)
    r01 = 2.0 * (qx * qy - qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r21 = 2.0 * (qy * qz + qx * qw)
    return [r00, r10, r20, r01, r11, r21]


def _normalize_quaternion(quat: Sequence[float]) -> list[float]:
    q = [float(v) for v in quat]
    norm = math.sqrt(sum(v * v for v in q))
    if norm < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [v / norm for v in q]


def _bimanual_gripper_state(arm_state: ArmState) -> tuple[float, float]:
    joints = [float(v) for v in arm_state.joint_positions]
    if len(joints) >= 14:
        return joints[6], joints[13]
    return 0.0, 0.0


def _state_gripper_scalar(
    value: float,
    *,
    mode: str,
    threshold: float,
    max_width: float,
) -> float:
    value = float(value)
    if mode == "raw":
        return value
    if mode == "width":
        max_width = float(max_width)
        if max_width > 0:
            return max(0.0, min(value, max_width))
        return max(0.0, value)
    max_width = float(max_width)
    comparable = value / max_width if max_width > 0 else value
    return 1.0 if comparable >= float(threshold) else 0.0


def _action_gripper_scalar(
    value: float,
    *,
    mode: str,
    threshold: float,
    max_width: float,
) -> float:
    value = float(value)
    if mode == "raw":
        return value
    max_width = float(max_width) if max_width > 0 else 0.08
    if mode == "width":
        return max(0.0, min(value, max_width))
    return max_width if value >= float(threshold) else 0.0


def _embedded_gripper_indices(
    values: Sequence[float],
    action_space: ActionSpace,
    arm_dim: int,
) -> list[int]:
    n = len(values)
    if action_space == ActionSpace.CARTESIAN:
        if n == 16:
            return [7, 15]
        if n == 8:
            return [7]
        return []
    if n >= 14 and arm_dim >= 14:
        return [6, 13]
    if n >= 7 and arm_dim == 7:
        return [6]
    return []


def _clamp_scalar_step(current: float, target: float, max_step: float) -> float:
    if max_step <= 0:
        return float(target)
    delta = float(target) - float(current)
    if abs(delta) <= max_step:
        return float(target)
    return float(current) + math.copysign(max_step, delta)


def _clamp_vector_step(current: Sequence[float], target: Sequence[float], max_step: float) -> list[float]:
    cur = [float(v) for v in current]
    tgt = [float(v) for v in target]
    if max_step <= 0:
        return tgt
    delta = [t - c for c, t in zip(cur, tgt)]
    norm = math.sqrt(sum(v * v for v in delta))
    if norm <= max_step or norm < 1e-12:
        return tgt
    scale = max_step / norm
    return [c + d * scale for c, d in zip(cur, delta)]


def _normalize_quaternion(q: Sequence[float]) -> list[float]:
    vals = [float(v) for v in q]
    norm = math.sqrt(sum(v * v for v in vals))
    if norm < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [v / norm for v in vals]


def _clamp_quaternion_step(current: Sequence[float], target: Sequence[float], max_angle: float) -> list[float]:
    cur = _normalize_quaternion(current)
    tgt = _normalize_quaternion(target)
    dot = sum(c * t for c, t in zip(cur, tgt))
    if dot < 0.0:
        tgt = [-v for v in tgt]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    angle = 2.0 * math.acos(dot)
    if max_angle <= 0 or angle <= max_angle or angle < 1e-12:
        return tgt
    return _slerp_quaternion(cur, tgt, max_angle / angle)


def _slerp_quaternion(a: Sequence[float], b: Sequence[float], t: float) -> list[float]:
    qa = _normalize_quaternion(a)
    qb = _normalize_quaternion(b)
    dot = sum(x * y for x, y in zip(qa, qb))
    if dot < 0.0:
        qb = [-v for v in qb]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        out = [(1.0 - t) * x + t * y for x, y in zip(qa, qb)]
        return _normalize_quaternion(out)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    wa = math.sin((1.0 - t) * theta) / sin_theta
    wb = math.sin(t * theta) / sin_theta
    return _normalize_quaternion([wa * x + wb * y for x, y in zip(qa, qb)])
