#!/usr/bin/env python3
"""ARX5 末端负载自动标定。

硬件入口延迟导入 ``arx5_interface``，因此本文件中的 URDF 和拟合函数可以
在没有 ARX SDK/CAN 设备的环境中单元测试。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


_DEFAULT_CALIBRATION_CENTER = (0.0, 0.948, 0.858, -0.573, 0.0, 0.0)


@dataclass(slots=True)
class PayloadEstimate:
    """静态重力残差拟合结果。"""

    mass_kg: float
    com_xyz_m: list[float]
    rmse_before_nm: float
    rmse_after_nm: float
    condition_number: float
    sample_count: int


@dataclass(slots=True)
class PoseSample:
    """单个静态姿态的聚合观测。"""

    commanded: list[float]
    actual: list[float]
    torque_nm: list[float]
    max_abs_velocity_rad_s: float
    max_torque_ratio: float
    eef_drift_m: float


class CalibrationCancelled(RuntimeError):
    """操作员拒绝当前阶段或主动取消标定。"""


@dataclass(slots=True)
class StageUI:
    """统一输出分阶段提示，并在危险边界等待人工确认。"""

    total_stages: int
    assume_yes: bool = False
    input_fn: Callable[[str], str] = input
    output_fn: Callable[[str], Any] = print

    def confirm(self, number: int, title: str, details: str) -> None:
        self.output_fn("")
        self.output_fn(f"{'=' * 16} 阶段 {number}/{self.total_stages}: {title} {'=' * 16}")
        for line in details.splitlines():
            self.output_fn(line)
        if self.assume_yes:
            self.output_fn("[--yes] 已自动确认当前阶段。")
            return
        answer = self.input_fn("确认继续？请输入 y，其他输入将安全取消: ").strip().lower()
        if answer not in {"y", "yes"}:
            raise CalibrationCancelled(f"操作员在阶段 {number} 取消: {title}")


def generate_local_poses(
    center: Sequence[float],
    joint_min: Sequence[float],
    joint_max: Sequence[float],
    *,
    delta_rad: float = 0.12,
    limit_margin_rad: float = 0.02,
) -> list[np.ndarray]:
    """围绕人工确认的中心姿态生成小范围激励姿态，不越过软件限位。"""
    q0 = np.asarray(center, dtype=np.float64)
    lower = np.asarray(joint_min, dtype=np.float64) + float(limit_margin_rad)
    upper = np.asarray(joint_max, dtype=np.float64) - float(limit_margin_rad)
    if q0.ndim != 1 or lower.shape != q0.shape or upper.shape != q0.shape:
        raise ValueError("中心姿态与关节限位维度不一致")
    if np.any(lower >= upper):
        raise ValueError("关节限位不足以保留安全 margin")
    if np.any(q0 < lower) or np.any(q0 > upper):
        raise ValueError("当前中心姿态过于接近关节限位")

    poses = [q0.copy()]
    # J2~J6 对末端重力矩更有辨识度；J1 绕重力轴通常贡献很小。
    for joint in range(1, min(q0.size, 6)):
        for direction in (-1.0, 1.0):
            available = (
                q0[joint] - lower[joint]
                if direction < 0
                else upper[joint] - q0[joint]
            )
            step = min(float(delta_rad), float(available))
            if step < 0.02:
                continue
            pose = q0.copy()
            pose[joint] += direction * step
            poses.append(pose)
    unique: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for pose in poses:
        key = tuple(np.round(pose, 10))
        if key not in seen:
            seen.add(key)
            unique.append(pose)
    if len(unique) < 8:
        raise ValueError("可用局部姿态少于 8 个，请选择更居中的安全姿态")
    return unique


def generate_compact_poses(
    center: Sequence[float],
    joint_max: Sequence[float],
    *,
    delta_rad: float = 0.12,
    limit_margin_rad: float = 0.05,
) -> list[np.ndarray]:
    """Return center plus one positive excitation for each of J2 through J6."""
    q0 = np.asarray(center, dtype=np.float64)
    upper = np.asarray(joint_max, dtype=np.float64) - float(limit_margin_rad)
    poses = [q0.copy()]
    for joint in range(1, min(q0.size, 6)):
        step = min(float(delta_rad), float(upper[joint] - q0[joint]))
        if step < 0.02:
            raise ValueError(f"固定标定中心的 J{joint + 1} 正向空间不足")
        pose = q0.copy()
        pose[joint] += step
        poses.append(pose)
    return poses


def promote_candidate_urdf(
    candidate_path: str | Path,
    final_path: str | Path,
    *,
    validation_passed: bool,
    ui: StageUI,
) -> bool:
    """只有自动验证通过且人工确认后，才把 candidate 提升为正式 URDF。"""
    candidate = Path(candidate_path).expanduser().resolve()
    final = Path(final_path).expanduser().resolve()
    if not validation_passed:
        ui.output_fn(f"验证未通过，保留候选文件但不部署: {candidate}")
        return False
    ui.confirm(
        ui.total_stages,
        "部署确认",
        f"自动验证已通过。\n候选: {candidate}\n正式文件: {final}\n"
        "确认后将替换同名正式 URDF（如存在）。",
    )
    if not candidate.is_file():
        raise FileNotFoundError(f"候选 URDF 不存在: {candidate}")
    final.parent.mkdir(parents=True, exist_ok=True)
    candidate.replace(final)
    ui.output_fn(f"已部署标定 URDF: {final}")
    return True


def combine_mass_com(
    base_mass: float,
    base_com: Sequence[float],
    payload_mass: float,
    payload_com: Sequence[float],
) -> tuple[float, np.ndarray]:
    """按质量一阶矩合并 link6 与末端负载。"""
    m0 = float(base_mass)
    mp = float(payload_mass)
    r0 = np.asarray(base_com, dtype=np.float64)
    rp = np.asarray(payload_com, dtype=np.float64)
    if r0.shape != (3,) or rp.shape != (3,):
        raise ValueError("质心必须是 3 维 xyz")
    total = m0 + mp
    if not math.isfinite(total) or total <= 0:
        raise ValueError("合并后的 link6 质量必须大于 0")
    combined = (m0 * r0 + mp * rp) / total
    return total, combined


def _read_link6_inertial(root: ET.Element) -> tuple[ET.Element, ET.Element, ET.Element]:
    inertial = root.find("./link[@name='link6']/inertial")
    if inertial is None:
        raise ValueError("URDF 缺少 link6/inertial")
    mass = inertial.find("mass")
    origin = inertial.find("origin")
    if mass is None or "value" not in mass.attrib:
        raise ValueError("URDF link6/inertial 缺少 mass/value")
    if origin is None or "xyz" not in origin.attrib:
        raise ValueError("URDF link6/inertial 缺少 origin/xyz")
    return inertial, mass, origin


def write_payload_urdf(
    source_path: str | Path,
    output_path: str | Path,
    payload_mass: float,
    payload_com: Sequence[float],
) -> Path:
    """复制基础 URDF，并把负载质量/质心合并进 link6。"""
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"基础 URDF 不存在: {source}")
    tree = ET.parse(source)
    _, mass_node, origin_node = _read_link6_inertial(tree.getroot())
    base_mass = float(mass_node.attrib["value"])
    base_com = [float(v) for v in origin_node.attrib["xyz"].split()]
    if len(base_com) != 3:
        raise ValueError("URDF link6 origin xyz 必须包含 3 个数值")
    total_mass, combined_com = combine_mass_com(
        base_mass, base_com, payload_mass, payload_com
    )
    mass_node.set("value", f"{total_mass:.12g}")
    origin_node.set("xyz", " ".join(f"{value:.12g}" for value in combined_com))
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def resolve_base_urdf(robot_config: Any, configured_path: str | Path | None) -> Path:
    """显式路径优先；未配置时使用 pip SDK RobotConfig 的默认 URDF。"""
    raw_path = configured_path if configured_path is not None else robot_config.urdf_path
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        source = "配置" if configured_path is not None else "ARX SDK 默认"
        raise FileNotFoundError(f"{source} URDF 不存在: {path}")
    return path


def _weighted_lstsq(regressor: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    scales = np.linalg.norm(regressor, axis=0)
    if np.any(scales <= 1e-12):
        raise ValueError("重力回归矩阵缺少有效姿态激励")
    normalized = regressor / scales
    condition = float(np.linalg.cond(normalized))
    weights = np.ones(target.shape[0], dtype=np.float64)
    solution = np.zeros(regressor.shape[1], dtype=np.float64)
    for _ in range(8):
        sqrt_w = np.sqrt(weights)
        scaled_solution, *_ = np.linalg.lstsq(
            normalized * sqrt_w[:, None], target * sqrt_w, rcond=None
        )
        solution = scaled_solution / scales
        error = target - regressor @ solution
        if float(np.max(np.abs(error), initial=0.0)) < 1e-12:
            break
        median = float(np.median(error))
        sigma = 1.4826 * float(np.median(np.abs(error - median)))
        if sigma <= 1e-12:
            break
        limit = 1.345 * sigma
        weights = np.minimum(1.0, limit / np.maximum(np.abs(error), 1e-12))
    return solution, condition


def fit_payload_parameters(
    regressor: Sequence[Sequence[float]],
    residual: Sequence[float],
    *,
    mode: str = "mass-com",
    assumed_com: Sequence[float] | None = None,
    max_condition: float = 1e6,
    max_payload_mass_kg: float = 1.5,
    max_com_abs_m: float = 0.5,
) -> PayloadEstimate:
    """从静态重力残差拟合附加负载质量和 link6 坐标系质心。"""
    matrix = np.asarray(regressor, dtype=np.float64)
    target = np.asarray(residual, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError("重力回归矩阵形状必须为 (N, 4)")
    if target.shape != (matrix.shape[0],) or matrix.shape[0] < 4:
        raise ValueError("重力残差数量不足或与回归矩阵不匹配")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        raise ValueError("拟合输入包含非有限数值")

    if mode == "mass-com":
        theta, condition = _weighted_lstsq(matrix, target)
        mass = float(theta[0])
        com = np.asarray(theta[1:4], dtype=np.float64) / mass if mass else np.full(3, np.nan)
        prediction = matrix @ theta
    elif mode == "mass-only":
        if assumed_com is None:
            raise ValueError("mass-only 模式必须提供 assumed_com")
        com = np.asarray(assumed_com, dtype=np.float64)
        if com.shape != (3,):
            raise ValueError("assumed_com 必须是 3 维 xyz")
        column = matrix @ np.r_[1.0, com]
        theta_mass, condition = _weighted_lstsq(column[:, None], target)
        mass = float(theta_mass[0])
        prediction = column * mass
    else:
        raise ValueError(f"未知拟合模式: {mode}")

    if not math.isfinite(mass) or mass <= 0 or mass > max_payload_mass_kg:
        raise ValueError(
            f"拟合负载质量不在物理范围内: {mass:.6f} kg，允许 (0, {max_payload_mass_kg}]"
        )
    if not np.all(np.isfinite(com)) or np.any(np.abs(com) > max_com_abs_m):
        raise ValueError(
            f"拟合质心超出 ±{max_com_abs_m:.3f} m: {com.tolist()}"
        )
    if not math.isfinite(condition) or condition > max_condition:
        raise ValueError(
            f"标定姿态激励不足，回归矩阵条件数 {condition:.3g} > {max_condition:.3g}"
        )

    error = target - prediction
    return PayloadEstimate(
        mass_kg=mass,
        com_xyz_m=com.tolist(),
        rmse_before_nm=float(np.sqrt(np.mean(target**2))),
        rmse_after_nm=float(np.sqrt(np.mean(error**2))),
        condition_number=condition,
        sample_count=int(matrix.shape[0]),
    )


def assemble_gravity_regressor(
    base_torque: Sequence[Sequence[float]],
    unit_mass_origin_torque: Sequence[Sequence[float]],
    unit_mass_x_torque: Sequence[Sequence[float]],
    unit_mass_y_torque: Sequence[Sequence[float]],
    unit_mass_z_torque: Sequence[Sequence[float]],
) -> np.ndarray:
    """把 1kg 点质量的离线动力学结果转换为 [m,mx,my,mz] 回归矩阵。"""
    base = np.asarray(base_torque, dtype=np.float64)
    origin = np.asarray(unit_mass_origin_torque, dtype=np.float64)
    at_x = np.asarray(unit_mass_x_torque, dtype=np.float64)
    at_y = np.asarray(unit_mass_y_torque, dtype=np.float64)
    at_z = np.asarray(unit_mass_z_torque, dtype=np.float64)
    if any(array.shape != base.shape for array in (origin, at_x, at_y, at_z)):
        raise ValueError("重力力矩数组形状不一致")
    mass_column = origin - base
    columns = (
        mass_column,
        at_x - origin,
        at_y - origin,
        at_z - origin,
    )
    return np.column_stack([column.reshape(-1) for column in columns])


def compute_static_residual(
    commanded_positions: Sequence[Sequence[float]],
    actual_positions: Sequence[Sequence[float]],
    kp: Sequence[float],
) -> np.ndarray:
    """用静态 PD 位置误差估计基础 URDF 未补偿的重力力矩。"""
    commanded = np.asarray(commanded_positions, dtype=np.float64)
    actual = np.asarray(actual_positions, dtype=np.float64)
    gains = np.asarray(kp, dtype=np.float64)
    if commanded.shape != actual.shape or commanded.ndim != 2:
        raise ValueError("commanded/actual 关节数组形状不一致")
    if gains.shape != (commanded.shape[1],):
        raise ValueError("Kp 维度与关节数不一致")
    return ((commanded - actual) * gains).reshape(-1)


def compute_torque_residual(
    measured_torque: Sequence[Sequence[float]],
    base_model_torque: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return measured static torque minus the base-URDF gravity torque."""
    measured = np.asarray(measured_torque, dtype=np.float64)
    modeled = np.asarray(base_model_torque, dtype=np.float64)
    if measured.shape != modeled.shape or measured.ndim != 2:
        raise ValueError("实测/基础模型关节力矩形状不一致")
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(modeled)):
        raise ValueError("关节力矩包含非有限值")
    return (measured - modeled).reshape(-1)


def evaluate_validation(
    *,
    baseline_errors: Sequence[Sequence[float]],
    candidate_errors: Sequence[Sequence[float]],
    candidate_drifts_m: Sequence[float],
    max_joint_error_rad: float,
    max_eef_drift_m: float,
    required_improvement_ratio: float,
) -> tuple[bool, dict[str, float]]:
    """根据静态误差、EEF 漂移和相对改善率判断候选 URDF 是否可部署。"""
    baseline = np.asarray(baseline_errors, dtype=np.float64)
    candidate = np.asarray(candidate_errors, dtype=np.float64)
    drifts = np.asarray(candidate_drifts_m, dtype=np.float64)
    if baseline.ndim != 2 or candidate.ndim != 2 or drifts.ndim != 1:
        raise ValueError("验证指标数组维度不正确")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(candidate)):
        raise ValueError("验证关节误差包含非有限值")
    if not np.all(np.isfinite(drifts)) or drifts.size == 0:
        raise ValueError("验证 EEF 漂移为空或包含非有限值")
    baseline_rms = float(np.sqrt(np.mean(baseline**2)))
    candidate_rms = float(np.sqrt(np.mean(candidate**2)))
    improvement = (
        (baseline_rms - candidate_rms) / baseline_rms
        if baseline_rms > 1e-12
        else 0.0
    )
    candidate_max_error = float(np.max(np.abs(candidate), initial=0.0))
    candidate_max_drift = float(np.max(np.abs(drifts), initial=0.0))
    metrics = {
        "baseline_rms_joint_error_rad": baseline_rms,
        "candidate_rms_joint_error_rad": candidate_rms,
        "candidate_max_joint_error_rad": candidate_max_error,
        "candidate_max_eef_drift_m": candidate_max_drift,
        "improvement_ratio": improvement,
    }
    passed = bool(
        candidate_max_error <= max_joint_error_rad
        and candidate_max_drift <= max_eef_drift_m
        and improvement >= required_improvement_ratio
    )
    return passed, metrics


def _load_arx5_interface() -> Any:
    try:
        import arx5_interface as arx5
    except ImportError as exc:
        raise RuntimeError(
            "未找到 arx5-interface。请先在当前 Python 环境执行: "
            "pip install arx5-interface"
        ) from exc
    return arx5


def _format_vector(values: Sequence[float], precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def _load_pose_file(path: str | Path, dof: int) -> list[np.ndarray]:
    pose_path = Path(path).expanduser().resolve()
    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    values = payload.get("poses") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("姿态 JSON 必须是二维数组，或包含 poses 数组")
    poses = [np.asarray(pose, dtype=np.float64) for pose in values]
    if len(poses) < 4 or any(pose.shape != (dof,) for pose in poses):
        raise ValueError(f"姿态文件至少需要 4 个 {dof} 维关节位置")
    return poses


def apply_calibration_robot_overrides(
    robot_config: Any,
    *,
    urdf_path: Path,
    gripper_open_readout: float | None = None,
    gripper_width: float | None = None,
) -> None:
    """在创建控制器前覆盖 URDF 与夹爪换算参数。

    新款 X5 的 SDK 默认 ``gripper_open_readout`` 为正（例如 5.03）。现场夹爪
    电机常反向安装，必须写成负数，否则米制开口会落到 0 以下并触发急停。
    """
    robot_config.urdf_path = str(urdf_path)
    if gripper_open_readout is not None:
        robot_config.gripper_open_readout = float(gripper_open_readout)
    if gripper_width is not None:
        robot_config.gripper_width = float(gripper_width)


def _create_controller(
    sdk: Any,
    *,
    model: str,
    interface_name: str,
    urdf_path: Path,
    gripper_open_readout: float | None = None,
    gripper_width: float | None = None,
) -> tuple[Any, Any, Any]:
    robot_config = sdk.RobotConfigFactory.get_instance().get_config(model)
    apply_calibration_robot_overrides(
        robot_config,
        urdf_path=urdf_path,
        gripper_open_readout=gripper_open_readout,
        gripper_width=gripper_width,
    )
    controller_config = sdk.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    controller_config.gravity_compensation = True
    controller_config.background_send_recv = True
    controller = sdk.Arx5JointController(
        robot_config, controller_config, interface_name
    )
    controller.set_log_level(sdk.LogLevel.INFO)
    return controller, robot_config, controller_config


def _set_joint_target(
    sdk: Any,
    controller: Any,
    positions: np.ndarray,
    gripper_position: float,
) -> None:
    positions = np.asarray(positions, dtype=np.float64)
    if (
        positions.ndim != 1
        or not np.all(np.isfinite(positions))
        or np.any(np.abs(positions) > 2.0 * math.pi)
        or not math.isfinite(float(gripper_position))
    ):
        raise RuntimeError(
            f"拒绝下发异常关节目标: positions={positions.tolist()}, "
            f"gripper={gripper_position}"
        )
    command = sdk.JointState(int(positions.size))
    command.pos()[:] = positions
    command.gripper_pos = float(gripper_position)
    controller.set_joint_cmd(command)


def _read_valid_state(
    controller: Any,
    robot_config: Any,
    *,
    attempts: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Copy one internally consistent SDK state or fail without using it."""
    lower = np.asarray(robot_config.joint_pos_min, dtype=np.float64)
    upper = np.asarray(robot_config.joint_pos_max, dtype=np.float64)
    torque_limit = np.asarray(robot_config.joint_torque_max, dtype=np.float64)
    for _ in range(attempts):
        state = controller.get_joint_state()
        timestamp_before = float(state.timestamp)
        position = np.array(state.pos(), dtype=np.float64, copy=True)
        velocity = np.array(state.vel(), dtype=np.float64, copy=True)
        torque = np.array(state.torque(), dtype=np.float64, copy=True)
        gripper = float(state.gripper_pos)
        timestamp_after = float(state.timestamp)
        if timestamp_before != timestamp_after:
            continue
        if (
            position.shape == lower.shape
            and velocity.shape == lower.shape
            and torque.shape == lower.shape
            and np.all(np.isfinite(position))
            and np.all(np.isfinite(velocity))
            and np.all(np.isfinite(torque))
            and math.isfinite(gripper)
            and np.all(position >= lower)
            and np.all(position <= upper)
            and np.all(np.abs(torque) <= 1.2 * np.maximum(torque_limit, 1e-6))
        ):
            return position, velocity, torque, gripper
    raise RuntimeError("连续读取到不一致或越界的 SDK 关节反馈，已停止标定")


def _wait_for_fresh_joint_state(
    controller: Any,
    robot_config: Any,
    *,
    timeout_s: float = 5.0,
    required_updates: int = 3,
    max_velocity_rad_s: float = 0.5,
    sleep_fn: Callable[[float], Any] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
) -> tuple[np.ndarray, float]:
    """Wait for real CAN feedback before using a state as an absolute target.

    A newly constructed background controller may initially expose its
    zero-initialized state cache.  Treating that cache as the calibration
    center can command a dangerous jump toward joint zero when gains are
    enabled.  Require several advancing SDK timestamps and a stationary,
    finite, in-limit state before position hold is activated.
    """
    deadline = clock_fn() + float(timeout_s)
    previous_timestamp: float | None = None
    consecutive = 0
    latest_position: np.ndarray | None = None
    latest_gripper: float | None = None
    while clock_fn() < deadline:
        position, velocity, _, gripper = _read_valid_state(
            controller, robot_config
        )
        timestamp = float(controller.get_timestamp())
        valid = (
            float(np.max(np.abs(velocity), initial=0.0)) <= max_velocity_rad_s
        )
        advanced = previous_timestamp is not None and timestamp > previous_timestamp
        if valid and advanced:
            consecutive += 1
            # SDK state arrays may be views into a C++ receive buffer.  Copy
            # while this sample is known-valid; never return the SDK object.
            latest_position = position.copy()
            latest_gripper = gripper
            if consecutive >= required_updates:
                return latest_position, latest_gripper
        else:
            consecutive = 0
        previous_timestamp = timestamp
        sleep_fn(0.05)
    raise RuntimeError(
        "未获得连续、静止的真实 CAN 关节反馈，拒绝启用 position hold；"
        "请托住末端并检查 CAN/控制器状态"
    )


def _activate_position_hold(
    sdk: Any,
    controller: Any,
    controller_config: Any,
    robot_config: Any,
) -> tuple[np.ndarray, float, np.ndarray]:
    current, gripper_position = _wait_for_fresh_joint_state(
        controller, robot_config
    )
    _set_joint_target(sdk, controller, current, gripper_position)
    time.sleep(0.1)
    gain = sdk.Gain(
        np.asarray(controller_config.default_kp, dtype=np.float64).copy(),
        np.asarray(controller_config.default_kd, dtype=np.float64).copy(),
        float(controller_config.default_gripper_kp),
        float(controller_config.default_gripper_kd),
    )
    controller.set_gain(gain)
    active_kp = np.asarray(controller.get_gain().kp(), dtype=np.float64).copy()
    if active_kp.size != current.size or np.all(active_kp <= 1e-9):
        raise RuntimeError("关节 Kp 为 0，机械臂仍处于 damping，禁止开始标定")
    return current, gripper_position, active_kp


def _check_motion_safety(
    state: Any,
    command: np.ndarray,
    torque_limits: np.ndarray,
    *,
    max_tracking_error_rad: float,
    max_velocity_rad_s: float,
    max_torque_ratio: float,
) -> tuple[float, float]:
    actual = np.asarray(state.pos(), dtype=np.float64)
    velocity = np.asarray(state.vel(), dtype=np.float64)
    torque = np.asarray(state.torque(), dtype=np.float64)
    tracking_error = float(np.max(np.abs(command - actual), initial=0.0))
    velocity_peak = float(np.max(np.abs(velocity), initial=0.0))
    torque_ratio = float(
        np.max(np.abs(torque) / np.maximum(torque_limits, 1e-6), initial=0.0)
    )
    if tracking_error > max_tracking_error_rad:
        raise RuntimeError(
            f"跟踪误差 {tracking_error:.3f} rad 超过 {max_tracking_error_rad:.3f} rad"
        )
    if velocity_peak > max_velocity_rad_s:
        raise RuntimeError(
            f"关节速度 {velocity_peak:.3f} rad/s 超过 {max_velocity_rad_s:.3f} rad/s"
        )
    if torque_ratio > max_torque_ratio:
        raise RuntimeError(
            f"关节力矩达到限位的 {torque_ratio:.1%}，超过 {max_torque_ratio:.1%}"
        )
    return velocity_peak, torque_ratio


def _validate_sample_peaks(
    *,
    max_velocity_rad_s: float,
    max_torque_ratio: float,
    velocity_limit_rad_s: float,
    torque_ratio_limit: float,
) -> None:
    """拒绝采样窗口中发生过的瞬态超速或高力矩。"""
    if max_velocity_rad_s > velocity_limit_rad_s:
        raise RuntimeError(
            f"采样期间关节速度峰值 {max_velocity_rad_s:.3f} rad/s 超过 "
            f"{velocity_limit_rad_s:.3f} rad/s"
        )
    if max_torque_ratio > torque_ratio_limit:
        raise RuntimeError(
            f"采样期间关节力矩达到限位的 {max_torque_ratio:.1%}，超过 "
            f"{torque_ratio_limit:.1%}"
        )


def _move_to_pose(
    sdk: Any,
    controller: Any,
    controller_config: Any,
    robot_config: Any,
    start: np.ndarray,
    target: np.ndarray,
    gripper_position: float,
    torque_limits: np.ndarray,
    args: argparse.Namespace,
) -> None:
    start = np.asarray(start, dtype=np.float64).copy()
    dt = max(float(controller_config.controller_dt), 0.002)
    max_delta = float(np.max(np.abs(target - start), initial=0.0))
    steps = max(int(math.ceil(max_delta / max(args.move_speed_rad_s * dt, 1e-6))), 1)
    next_time = time.perf_counter()
    for index in range(1, steps + 1):
        alpha = index / steps
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        command = start + smooth * (target - start)
        _set_joint_target(sdk, controller, command, gripper_position)
        if index % max(int(round(0.02 / dt)), 1) == 0 or index == steps:
            actual, velocity, torque, _ = _read_valid_state(
                controller, robot_config
            )
            state = sdk.JointState(actual, velocity, torque, gripper_position)
            _check_motion_safety(
                state,
                command,
                torque_limits,
                max_tracking_error_rad=args.max_tracking_error_rad,
                max_velocity_rad_s=args.max_velocity_rad_s,
                max_torque_ratio=args.max_torque_ratio,
            )
        next_time += dt
        remaining = next_time - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)


def _sample_static_pose(
    controller: Any,
    robot_config: Any,
    target: np.ndarray,
    torque_limits: np.ndarray,
    args: argparse.Namespace,
) -> PoseSample:
    time.sleep(args.settle_s)
    actuals: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    torques: list[np.ndarray] = []
    torque_ratios: list[float] = []
    eef_positions: list[np.ndarray] = []
    period = 1.0 / args.sample_hz
    deadline = time.perf_counter() + args.sample_s
    while time.perf_counter() < deadline:
        actual, velocity, torque, _ = _read_valid_state(
            controller, robot_config
        )
        if (
            actual.shape != target.shape
            or velocity.shape != target.shape
            or torque.shape != target.shape
            or not np.all(np.isfinite(actual))
            or not np.all(np.isfinite(velocity))
            or not np.all(np.isfinite(torque))
            or float(np.max(np.abs(actual - target), initial=0.0))
            > args.max_tracking_error_rad
        ):
            raise RuntimeError(
                "静态采样收到异常关节反馈，已拒绝该数据且停止标定: "
                f"position={actual.tolist()}"
            )
        actuals.append(actual)
        velocities.append(velocity)
        torques.append(torque)
        torque_ratios.append(
            float(np.max(np.abs(torque) / np.maximum(torque_limits, 1e-6)))
        )
        eef = controller.get_eef_state()
        eef_position = np.array(eef.pose_6d()[:3], dtype=np.float64, copy=True)
        if eef_position.shape != (3,) or not np.all(np.isfinite(eef_position)):
            raise RuntimeError("静态采样收到异常 EEF 反馈")
        eef_positions.append(eef_position)
        time.sleep(period)
    actual = np.median(np.vstack(actuals), axis=0)
    max_velocity = float(np.max(np.abs(np.vstack(velocities)), initial=0.0))
    max_ratio = max(torque_ratios, default=0.0)
    drift = (
        float(np.linalg.norm(eef_positions[-1] - eef_positions[0]))
        if len(eef_positions) >= 2
        else math.inf
    )
    _validate_sample_peaks(
        max_velocity_rad_s=max_velocity,
        max_torque_ratio=max_ratio,
        velocity_limit_rad_s=args.max_velocity_rad_s,
        torque_ratio_limit=args.max_torque_ratio,
    )
    return PoseSample(
        commanded=target.tolist(),
        actual=actual.tolist(),
        torque_nm=np.median(np.vstack(torques), axis=0).tolist(),
        max_abs_velocity_rad_s=max_velocity,
        max_torque_ratio=max_ratio,
        eef_drift_m=drift,
    )


def _collect_samples(
    sdk: Any,
    controller: Any,
    controller_config: Any,
    robot_config: Any,
    initial_position: np.ndarray,
    poses: Sequence[np.ndarray],
    gripper_position: float,
    torque_limits: np.ndarray,
    args: argparse.Namespace,
    ui: StageUI,
    *,
    reverse_pass: bool,
) -> list[PoseSample]:
    center = np.asarray(poses[0], dtype=np.float64)
    excursions = [np.asarray(pose, dtype=np.float64) for pose in poses[1:]]
    if reverse_pass:
        excursions += list(reversed(excursions))
    sequence = [center]
    for pose in excursions:
        sequence.extend((pose, center))
    samples: list[PoseSample] = []
    commanded_position = np.asarray(initial_position, dtype=np.float64).copy()
    for index, pose in enumerate(sequence, start=1):
        ui.output_fn(f"姿态 {index}/{len(sequence)}: {_format_vector(pose)}")
        _move_to_pose(
            sdk,
            controller,
            controller_config,
            robot_config,
            commanded_position,
            pose,
            gripper_position,
            torque_limits,
            args,
        )
        commanded_position = np.asarray(pose, dtype=np.float64).copy()
        sample = _sample_static_pose(
            controller, robot_config, pose, torque_limits, args
        )
        samples.append(sample)
        ui.output_fn(
            "  静态 max|dq|=%.4f rad, EEF drift=%.2f mm, torque=%.1f%%"
            % (
                max(
                    abs(c - a)
                    for c, a in zip(sample.commanded, sample.actual)
                ),
                sample.eef_drift_m * 1000.0,
                sample.max_torque_ratio * 100.0,
            )
        )
    return samples


def _make_solver(sdk: Any, robot_config: Any, urdf_path: Path) -> Any:
    return sdk.Arx5Solver(
        str(urdf_path),
        int(robot_config.joint_dof),
        robot_config.joint_pos_min,
        robot_config.joint_pos_max,
        robot_config.base_link_name,
        robot_config.eef_link_name,
        robot_config.gravity_vector,
    )


def _solver_gravity_torques(solver: Any, positions: np.ndarray) -> np.ndarray:
    zero = np.zeros(positions.shape[1], dtype=np.float64)
    return np.vstack(
        [
            np.asarray(solver.inverse_dynamics(q, zero, zero), dtype=np.float64)
            for q in positions
        ]
    )


def _build_solver_regressor(
    sdk: Any,
    robot_config: Any,
    base_urdf: Path,
    positions: np.ndarray,
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="arx5-payload-basis-") as temp_dir:
        root = Path(temp_dir)
        origin_path = write_payload_urdf(base_urdf, root / "origin.urdf", 1.0, [0, 0, 0])
        x_path = write_payload_urdf(base_urdf, root / "x.urdf", 1.0, [1, 0, 0])
        y_path = write_payload_urdf(base_urdf, root / "y.urdf", 1.0, [0, 1, 0])
        z_path = write_payload_urdf(base_urdf, root / "z.urdf", 1.0, [0, 0, 1])
        base = _solver_gravity_torques(
            _make_solver(sdk, robot_config, base_urdf), positions
        )
        origin = _solver_gravity_torques(
            _make_solver(sdk, robot_config, origin_path), positions
        )
        at_x = _solver_gravity_torques(
            _make_solver(sdk, robot_config, x_path), positions
        )
        at_y = _solver_gravity_torques(
            _make_solver(sdk, robot_config, y_path), positions
        )
        at_z = _solver_gravity_torques(
            _make_solver(sdk, robot_config, z_path), positions
        )
    return assemble_gravity_regressor(base, origin, at_x, at_y, at_z)


def _sample_arrays(samples: Sequence[PoseSample]) -> tuple[np.ndarray, np.ndarray]:
    commanded = np.asarray([sample.commanded for sample in samples], dtype=np.float64)
    actual = np.asarray([sample.actual for sample in samples], dtype=np.float64)
    return commanded, actual


def _sample_torques(samples: Sequence[PoseSample]) -> np.ndarray:
    torques = np.asarray([sample.torque_nm for sample in samples], dtype=np.float64)
    if torques.ndim != 2 or not np.all(np.isfinite(torques)):
        raise ValueError("静态关节力矩样本无效")
    return torques


def _release_controller(controller: Any, ui: StageUI) -> None:
    ui.output_fn("正在让控制器进入 damping；请继续托住末端，直到下一控制器恢复 hold。")
    controller.set_to_damping()
    del controller
    gc.collect()
    time.sleep(2.2)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_calibration(args: argparse.Namespace) -> int:
    ui = StageUI(total_stages=6, assume_yes=args.yes)
    ui.confirm(
        1,
        "安全检查",
        "仅连接一只机械臂；移除工具接触和障碍物。\n"
        "操作员必须位于急停旁，并能在控制器重启期间托住末端。\n"
        "程序将慢速移动到固定安全中心，再执行 5 个单关节激励。",
    )
    sdk = _load_arx5_interface()
    robot_config = sdk.RobotConfigFactory.get_instance().get_config(args.model)
    base_urdf = resolve_base_urdf(robot_config, args.urdf_path)
    controller_config = sdk.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    ui.confirm(
        2,
        "SDK 与控制器检查",
        f"model={args.model}, interface={args.interface}\n"
        f"base URDF={base_urdf}\n"
        f"gravity={_format_vector(robot_config.gravity_vector)}\n"
        f"gripper_open_readout={robot_config.gripper_open_readout:.5f}"
        + (
            f" (将覆盖为 {args.gripper_open_readout:.5f})"
            if args.gripper_open_readout is not None
            else " (SDK 默认；反向夹爪请传 --gripper-open-readout)"
        )
        + "\n"
        f"gripper_width={robot_config.gripper_width:.4f}"
        + (
            f" (将覆盖为 {args.gripper_width:.4f})"
            if args.gripper_width is not None
            else ""
        )
        + "\n"
        f"gravity_compensation 将强制设为 True\n"
        f"default Kp={_format_vector(controller_config.default_kp, 2)}",
    )

    controller: Any | None = None
    candidate_controller: Any | None = None
    try:
        controller, active_robot_config, active_controller_config = _create_controller(
            sdk,
            model=args.model,
            interface_name=args.interface,
            urdf_path=base_urdf,
            gripper_open_readout=args.gripper_open_readout,
            gripper_width=args.gripper_width,
        )
        center, gripper_position, active_kp = _activate_position_hold(
            sdk, controller, active_controller_config, active_robot_config
        )
        limits_max = np.asarray(active_robot_config.joint_pos_max, dtype=np.float64)
        torque_limits = np.asarray(
            active_robot_config.joint_torque_max, dtype=np.float64
        )
        if args.poses:
            poses = _load_pose_file(args.poses, int(active_robot_config.joint_dof))
        else:
            poses = generate_compact_poses(
                _DEFAULT_CALIBRATION_CENTER,
                limits_max,
                delta_rad=args.pose_delta_rad,
                limit_margin_rad=args.limit_margin_rad,
            )
        output_urdf = Path(args.output_urdf).expanduser().resolve()
        candidate_urdf = output_urdf.with_name(
            f"{output_urdf.stem}.candidate{output_urdf.suffix or '.urdf'}"
        )
        report_path = (
            Path(args.report).expanduser().resolve()
            if args.report
            else output_urdf.with_suffix(".calibration.json")
        )
        pose_lines = "\n".join(
            f"  {index:02d}: {_format_vector(pose)}"
            for index, pose in enumerate(poses, start=1)
        )
        ui.confirm(
            3,
            "标定姿态路径确认",
            f"当前关节姿态: {_format_vector(center)}\n"
            f"固定标定中心: {_format_vector(_DEFAULT_CALIBRATION_CENTER)}\n"
            f"将以不超过 {args.move_speed_rad_s:.3f} rad/s 依次经过:\n{pose_lines}\n"
            "请确认所有关节插值路径均不会碰撞。",
        )
        ui.confirm(
            4,
            "基础模型数据采集",
            f"每个姿态稳定 {args.settle_s:.1f}s、采样 {args.sample_s:.1f}s，"
            "J2～J6 各激励一次，每次偏移后返回固定中心。",
        )
        baseline_samples = _collect_samples(
            sdk,
            controller,
            active_controller_config,
            active_robot_config,
            center,
            poses,
            gripper_position,
            torque_limits,
            args,
            ui,
            reverse_pass=False,
        )
        _write_report(
            report_path,
            {
                "status": "samples_collected",
                "model": args.model,
                "interface": args.interface,
                "base_urdf": str(base_urdf),
                "candidate_urdf": str(candidate_urdf),
                "baseline_samples": [asdict(sample) for sample in baseline_samples],
            },
        )
        ui.output_fn(f"原始样本已写入: {report_path}")
        commanded, actual = _sample_arrays(baseline_samples)
        measured_torque = _sample_torques(baseline_samples)
        base_torque = _solver_gravity_torques(
            _make_solver(sdk, active_robot_config, base_urdf), actual
        )
        residual = compute_torque_residual(measured_torque, base_torque)
        regressor = _build_solver_regressor(
            sdk, active_robot_config, base_urdf, actual
        )
        estimate = fit_payload_parameters(
            regressor,
            residual,
            mode=args.mode,
            assumed_com=args.assumed_com,
            max_condition=args.max_condition,
            max_payload_mass_kg=args.max_payload_mass_kg,
            max_com_abs_m=args.max_com_abs_m,
        )
        ui.confirm(
            5,
            "负载辨识结果",
            f"payload mass={estimate.mass_kg:.4f} kg\n"
            f"payload COM(link6)={_format_vector(estimate.com_xyz_m, 5)} m\n"
            f"residual RMSE: {estimate.rmse_before_nm:.4f} -> "
            f"{estimate.rmse_after_nm:.4f} N·m\n"
            f"condition number={estimate.condition_number:.2f}\n"
            "数值明显偏离实物重量/安装位置时请选择取消。",
        )

        ui.confirm(
            6,
            "生成候选 URDF",
            f"基础模型: {base_urdf}\n候选模型: {candidate_urdf}\n"
            f"报告: {report_path}\n正式文件尚不会被覆盖。",
        )
        write_payload_urdf(
            base_urdf,
            candidate_urdf,
            estimate.mass_kg,
            estimate.com_xyz_m,
        )
        report: dict[str, Any] = {
            "model": args.model,
            "interface": args.interface,
            "base_urdf": str(base_urdf),
            "candidate_urdf": str(candidate_urdf),
            "output_urdf": str(output_urdf),
            "mode": args.mode,
            "estimate": asdict(estimate),
            "baseline_samples": [asdict(sample) for sample in baseline_samples],
            "validation_passed": False,
        }
        _write_report(report_path, report)
        ui.output_fn(f"候选 URDF 已生成（未自动部署）: {candidate_urdf}")
        ui.output_fn(f"原始样本与拟合报告: {report_path}")
        return 0
    finally:
        active = candidate_controller if candidate_controller is not None else controller
        if active is not None:
            ui.output_fn(
                "标定程序即将退出，X5 控制器会进入 damping。请立即托住末端并准备断电。"
            )
            if not args.yes:
                try:
                    input("托住末端后按 Enter 退出: ")
                except EOFError:
                    pass
            try:
                active.set_to_damping()
            except Exception:
                pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="单文件 ARX5 末端负载自动标定（一次只连接一只机械臂）"
    )
    parser.add_argument("--model", default="X5")
    parser.add_argument("--interface", required=True, help="CAN 接口，如 can0")
    parser.add_argument(
        "--urdf-path",
        default=None,
        help="基础 URDF；省略时使用 pip arx5-interface 的 SDK 默认 URDF",
    )
    parser.add_argument(
        "--output-urdf",
        required=True,
        help="候选 URDF 的目标名称（实际写为 *.candidate.urdf，不自动部署）",
    )
    parser.add_argument("--report", default=None, help="JSON 标定报告路径")
    parser.add_argument("--poses", default=None, help="可选安全关节姿态 JSON")
    parser.add_argument(
        "--gripper-open-readout",
        type=float,
        default=None,
        help=(
            "覆盖 SDK gripper_open_readout。反向安装夹爪必须为负，"
            "例如现场 YAML 中的 left_gripper_open_readout=-3.43881"
        ),
    )
    parser.add_argument(
        "--gripper-width",
        type=float,
        default=None,
        help="覆盖 SDK gripper_width（m）；省略时使用工厂默认值",
    )
    parser.add_argument(
        "--mode", choices=("mass-com", "mass-only"), default="mass-com"
    )
    parser.add_argument(
        "--assumed-com",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="mass-only 模式的 link6 质心位置（m）",
    )
    parser.add_argument("--pose-delta-rad", type=float, default=0.12)
    parser.add_argument("--limit-margin-rad", type=float, default=0.05)
    parser.add_argument("--move-speed-rad-s", type=float, default=0.12)
    parser.add_argument("--settle-s", type=float, default=1.5)
    parser.add_argument("--sample-s", type=float, default=2.0)
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument("--max-tracking-error-rad", type=float, default=0.20)
    parser.add_argument("--max-velocity-rad-s", type=float, default=0.50)
    parser.add_argument("--max-torque-ratio", type=float, default=0.75)
    parser.add_argument("--max-payload-mass-kg", type=float, default=1.0)
    parser.add_argument("--max-com-abs-m", type=float, default=0.35)
    parser.add_argument("--max-condition", type=float, default=1e4)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过阶段确认（仅用于已验证的受控环境，不推荐首次使用）",
    )
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    positive = {
        "sample-hz": args.sample_hz,
        "sample-s": args.sample_s,
        "settle-s": args.settle_s,
        "move-speed-rad-s": args.move_speed_rad_s,
        "pose-delta-rad": args.pose_delta_rad,
        "limit-margin-rad": args.limit_margin_rad,
        "max-tracking-error-rad": args.max_tracking_error_rad,
        "max-velocity-rad-s": args.max_velocity_rad_s,
        "max-payload-mass-kg": args.max_payload_mass_kg,
        "max-com-abs-m": args.max_com_abs_m,
        "max-condition": args.max_condition,
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"--{name} 必须是大于 0 的有限数值")
    if not 0 < args.max_torque_ratio < 1:
        raise ValueError("--max-torque-ratio 必须位于 (0, 1)")
    gripper_open_readout = getattr(args, "gripper_open_readout", None)
    gripper_width = getattr(args, "gripper_width", None)
    if gripper_open_readout is not None and (
        not math.isfinite(float(gripper_open_readout))
        or abs(float(gripper_open_readout)) < 1e-6
    ):
        raise ValueError("--gripper-open-readout 必须是非零有限数值")
    if gripper_width is not None and (
        not math.isfinite(float(gripper_width)) or float(gripper_width) <= 0
    ):
        raise ValueError("--gripper-width 必须是大于 0 的有限数值")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _validate_cli_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.mode == "mass-only" and args.assumed_com is None:
        raise SystemExit("mass-only 模式必须提供 --assumed-com X Y Z")
    try:
        return _run_calibration(args)
    except CalibrationCancelled as exc:
        print(f"标定已取消: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在进入安全退出流程。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"标定失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
