"""Action vector conversion and EEF safety helpers."""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from Core import ActionSpace


def normalize_action_space(action_space: Any) -> ActionSpace:
    """Return a Core ActionSpace from Core/Pydantic/string enum values."""
    value = getattr(action_space, "value", action_space)
    return ActionSpace(value)


def normalize_policy_format(policy_format: Any) -> str:
    """Return policy I/O format name from Pydantic/string values."""
    value = getattr(policy_format, "value", policy_format)
    return str(value or "normal")


def format_policy_state_vector(
    state_values: Sequence[float],
    *,
    action_space: Any,
    policy_format: Any = "normal",
    canonical_dim: int = 32,
) -> list[float]:
    """Format a policy-facing state vector.

    normal: return the state unchanged.
    canonical: cartesian-only fixed-width state, padded with trailing zeros.
    """
    values = [float(v) for v in state_values]
    fmt = normalize_policy_format(policy_format)
    if fmt == "normal":
        return values
    if fmt != "canonical":
        raise ValueError(f"unsupported policy_format: {fmt}")
    if normalize_action_space(action_space) != ActionSpace.CARTESIAN:
        raise ValueError("canonical policy_format only supports cartesian action_space")

    dim = int(canonical_dim)
    if dim <= 0:
        raise ValueError(f"canonical_dim must be positive, got {canonical_dim}")
    if len(values) > dim:
        raise ValueError(
            f"canonical state length {len(values)} exceeds canonical_dim {dim}"
        )
    return values + [0.0] * (dim - len(values))


def rotation6d_to_quaternion(values: Sequence[float]) -> list[float]:
    """Convert column-major rotation-6D to [qw, qx, qy, qz].

    The input is the first two columns of a rotation matrix:
    [R00, R10, R20, R01, R11, R21].
    """
    if len(values) != 6:
        raise ValueError(f"rotation6d must contain 6 values, got {len(values)}")
    c1 = np.asarray(values[:3], dtype=np.float64)
    c2 = np.asarray(values[3:6], dtype=np.float64)
    b1 = _normalize_vector(c1, fallback=np.array([1.0, 0.0, 0.0]))
    c2_orth = c2 - float(np.dot(b1, c2)) * b1
    b2 = _normalize_vector(c2_orth, fallback=_orthogonal_unit(b1))
    b3 = np.cross(b1, b2)
    rot = np.column_stack((b1, b2, b3))
    return _matrix_to_quaternion(rot)


def canonicalize_action_values(
    action_values: Sequence[float],
    action_space: Any,
    *,
    policy_format: Any = "normal",
    canonical_dim: int = 32,
    effective_action_dim: int | None = None,
) -> list[float]:
    """Convert model action values into the robot-facing action format.

    Joint actions pass through as floats. Cartesian/eef actions accept the
    model's rotation-6D format and return pose7 quaternion actions:
    - 10D single arm: [xyz, rot6d, gripper] -> [pose7, gripper]
    - 20D bimanual: [left_10, right_10] -> [left_pose7, lg, right_pose7, rg]

    Already-canonical pose7 layouts (7/8/14/16D) are preserved with normalized
    quaternion fields.
    """
    values = [float(v) for v in action_values]
    action_space = normalize_action_space(action_space)
    values = _strip_policy_action_padding(
        values,
        action_space=action_space,
        policy_format=policy_format,
        canonical_dim=canonical_dim,
        effective_action_dim=effective_action_dim,
    )
    if action_space != ActionSpace.CARTESIAN:
        return values

    n = len(values)
    if n == 10:
        return _canonicalize_rotation6d_single(values)
    if n == 20:
        return (
            _canonicalize_rotation6d_single(values[:10])
            + _canonicalize_rotation6d_single(values[10:])
        )
    if n in (7, 8, 14, 16):
        return normalize_cartesian_quaternions(values)
    raise ValueError(
        "cartesian/eef action expects 10D single-arm, 20D bimanual, "
        f"or canonical pose7 layout, got {n} values"
    )


def canonicalize_action_chunk(
    actions: Iterable[Sequence[float]],
    action_space: Any,
    *,
    policy_format: Any = "normal",
    canonical_dim: int = 32,
    effective_action_dim: int | None = None,
) -> list[list[float]]:
    return [
        canonicalize_action_values(
            action,
            action_space,
            policy_format=policy_format,
            canonical_dim=canonical_dim,
            effective_action_dim=effective_action_dim,
        )
        for action in actions
    ]


def _strip_policy_action_padding(
    values: list[float],
    *,
    action_space: ActionSpace,
    policy_format: Any,
    canonical_dim: int,
    effective_action_dim: int | None,
) -> list[float]:
    fmt = normalize_policy_format(policy_format)
    if fmt == "normal":
        return values
    if fmt != "canonical":
        raise ValueError(f"unsupported policy_format: {fmt}")
    if action_space != ActionSpace.CARTESIAN:
        raise ValueError("canonical policy_format only supports cartesian action_space")

    dim = int(canonical_dim)
    if len(values) != dim:
        raise ValueError(
            f"canonical policy action expects {dim} values, got {len(values)}"
        )
    if effective_action_dim not in (10, 20):
        raise ValueError(
            "canonical cartesian action requires effective_action_dim 10 or 20, "
            f"got {effective_action_dim}"
        )
    return values[:effective_action_dim]


def normalize_cartesian_quaternions(values: Sequence[float]) -> list[float]:
    """Normalize quaternion fields in canonical Cartesian action layouts."""
    out = [float(v) for v in values]
    n = len(out)
    if n in (7, 8):
        out[3:7] = _normalize_quaternion(out[3:7])
    elif n == 14:
        out[3:7] = _normalize_quaternion(out[3:7])
        out[10:14] = _normalize_quaternion(out[10:14])
    elif n == 16:
        out[3:7] = _normalize_quaternion(out[3:7])
        out[11:15] = _normalize_quaternion(out[11:15])
    return out


def blend_action_values(
    old_action: Sequence[float],
    new_action: Sequence[float],
    *,
    weight_old: float,
    action_space: Any,
) -> list[float]:
    """Blend two equally-shaped actions.

    Cartesian pose quaternions use slerp while translation and scalar channels
    are linearly interpolated.
    """
    old = np.asarray(old_action, dtype=np.float64)
    new = np.asarray(new_action, dtype=np.float64)
    if old.shape != new.shape:
        raise ValueError(f"cannot blend action shapes {old.shape} and {new.shape}")
    t = 1.0 - float(weight_old)
    blended = (float(weight_old) * old + t * new).tolist()
    if normalize_action_space(action_space) == ActionSpace.CARTESIAN:
        blended = _slerp_cartesian_quaternions(blended, old.tolist(), new.tolist(), t)
    return [float(v) for v in blended]


def eef_pose_delta(current_pose: Sequence[float], target_pose: Sequence[float]) -> tuple[float, float]:
    """Return translation meters and rotation radians between two pose7 values."""
    if len(current_pose) < 7 or len(target_pose) < 7:
        raise ValueError("eef_pose_delta requires pose7 values")
    c_pos = np.asarray(current_pose[:3], dtype=np.float64)
    t_pos = np.asarray(target_pose[:3], dtype=np.float64)
    translation = float(np.linalg.norm(t_pos - c_pos))
    c_q = np.asarray(_normalize_quaternion(current_pose[3:7]), dtype=np.float64)
    t_q = np.asarray(_normalize_quaternion(target_pose[3:7]), dtype=np.float64)
    dot = abs(float(np.dot(c_q, t_q)))
    dot = max(-1.0, min(1.0, dot))
    rotation = float(2.0 * math.acos(dot))
    return translation, rotation


def max_eef_action_delta(
    current_eef_pose: Sequence[float],
    target_action: Sequence[float],
) -> tuple[float, float]:
    """Return max translation/rotation delta for single or bimanual eef actions."""
    target = normalize_cartesian_quaternions(target_action)
    current = [float(v) for v in current_eef_pose]
    if len(target) in (7, 8):
        return eef_pose_delta(current[:7], target[:7])
    if len(target) == 14:
        pairs = [(current[:7], target[:7]), (current[7:14], target[7:14])]
    elif len(target) == 16:
        pairs = [(current[:7], target[:7]), (current[7:14], target[8:15])]
    else:
        raise ValueError(f"unsupported canonical eef action length: {len(target)}")
    deltas = [eef_pose_delta(cur, tgt) for cur, tgt in pairs]
    return max(d[0] for d in deltas), max(d[1] for d in deltas)


def rebase_cartesian_chunk_to_state(
    actions: Iterable[Sequence[float]],
    state_values: Sequence[float],
) -> list[list[float]]:
    """Rigidly align an absolute EEF chunk with the measured observation.

    Used exactly once when control returns from an intervention.  The first
    policy pose becomes the observed pose; the same rigid offset is applied to
    the remainder so relative policy motion is preserved. Grippers are unchanged.
    """
    chunk = [normalize_cartesian_quaternions(row) for row in actions]
    if not chunk:
        raise ValueError("cannot rebase an empty action chunk")
    width = len(chunk[0])
    if width not in (8, 16) or any(len(row) != width for row in chunk):
        raise ValueError(f"rebase expects canonical 8D/16D actions, got {width}D")
    state = [float(value) for value in state_values]
    if len(state) in (10, 20):
        measured = canonicalize_action_values(state, ActionSpace.CARTESIAN)
    elif len(state) in (8, 16):
        measured = normalize_cartesian_quaternions(state)
    else:
        raise ValueError(f"rebase expects 10D/20D policy state, got {len(state)}D")
    if len(measured) != width:
        raise ValueError(
            f"state/action arm-count mismatch: state={len(measured)} action={width}"
        )
    if not np.all(np.isfinite(np.asarray(chunk + [measured], dtype=np.float64))):
        raise ValueError("rebase input contains NaN/Inf")

    transforms = []
    first = chunk[0]
    for offset in range(0, width, 8):
        translation = (
            np.asarray(measured[offset : offset + 3], dtype=np.float64)
            - np.asarray(first[offset : offset + 3], dtype=np.float64)
        )
        rotation = _quaternion_multiply(
            measured[offset + 3 : offset + 7],
            _quaternion_conjugate(first[offset + 3 : offset + 7]),
        )
        transforms.append((translation, rotation))

    rebased = []
    for row in chunk:
        output = list(row)
        for arm_index, offset in enumerate(range(0, width, 8)):
            translation, rotation = transforms[arm_index]
            output[offset : offset + 3] = (
                np.asarray(row[offset : offset + 3], dtype=np.float64) + translation
            ).tolist()
            output[offset + 3 : offset + 7] = _quaternion_multiply(
                rotation, row[offset + 3 : offset + 7]
            ).tolist()
        rebased.append(normalize_cartesian_quaternions(output))
    return rebased


def _canonicalize_rotation6d_single(values: Sequence[float]) -> list[float]:
    if len(values) != 10:
        raise ValueError(f"single-arm eef action expects 10 values, got {len(values)}")
    return [float(v) for v in values[:3]] + rotation6d_to_quaternion(values[3:9]) + [float(values[9])]


def _normalize_vector(vec: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return fallback.astype(np.float64)
    return vec / norm


def _orthogonal_unit(vec: np.ndarray) -> np.ndarray:
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(vec, axis))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return _normalize_vector(axis - float(np.dot(vec, axis)) * vec, fallback=np.array([0.0, 0.0, 1.0]))


def _normalize_quaternion(quat: Sequence[float]) -> list[float]:
    q = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    q = q / norm
    if q[0] < 0:
        q = -q
    return q.tolist()


def _quaternion_conjugate(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(_normalize_quaternion(quat), dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(_normalize_quaternion(left), dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(_normalize_quaternion(right), dtype=np.float64)
    result = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)
    return np.asarray(_normalize_quaternion(result), dtype=np.float64)


def _unit_quaternion_no_sign_flip(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _slerp_quaternion(q0: Sequence[float], q1: Sequence[float], t: float) -> list[float]:
    q0_arr = _unit_quaternion_no_sign_flip(q0)
    q1_arr = _unit_quaternion_no_sign_flip(q1)
    dot = float(np.dot(q0_arr, q1_arr))
    if dot < 0.0:
        q1_arr = -q1_arr
        dot = -dot
    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9995:
        q = q0_arr + float(t) * (q1_arr - q0_arr)
        return _normalize_quaternion(q)

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * float(t)
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _normalize_quaternion((s0 * q0_arr) + (s1 * q1_arr))


def _slerp_cartesian_quaternions(
    blended: list[float],
    old: Sequence[float],
    new: Sequence[float],
    t: float,
) -> list[float]:
    n = len(blended)
    if n in (7, 8):
        blended[3:7] = _slerp_quaternion(old[3:7], new[3:7], t)
    elif n == 14:
        blended[3:7] = _slerp_quaternion(old[3:7], new[3:7], t)
        blended[10:14] = _slerp_quaternion(old[10:14], new[10:14], t)
    elif n == 16:
        blended[3:7] = _slerp_quaternion(old[3:7], new[3:7], t)
        blended[11:15] = _slerp_quaternion(old[11:15], new[11:15], t)
    else:
        blended = normalize_cartesian_quaternions(blended)
    return blended


def _matrix_to_quaternion(rot: np.ndarray) -> list[float]:
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + float(rot[0, 0] - rot[1, 1] - rot[2, 2])) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + float(rot[1, 1] - rot[0, 0] - rot[2, 2])) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + float(rot[2, 2] - rot[0, 0] - rot[1, 1])) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    return _normalize_quaternion([qw, qx, qy, qz])
