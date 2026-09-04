from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from Example.arx5.auto_payload_calibration import (
    CalibrationCancelled,
    StageUI,
    _build_parser,
    _create_controller,
    _wait_for_fresh_joint_state,
    _read_valid_state,
    _validate_cli_args,
    _validate_sample_peaks,
    apply_calibration_robot_overrides,
    assemble_gravity_regressor,
    combine_mass_com,
    compute_static_residual,
    compute_torque_residual,
    evaluate_validation,
    fit_payload_parameters,
    generate_local_poses,
    promote_candidate_urdf,
    resolve_base_urdf,
    write_payload_urdf,
)


class _FeedbackState:
    def __init__(self, positions, velocities, *, timestamp=0.0):
        self._positions = np.asarray(positions, dtype=float)
        self._velocities = np.asarray(velocities, dtype=float)
        self._torques = np.zeros_like(self._positions)
        self.timestamp = timestamp
        self.gripper_pos = 0.03

    def pos(self):
        return self._positions

    def vel(self):
        return self._velocities

    def torque(self):
        return self._torques


class _FeedbackController:
    def __init__(self, timestamps, states):
        self._timestamps = iter(timestamps)
        self._states = iter(states)
        self._timestamp = 0.0

    def get_joint_state(self):
        state = next(self._states)
        self._timestamp = next(self._timestamps)
        return state

    def get_timestamp(self):
        return self._timestamp


def test_wait_for_fresh_joint_state_rejects_initial_cache_until_updates_advance():
    zero = _FeedbackState([0.0] * 6, [0.0] * 6)
    actual = _FeedbackState([0.2, 1.0, 0.8, -0.5, 0.1, 0.0], [0.0] * 6)
    controller = _FeedbackController(
        [0.0, 0.0, 1.0, 2.0, 3.0],
        [zero, zero, actual, actual, actual],
    )
    config = SimpleNamespace(
        joint_pos_min=[-3.0] * 6,
        joint_pos_max=[3.0] * 6,
        joint_torque_max=[30.0] * 6,
    )
    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    position, gripper = _wait_for_fresh_joint_state(
        controller,
        config,
        timeout_s=1.0,
        sleep_fn=lambda _: None,
        clock_fn=lambda: next(ticks),
    )

    assert position == pytest.approx(actual.pos())
    assert gripper == pytest.approx(0.03)


def test_read_valid_state_retries_when_sdk_buffer_changes_during_copy():
    class TornState(_FeedbackState):
        @property
        def timestamp(self):
            self._timestamp += 1.0
            return self._timestamp

        @timestamp.setter
        def timestamp(self, value):
            self._timestamp = value

    torn = TornState([1e100] * 6, [0.0] * 6)
    stable = _FeedbackState([0.2, 1.0, 0.8, -0.5, 0.1, 0.0], [0.0] * 6)
    controller = SimpleNamespace(
        get_joint_state=lambda: next(iter([torn, stable]))
    )
    states = iter([torn, stable])
    controller.get_joint_state = lambda: next(states)
    config = SimpleNamespace(
        joint_pos_min=[-3.0] * 6,
        joint_pos_max=[3.0] * 6,
        joint_torque_max=[30.0] * 6,
    )

    position, _, _, _ = _read_valid_state(controller, config, attempts=2)

    assert position == pytest.approx(stable.pos())


def test_compute_torque_residual_uses_measured_minus_base_model():
    measured = [[3.0, -1.0], [4.5, 2.0]]
    modeled = [[2.0, -1.5], [4.0, 1.0]]

    residual = compute_torque_residual(measured, modeled)

    assert residual == pytest.approx([1.0, 0.5, 0.5, 1.0])


def _write_base_urdf(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0"?>
<robot name="X5">
  <link name="base_link"/>
  <link name="link6">
    <inertial>
      <origin xyz="0.05 0.01 -0.02" rpy="0 0 0"/>
      <mass value="0.6"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <link name="eef_link"/>
</robot>
""",
        encoding="utf-8",
    )


def test_combine_mass_com_uses_first_moment_balance():
    mass, com = combine_mass_com(
        0.6,
        [0.05, 0.01, -0.02],
        0.2,
        [0.17, -0.01, 0.04],
    )

    assert mass == pytest.approx(0.8)
    assert com == pytest.approx([0.08, 0.005, -0.005])


def test_write_payload_urdf_updates_link6_without_modifying_source(tmp_path):
    source = tmp_path / "X5.urdf"
    output = tmp_path / "X5_payload.urdf"
    _write_base_urdf(source)
    original = source.read_text(encoding="utf-8")

    result = write_payload_urdf(source, output, 0.2, [0.17, -0.01, 0.04])

    assert result == output.resolve()
    assert source.read_text(encoding="utf-8") == original
    root = ET.parse(output).getroot()
    inertial = root.find("./link[@name='link6']/inertial")
    assert inertial is not None
    assert float(inertial.find("mass").attrib["value"]) == pytest.approx(0.8)
    xyz = [float(v) for v in inertial.find("origin").attrib["xyz"].split()]
    assert xyz == pytest.approx([0.08, 0.005, -0.005])


def test_write_payload_urdf_rejects_missing_link6_inertial(tmp_path):
    source = tmp_path / "bad.urdf"
    source.write_text("<robot><link name='link6'/></robot>", encoding="utf-8")

    with pytest.raises(ValueError, match="link6"):
        write_payload_urdf(source, tmp_path / "out.urdf", 0.2, [0.1, 0, 0])


def test_fit_payload_parameters_recovers_mass_and_com():
    rng = np.random.default_rng(7)
    regressor = rng.normal(size=(60, 4))
    expected_mass = 0.28
    expected_com = np.array([0.16, -0.025, 0.035])
    theta = np.r_[expected_mass, expected_mass * expected_com]
    residual = regressor @ theta

    estimate = fit_payload_parameters(regressor, residual, mode="mass-com")

    assert estimate.mass_kg == pytest.approx(expected_mass, abs=1e-8)
    assert estimate.com_xyz_m == pytest.approx(expected_com, abs=1e-8)
    assert estimate.rmse_after_nm < 1e-9
    assert estimate.condition_number < 10.0


def test_fit_payload_parameters_supports_mass_only():
    rng = np.random.default_rng(11)
    regressor = rng.normal(size=(40, 4))
    assumed_com = np.array([0.17, 0.0, 0.02])
    expected_mass = 0.24
    residual = expected_mass * (regressor @ np.r_[1.0, assumed_com])

    estimate = fit_payload_parameters(
        regressor,
        residual,
        mode="mass-only",
        assumed_com=assumed_com,
    )

    assert estimate.mass_kg == pytest.approx(expected_mass, abs=1e-8)
    assert estimate.com_xyz_m == pytest.approx(assumed_com)


def test_fit_payload_parameters_rejects_nonphysical_mass():
    regressor = np.eye(4)
    residual = regressor @ np.array([-0.2, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="质量"):
        fit_payload_parameters(regressor, residual, mode="mass-com")


def test_resolve_base_urdf_uses_sdk_default_when_option_is_missing(tmp_path):
    sdk_urdf = tmp_path / "sdk_X5.urdf"
    _write_base_urdf(sdk_urdf)
    robot_config = SimpleNamespace(urdf_path=str(sdk_urdf))

    assert resolve_base_urdf(robot_config, None) == sdk_urdf.resolve()


def test_resolve_base_urdf_prefers_explicit_path(tmp_path):
    sdk_urdf = tmp_path / "sdk_X5.urdf"
    explicit = tmp_path / "custom.urdf"
    _write_base_urdf(sdk_urdf)
    _write_base_urdf(explicit)
    robot_config = SimpleNamespace(urdf_path=str(sdk_urdf))

    assert resolve_base_urdf(robot_config, explicit) == explicit.resolve()


def test_stage_ui_stops_when_operator_does_not_confirm():
    output = []
    ui = StageUI(
        total_stages=8,
        assume_yes=False,
        input_fn=lambda prompt: "n",
        output_fn=output.append,
    )

    with pytest.raises(CalibrationCancelled):
        ui.confirm(1, "安全检查", "清空工作空间")

    assert any("阶段 1/8" in line for line in output)
    assert any("安全检查" in line for line in output)


def test_stage_ui_assume_yes_does_not_read_input():
    ui = StageUI(
        total_stages=8,
        assume_yes=True,
        input_fn=lambda prompt: pytest.fail("--yes 不应读取 stdin"),
        output_fn=lambda line: None,
    )

    ui.confirm(2, "控制器检查", "只读检查")


def test_generate_local_poses_stays_inside_joint_limits():
    center = np.array([0.0, 0.2, 0.25, 0.0, 0.0, 0.0])
    lower = np.array([-1.0, 0.1, 0.1, -0.2, -0.2, -0.2])
    upper = np.array([1.0, 0.3, 0.4, 0.2, 0.2, 0.2])

    poses = generate_local_poses(center, lower, upper, delta_rad=0.12)

    assert len(poses) >= 8
    assert all(np.all(pose >= lower + 0.02 - 1e-12) for pose in poses)
    assert all(np.all(pose <= upper - 0.02 + 1e-12) for pose in poses)
    assert len({tuple(np.round(pose, 8)) for pose in poses}) == len(poses)


def test_failed_validation_never_promotes_candidate(tmp_path):
    candidate = tmp_path / "payload.candidate.urdf"
    final = tmp_path / "payload.urdf"
    candidate.write_text("candidate", encoding="utf-8")
    final.write_text("existing", encoding="utf-8")
    ui = StageUI(
        total_stages=8,
        assume_yes=True,
        input_fn=lambda prompt: "y",
        output_fn=lambda line: None,
    )

    promoted = promote_candidate_urdf(
        candidate,
        final,
        validation_passed=False,
        ui=ui,
    )

    assert promoted is False
    assert final.read_text(encoding="utf-8") == "existing"
    assert candidate.read_text(encoding="utf-8") == "candidate"


def test_successful_validation_promotes_after_confirmation(tmp_path):
    candidate = tmp_path / "payload.candidate.urdf"
    final = tmp_path / "payload.urdf"
    candidate.write_text("candidate", encoding="utf-8")
    ui = StageUI(
        total_stages=8,
        assume_yes=True,
        input_fn=lambda prompt: "y",
        output_fn=lambda line: None,
    )

    promoted = promote_candidate_urdf(
        candidate,
        final,
        validation_passed=True,
        ui=ui,
    )

    assert promoted is True
    assert final.read_text(encoding="utf-8") == "candidate"
    assert not candidate.exists()


def test_assemble_gravity_regressor_uses_mass_and_first_moment_basis():
    base = np.array([[1.0, 2.0], [3.0, 4.0]])
    origin = base + np.array([[10.0, 20.0], [30.0, 40.0]])
    at_x = origin + 2.0
    at_y = origin + 3.0
    at_z = origin + 4.0

    regressor = assemble_gravity_regressor(base, origin, at_x, at_y, at_z)

    assert regressor.shape == (4, 4)
    assert regressor[:, 0] == pytest.approx([10, 20, 30, 40])
    assert regressor[:, 1] == pytest.approx([2, 2, 2, 2])
    assert regressor[:, 2] == pytest.approx([3, 3, 3, 3])
    assert regressor[:, 3] == pytest.approx([4, 4, 4, 4])


def test_compute_static_residual_uses_position_hold_error():
    commanded = np.array([[0.5, 0.3], [0.4, 0.2]])
    actual = np.array([[0.4, 0.25], [0.35, 0.1]])
    kp = np.array([80.0, 40.0])

    residual = compute_static_residual(commanded, actual, kp)

    assert residual == pytest.approx([8.0, 2.0, 4.0, 4.0])


def test_validation_requires_drift_error_and_improvement_thresholds():
    passed, metrics = evaluate_validation(
        baseline_errors=np.array([[0.02, 0.01], [0.03, 0.02]]),
        candidate_errors=np.array([[0.006, 0.004], [0.008, 0.005]]),
        candidate_drifts_m=[0.0008, 0.0012],
        max_joint_error_rad=0.01,
        max_eef_drift_m=0.002,
        required_improvement_ratio=0.2,
    )

    assert passed is True
    assert metrics["candidate_max_joint_error_rad"] == pytest.approx(0.008)
    assert metrics["candidate_max_eef_drift_m"] == pytest.approx(0.0012)


def test_validation_rejects_candidate_that_does_not_improve():
    passed, metrics = evaluate_validation(
        baseline_errors=np.array([[0.01, 0.01], [0.01, 0.01]]),
        candidate_errors=np.array([[0.009, 0.009], [0.009, 0.009]]),
        candidate_drifts_m=[0.0005, 0.0005],
        max_joint_error_rad=0.02,
        max_eef_drift_m=0.002,
        required_improvement_ratio=0.2,
    )

    assert passed is False
    assert metrics["improvement_ratio"] == pytest.approx(0.1)


def test_sample_peak_guard_rejects_transient_velocity_and_torque():
    with pytest.raises(RuntimeError, match="采样期间关节速度"):
        _validate_sample_peaks(
            max_velocity_rad_s=0.6,
            max_torque_ratio=0.2,
            velocity_limit_rad_s=0.5,
            torque_ratio_limit=0.75,
        )

    with pytest.raises(RuntimeError, match="采样期间关节力矩"):
        _validate_sample_peaks(
            max_velocity_rad_s=0.1,
            max_torque_ratio=0.8,
            velocity_limit_rad_s=0.5,
            torque_ratio_limit=0.75,
        )


def test_cli_validation_runs_before_hardware_for_invalid_timing():
    args = SimpleNamespace(
        sample_hz=0.0,
        sample_s=2.0,
        settle_s=1.0,
        move_speed_rad_s=0.1,
        pose_delta_rad=0.1,
        limit_margin_rad=0.05,
        max_tracking_error_rad=0.2,
        max_velocity_rad_s=0.5,
        max_torque_ratio=0.75,
        max_payload_mass_kg=1.0,
        max_com_abs_m=0.35,
        max_condition=1e4,
        validation_max_joint_error_rad=0.02,
        validation_max_eef_drift_m=0.002,
        required_improvement_ratio=0.2,
    )

    with pytest.raises(ValueError, match="sample-hz"):
        _validate_cli_args(args)


def test_apply_calibration_overrides_sets_negative_gripper_open_readout(tmp_path):
    urdf = tmp_path / "X5.urdf"
    urdf.write_text("<robot/>", encoding="utf-8")
    robot_config = SimpleNamespace(
        urdf_path="/sdk/models/X5.urdf",
        gripper_open_readout=5.03,
        gripper_width=0.088,
    )

    apply_calibration_robot_overrides(
        robot_config,
        urdf_path=urdf,
        gripper_open_readout=-3.43881,
        gripper_width=0.085,
    )

    assert robot_config.urdf_path == str(urdf)
    assert robot_config.gripper_open_readout == pytest.approx(-3.43881)
    assert robot_config.gripper_width == pytest.approx(0.085)


def test_create_controller_applies_gripper_open_readout_before_init(tmp_path):
    urdf = tmp_path / "X5.urdf"
    urdf.write_text("<robot/>", encoding="utf-8")
    captured = {}

    class _RobotConfig:
        def __init__(self):
            self.urdf_path = "/sdk/models/X5.urdf"
            self.joint_dof = 6
            self.gripper_open_readout = 5.03
            self.gripper_width = 0.088

    robot_config = _RobotConfig()
    controller_config = SimpleNamespace(
        gravity_compensation=False,
        background_send_recv=False,
    )

    def _controller(rc, cc, interface):
        captured["gripper_open_readout"] = rc.gripper_open_readout
        captured["gripper_width"] = rc.gripper_width
        captured["interface"] = interface
        return SimpleNamespace(set_log_level=lambda level: None)

    sdk = SimpleNamespace(
        RobotConfigFactory=SimpleNamespace(
            get_instance=lambda: SimpleNamespace(get_config=lambda model: robot_config)
        ),
        ControllerConfigFactory=SimpleNamespace(
            get_instance=lambda: SimpleNamespace(
                get_config=lambda controller_type, dof: controller_config
            )
        ),
        Arx5JointController=_controller,
        LogLevel=SimpleNamespace(INFO="INFO"),
    )

    _create_controller(
        sdk,
        model="X5",
        interface_name="can0",
        urdf_path=urdf,
        gripper_open_readout=-3.43881,
        gripper_width=0.085,
    )

    assert captured["gripper_open_readout"] == pytest.approx(-3.43881)
    assert captured["gripper_width"] == pytest.approx(0.085)
    assert captured["interface"] == "can0"


def test_parser_accepts_negative_gripper_open_readout():
    args = _build_parser().parse_args(
        [
            "--interface",
            "can0",
            "--output-urdf",
            "Config/RobotModels/arx1_left_payload.urdf",
            "--gripper-open-readout",
            "-3.43881",
            "--gripper-width",
            "0.085",
        ]
    )

    assert args.gripper_open_readout == pytest.approx(-3.43881)
    assert args.gripper_width == pytest.approx(0.085)
