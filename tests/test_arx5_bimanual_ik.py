import logging
from types import SimpleNamespace

import numpy as np
import pytest

from Core import ArmState
from Core.config_schema import Arx5ArmEndpointConfig
from Robot.arx5_bimanual import Arx5BimanualRobot


class _FakeSolver:
    def __init__(self, solve):
        self._solve = solve
        self.pose_attempts: list[list[float]] = []

    def multi_trial_ik(self, pose, current_q, trials):
        pose_list = np.asarray(pose, dtype=float).tolist()
        self.pose_attempts.append(pose_list)
        return self._solve(pose_list, np.asarray(current_q, dtype=float), trials)

    def get_ik_status_name(self, status):
        return "E_EXCEED_JOINT_LIMIT" if status == -9 else "SUCCESS"


class _FakeJointState:
    def __init__(self, dof):
        self._pos = np.zeros(dof, dtype=float)
        self.gripper_pos = 0.0

    def pos(self):
        return self._pos


class _FakeArx5Module:
    JointState = _FakeJointState


class _FakeController:
    def __init__(self):
        self.commands = []
        self.send_count = 0

    def set_joint_cmd(self, command):
        self.commands.append(command)

    def send_recv_once(self):
        self.send_count += 1


class _FakeGainController:
    def __init__(self):
        self.controller_config = SimpleNamespace(
            default_kp=np.array([80, 70, 70, 30, 30, 20], dtype=float),
            default_kd=np.array([2, 2, 2, 1, 1, 0.7], dtype=float),
        )
        self._kp = np.zeros(6, dtype=float)
        self._kd = np.zeros(6, dtype=float)
        self.gain = SimpleNamespace(
            kp=lambda: self._kp,
            kd=lambda: self._kd,
            gripper_kp=0.0,
            gripper_kd=0.0,
        )
        self.set_history = []

    def get_controller_config(self):
        return self.controller_config

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.set_history.append((gain.kp().copy(), gain.kd().copy()))

    def reset_to_home(self):
        self.gain.kp()[:] = 0.0
        self.gain.kd()[:] = 0.0


def _robot(*, factors=(0.5, 0.25)):
    return Arx5BimanualRobot(
        left_model="X5",
        left_interface="can0",
        right_model="X5",
        right_interface="can1",
        ctrl_cfg={"cartesian_ik_backoff_factors": list(factors)},
    )


def _state():
    return ArmState(
        timestamp=1.0,
        joint_positions=[0.1] * 6 + [0.03] + [0.2] * 6 + [0.04],
        eef_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        + [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    )


def _target():
    return (
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.07]
        + [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.06]
    )


def test_bimanual_ik_failure_never_sends_one_sided_solution():
    robot = _robot(factors=())
    robot._solvers = (
        _FakeSolver(lambda pose, q, trials: (0, np.full(6, 0.8))),
        _FakeSolver(lambda pose, q, trials: (-9, np.full(6, 9.0))),
    )
    sent: list[list[float]] = []
    held: list[list[float]] = []
    robot._act_joint_position = lambda values: sent.append(list(values))
    robot._hold_current_state = lambda values: held.append(list(values))
    state = _state()

    result = robot._act_cartesian(_target(), state=state)

    assert result is False
    assert sent == []
    assert held == [pytest.approx(state.joint_positions)]


def test_bimanual_ik_uses_same_backoff_for_both_arms():
    robot = _robot(factors=(0.5,))
    left = _FakeSolver(
        lambda pose, q, trials: (0, np.full(6, pose[0] + 0.1))
    )
    right = _FakeSolver(
        lambda pose, q, trials: (
            (0, np.full(6, pose[0] + 0.2))
            if pose[0] <= 1.0
            else (-9, np.full(6, 9.0))
        )
    )
    robot._solvers = (left, right)
    sent: list[list[float]] = []
    robot._act_joint_position = lambda values: sent.append(list(values))

    result = robot._act_cartesian(_target(), state=_state())

    assert result is True
    assert [attempt[0] for attempt in left.pose_attempts] == pytest.approx([1.0, 0.5])
    assert [attempt[0] for attempt in right.pose_attempts] == pytest.approx([2.0, 1.0])
    assert sent[-1][:6] == pytest.approx([0.6] * 6)
    assert sent[-1][6] == pytest.approx(0.05)
    assert sent[-1][7:13] == pytest.approx([1.2] * 6)
    assert sent[-1][13] == pytest.approx(0.05)


def test_bimanual_ik_failure_logs_joint_limit_diagnostics(caplog):
    robot = _robot(factors=())
    robot._solvers = (
        _FakeSolver(lambda pose, q, trials: (0, np.full(6, 0.5))),
        _FakeSolver(lambda pose, q, trials: (-9, np.arange(6, dtype=float))),
    )
    robot._act_joint_position = lambda values: None
    robot._hold_current_state = lambda values: None

    with caplog.at_level(logging.WARNING):
        robot._act_cartesian(_target(), state=_state())

    message = caplog.text
    assert "右臂 IK 解算失败" in message
    assert "q_current=" in message
    assert "q_candidate=" in message
    assert "q_min=" in message
    assert "q_max=" in message


def test_bimanual_hold_bypasses_gripper_action_offset_and_sign_mapping():
    robot = _robot(factors=())
    left_ctrl = _FakeController()
    right_ctrl = _FakeController()
    robot._arx5 = _FakeArx5Module()
    robot._ctrls = (left_ctrl, right_ctrl)
    robot._gripper_sdk_sign = (-1, 1)
    robot._use_background_send_recv = False
    q14 = [0.1] * 6 + [0.03] + [0.2] * 6 + [0.04]

    robot._hold_current_state(q14)

    assert left_ctrl.commands[-1].pos() == pytest.approx([0.1] * 6)
    assert left_ctrl.commands[-1].gripper_pos == pytest.approx(-0.03)
    assert right_ctrl.commands[-1].pos() == pytest.approx([0.2] * 6)
    assert right_ctrl.commands[-1].gripper_pos == pytest.approx(0.04)
    assert left_ctrl.send_count == 1
    assert right_ctrl.send_count == 1


def test_arx5_arm_endpoint_urdf_path_defaults_to_none():
    endpoint = Arx5ArmEndpointConfig()

    assert endpoint.urdf_path is None


def test_apply_joint_gains_scales_left_and_right_from_sdk_defaults():
    robot = Arx5BimanualRobot(
        left_model="X5",
        left_interface="can0",
        right_model="X5",
        right_interface="can1",
        ctrl_cfg={
            "left_joint_kp_scale": 1.1,
            "left_joint_kd_scale": 1.05,
            "right_joint_kp_scale": 0.9,
            "right_joint_kd_scale": 0.95,
        },
    )
    left = _FakeGainController()
    right = _FakeGainController()
    robot._ctrls = (left, right)

    robot._apply_joint_gains()

    assert left.gain.kp() == pytest.approx(left.controller_config.default_kp * 1.1)
    assert left.gain.kd() == pytest.approx(left.controller_config.default_kd * 1.05)
    assert right.gain.kp() == pytest.approx(right.controller_config.default_kp * 0.9)
    assert right.gain.kd() == pytest.approx(right.controller_config.default_kd * 0.95)


def test_go_home_reapplies_joint_gains_after_sdk_reset():
    robot = _robot()
    left = _FakeGainController()
    right = _FakeGainController()
    robot._ctrls = (left, right)
    robot._connected = True
    robot._apply_gripper_gains = lambda: None

    assert robot.go_home(timeout_s=1.0) is True

    assert left.gain.kp() == pytest.approx(left.controller_config.default_kp)
    assert left.gain.kd() == pytest.approx(left.controller_config.default_kd)
    assert right.gain.kp() == pytest.approx(right.controller_config.default_kp)
    assert right.gain.kd() == pytest.approx(right.controller_config.default_kd)


def test_bimanual_factory_preserves_per_arm_urdf_paths(tmp_path):
    left_urdf = tmp_path / "left.urdf"
    right_urdf = tmp_path / "right.urdf"
    left_urdf.write_text("<robot/>", encoding="utf-8")
    right_urdf.write_text("<robot/>", encoding="utf-8")

    robot = Arx5BimanualRobot._from_config_dict(
        {
            "type": "arx5_bimanual",
            "left_arm": {"urdf_path": str(left_urdf)},
            "right_arm": {"interface": "can1", "urdf_path": str(right_urdf)},
        }
    )

    assert robot._left_urdf_path == left_urdf.resolve()
    assert robot._right_urdf_path == right_urdf.resolve()


def test_init_one_without_urdf_override_keeps_sdk_default():
    robot = _robot()
    sdk_robot_cfg = SimpleNamespace(
        joint_dof=6,
        urdf_path="/sdk/models/X5.urdf",
    )
    sdk_ctrl_cfg = SimpleNamespace(
        background_send_recv=True,
        controller_dt=0.002,
        over_current_cnt_max=1000,
    )
    captured = []
    robot._arx5 = SimpleNamespace(
        RobotConfigFactory=SimpleNamespace(
            get_instance=lambda: SimpleNamespace(get_config=lambda model: sdk_robot_cfg)
        ),
        ControllerConfigFactory=SimpleNamespace(
            get_instance=lambda: SimpleNamespace(
                get_config=lambda controller_type, dof: sdk_ctrl_cfg
            )
        ),
        Arx5JointController=lambda rc, cc, interface: captured.append(rc) or object(),
    )

    robot._init_one("X5", "can0", arm_side="left", urdf_path=None)

    assert captured[0].urdf_path == "/sdk/models/X5.urdf"


def test_init_one_applies_explicit_urdf_before_controller_creation(tmp_path):
    custom_urdf = tmp_path / "payload.urdf"
    custom_urdf.write_text("<robot/>", encoding="utf-8")
    robot = _robot()
    sdk_robot_cfg = SimpleNamespace(joint_dof=6, urdf_path="/sdk/models/X5.urdf")
    sdk_ctrl_cfg = SimpleNamespace(
        background_send_recv=True,
        controller_dt=0.002,
        over_current_cnt_max=1000,
    )
    captured = []
    robot._arx5 = SimpleNamespace(
        RobotConfigFactory=SimpleNamespace(
            get_instance=lambda: SimpleNamespace(get_config=lambda model: sdk_robot_cfg)
        ),
        ControllerConfigFactory=SimpleNamespace(
            get_instance=lambda: SimpleNamespace(
                get_config=lambda controller_type, dof: sdk_ctrl_cfg
            )
        ),
        Arx5JointController=lambda rc, cc, interface: captured.append(rc) or object(),
    )

    robot._init_one(
        "X5", "can0", arm_side="left", urdf_path=custom_urdf.resolve()
    )

    assert captured[0].urdf_path == str(custom_urdf.resolve())


def test_bimanual_factory_rejects_missing_explicit_urdf(tmp_path):
    missing = tmp_path / "missing.urdf"

    with pytest.raises(FileNotFoundError, match="URDF"):
        Arx5BimanualRobot._from_config_dict(
            {
                "type": "arx5_bimanual",
                "left_arm": {"urdf_path": str(missing)},
                "right_arm": {"interface": "can1"},
            }
        )
