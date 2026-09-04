import csv
import json
from pathlib import Path

import pytest

from Core import ActionSpace, ArmState
from Example.arx5.replay_eef import (
    approach_first_action,
    canonicalize_episode_actions,
    interpolate_bimanual_action,
    load_eef_episode,
    main,
    replay_canonical_actions,
    scaled_recorded_gaps,
    validate_action_continuity,
)


ACTION_HEADER = [
    "timestamp_ms",
    *[f"left_tcp.{name}" for name in ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6")],
    "left_gripper.pos",
    *[f"right_tcp.{name}" for name in ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6")],
    "right_gripper.pos",
]


def _valid_action(offset: float = 0.0) -> list[float]:
    left = [0.30 + offset, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.08]
    right = [0.29 + offset, 0.0, 0.18, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.07]
    return left + right


def _write_episode(
    tmp_path: Path,
    *,
    timestamps_ms: list[float] | None = None,
    actions: list[list[float]] | None = None,
    header: list[str] | None = None,
) -> Path:
    episode = tmp_path / "episode_0002"
    action_dir = episode / "actions.eef_pose"
    action_dir.mkdir(parents=True)
    (episode / "metadata.json").write_text(
        json.dumps({"fps_config": 30, "task_title": "test", "robot": "arx5"}),
        encoding="utf-8",
    )
    timestamps_ms = timestamps_ms or [33.0, 67.0, 100.0]
    actions = actions or [_valid_action(i * 0.001) for i in range(len(timestamps_ms))]
    with (action_dir / "data.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header or ACTION_HEADER)
        writer.writerows([[ts, *action] for ts, action in zip(timestamps_ms, actions)])
    return episode


def test_load_eef_episode_reads_bimanual_actions_and_relative_timestamps(tmp_path):
    episode_dir = _write_episode(tmp_path)

    episode = load_eef_episode(episode_dir)

    assert len(episode.actions_rot6d) == 3
    assert episode.actions_rot6d[0] == pytest.approx(_valid_action())
    assert episode.timestamps_s == pytest.approx([0.0, 0.034, 0.067])
    assert episode.metadata["fps_config"] == 30


def test_load_eef_episode_slices_before_rebasing_timestamps(tmp_path):
    episode_dir = _write_episode(tmp_path)

    episode = load_eef_episode(episode_dir, start=1, end=3)

    assert len(episode.actions_rot6d) == 2
    assert episode.timestamps_s == pytest.approx([0.0, 0.033])
    assert episode.frame_indices == [1, 2]


def test_load_eef_episode_rejects_missing_required_column(tmp_path):
    episode_dir = _write_episode(tmp_path, header=ACTION_HEADER[:-1])

    with pytest.raises(ValueError, match="缺少列.*right_gripper.pos"):
        load_eef_episode(episode_dir)


@pytest.mark.parametrize(
    ("timestamps", "actions", "message"),
    [
        ([33.0, 20.0], [_valid_action(), _valid_action()], "严格递增"),
        ([33.0], [[*_valid_action()[:-1], float("nan")]], "非有限数值"),
    ],
)
def test_load_eef_episode_rejects_unsafe_numeric_data(
    tmp_path, timestamps, actions, message
):
    episode_dir = _write_episode(
        tmp_path, timestamps_ms=timestamps, actions=actions
    )

    with pytest.raises(ValueError, match=message):
        load_eef_episode(episode_dir)


def test_canonicalize_episode_actions_converts_both_rot6d_segments():
    canonical = canonicalize_episode_actions([_valid_action()])

    assert len(canonical[0]) == 16
    assert canonical[0][:8] == pytest.approx(
        [0.30, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0, 0.08]
    )
    assert canonical[0][8:] == pytest.approx(
        [0.29, 0.0, 0.18, 1.0, 0.0, 0.0, 0.0, 0.07]
    )


def test_interpolate_bimanual_action_uses_shortest_quaternion_path():
    start = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.02] * 2
    target = [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.08] * 2

    midpoint = interpolate_bimanual_action(start, target, 0.5)

    assert midpoint[:3] == pytest.approx([0.5, 0.0, 0.0])
    assert midpoint[3:7] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert midpoint[7] == pytest.approx(0.05)
    assert midpoint[8:11] == pytest.approx([0.5, 0.0, 0.0])
    assert midpoint[11:15] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert midpoint[15] == pytest.approx(0.05)


def test_validate_action_continuity_reports_peak_steps():
    actions = canonicalize_episode_actions(
        [_valid_action(), _valid_action(0.003), _valid_action(0.007)]
    )

    stats = validate_action_continuity(
        actions, max_translation_m=0.01, max_rotation_rad=0.2
    )

    assert stats.max_translation_m == pytest.approx(0.004)
    assert stats.max_translation_frame == 2
    assert stats.max_rotation_rad == pytest.approx(0.0)


def test_validate_action_continuity_rejects_large_step():
    actions = canonicalize_episode_actions([_valid_action(), _valid_action(0.03)])

    with pytest.raises(ValueError, match="frame 0 -> 1.*0.0300 m"):
        validate_action_continuity(
            actions, max_translation_m=0.02, max_rotation_rad=0.2
        )


def test_scaled_recorded_gaps_applies_speed_and_clamps_long_pause():
    gaps = scaled_recorded_gaps(
        [0.0, 0.033, 0.100, 1.000], speed=0.5, max_gap_s=0.2
    )

    assert gaps == pytest.approx([0.066, 0.134, 0.2])


class _FakeRobot:
    def __init__(self, results=None):
        self.results = iter(results or [])
        self.actions = []
        self.state = ArmState(
            timestamp=0.0,
            joint_positions=[0.0] * 6 + [0.02] + [0.0] * 6 + [0.03],
            eef_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] * 2,
        )

    def observe(self):
        return self.state

    def act(self, action, *, state=None):
        self.actions.append((action, state))
        return next(self.results, True)


def test_replay_canonical_actions_aborts_on_first_rejected_ik():
    robot = _FakeRobot(results=[True, False, True])
    actions = canonicalize_episode_actions(
        [_valid_action(), _valid_action(0.001), _valid_action(0.002)]
    )
    sleeps = []

    result = replay_canonical_actions(
        robot,
        actions,
        [0.0, 0.033, 0.067],
        speed=1.0,
        max_gap_s=0.2,
        sleep_fn=sleeps.append,
    )

    assert result.executed_frames == 1
    assert result.rejected_frame == 1
    assert result.completed is False
    assert len(robot.actions) == 2
    assert robot.actions[0][0].space is ActionSpace.CARTESIAN
    assert robot.actions[0][1] is robot.state
    assert sleeps == pytest.approx([0.033])


def test_replay_canonical_actions_sleeps_each_recorded_gap_without_catchup():
    robot = _FakeRobot(results=[True, True, True])
    actions = canonicalize_episode_actions(
        [_valid_action(), _valid_action(0.001), _valid_action(0.002)]
    )
    sleeps = []

    result = replay_canonical_actions(
        robot,
        actions,
        [0.0, 0.030, 0.080],
        speed=2.0,
        max_gap_s=0.2,
        sleep_fn=sleeps.append,
    )

    assert result.completed is True
    assert result.executed_frames == 3
    assert sleeps == pytest.approx([0.015, 0.025])


def test_approach_first_action_builds_start_from_observed_dual_state():
    robot = _FakeRobot(results=[True, True])
    target = canonicalize_episode_actions([_valid_action()])[0]
    sleeps = []

    completed = approach_first_action(
        robot,
        target,
        duration_s=1.0,
        rate_hz=2.0,
        sleep_fn=sleeps.append,
    )

    assert completed is True
    assert len(robot.actions) == 2
    first = robot.actions[0][0]
    assert first.space is ActionSpace.CARTESIAN
    assert first.values[7] > 0.02
    assert first.values[15] > 0.03
    assert robot.actions[-1][0].values == pytest.approx(target)
    assert sleeps == pytest.approx([0.5, 0.5])


def test_main_defaults_to_dry_run_without_loading_robot_config(tmp_path, capsys):
    episode_dir = _write_episode(tmp_path)

    exit_code = main(["/definitely/missing/config.yaml", str(episode_dir)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "DRY RUN" in output
    assert "3 帧" in output
    assert "20D rot6d -> 16D pose7" in output


def test_main_rejects_invalid_speed_during_offline_preflight(tmp_path, capsys):
    episode_dir = _write_episode(tmp_path)

    exit_code = main(
        ["/definitely/missing/config.yaml", str(episode_dir), "--speed", "0"]
    )

    assert exit_code == 2
    assert "speed 必须大于 0" in capsys.readouterr().err
