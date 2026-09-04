# ARX5 EEF Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe, dry-run-by-default ARX5 bimanual replay tool for DataCollectionSystemV2 `actions.eef_pose/data.csv` episodes.

**Architecture:** Keep episode parsing, schema validation, timing, continuity checks, and pose interpolation as importable pure functions in one example module. The hardware path loads `BaseRobot` from YAML, approaches the first absolute EEF target smoothly, then dispatches canonical Cartesian actions frame by frame using recorded timestamps. Any rejected Cartesian action aborts replay immediately.

**Tech Stack:** Python 3.10+, standard-library CSV/argparse/time, NumPy, InferSystem `BaseRobot`, `Action`, and `Inference.action_processing`.

---

### Task 1: Parse and validate DCSv2 EEF episodes

**Files:**
- Create: `Example/arx5/replay_eef.py`
- Create: `tests/test_arx5_eef_replay.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing loader tests**

Test that `load_eef_episode()` accepts the sample-style 21-column header, returns 20D float actions and relative seconds, supports frame slicing, and rejects missing/empty/non-finite/non-monotonic data.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_arx5_eef_replay.py -q`

Expected: collection fails because `Example.arx5.replay_eef` does not exist.

- [ ] **Step 3: Implement the loader**

Use explicit column names for both arms. Prefer `actions.eef_pose/data.csv`; require `metadata.json` only for informational FPS/task metadata. Convert timestamps to seconds relative to the first selected frame and preserve each 20D action unchanged.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_arx5_eef_replay.py -q`

Expected: loader tests pass.

### Task 2: Add offline safety and smooth approach helpers

**Files:**
- Modify: `Example/arx5/replay_eef.py`
- Modify: `tests/test_arx5_eef_replay.py`

- [ ] **Step 1: Write failing safety tests**

Cover 20D rot6d canonicalization, quaternion-safe bimanual interpolation, continuity limit rejection, and recorded-gap calculation with speed scaling and an upper gap clamp.

- [ ] **Step 2: Run tests and verify RED**

Run the new individual tests and confirm they fail because the helpers are absent.

- [ ] **Step 3: Implement minimal pure helpers**

Reuse `canonicalize_action_values()` and `max_eef_action_delta()`. Interpolate XYZ/gripper linearly and quaternion fields with shortest-path slerp. Reject steps above CLI translation/rotation thresholds before any robot connection.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_arx5_eef_replay.py -q`

Expected: all parser and safety tests pass.

### Task 3: Implement dry-run and hardware replay workflow

**Files:**
- Modify: `Example/arx5/replay_eef.py`
- Modify: `tests/test_arx5_eef_replay.py`

- [ ] **Step 1: Write failing execution tests**

Use a fake robot to verify a rejected `robot.act()` aborts immediately, successful frames use `ActionSpace.CARTESIAN`, and timing never attempts to catch up after a slow frame.

- [ ] **Step 2: Run tests and verify RED**

Run the execution tests and confirm missing replay functions are the failure reason.

- [ ] **Step 3: Implement staged CLI workflow**

Default to offline dry-run; require `--execute` plus an interactive safety confirmation. On execution: connect, enable, optionally home, observe current dual pose, smooth-approach the first target, replay according to recorded timestamps divided by `--speed`, log progress, abort on IK rejection, and disconnect in `finally`. Support Space pause and `q` abort without time catch-up.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_arx5_eef_replay.py -q`

Expected: all replay tests pass without hardware.

### Task 4: Document and verify the sample episode

**Files:**
- Modify: `Example/arx5/README.md`

- [ ] **Step 1: Add dry-run and execution examples**

Document the DCSv2 episode layout, config selection, `--start/--end`, continuity limits, recorded timing, pause/quit controls, and the abort-on-IK-failure policy.

- [ ] **Step 2: Run sample dry-run**

Run the tool without `--execute` against the supplied episode and verify it reports 1283 frames, 20D actions, and valid continuity.

- [ ] **Step 3: Run focused regression and syntax checks**

Run:

```bash
python -m pytest tests/test_arx5_eef_replay.py tests/test_arx5_bimanual_ik.py tests/test_inference_action_processing.py -q
python -m py_compile Example/arx5/replay_eef.py
git diff --check
```

Expected: focused tests and static checks pass. Do not claim hardware execution was tested unless a physical ARX5 run was performed.
