# ARX5 Bimanual IK Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent one-sided motion after ARX5 bimanual IK joint-limit failures, recover small infeasible Cartesian steps with a shared backoff, and force fresh inference when recovery fails.

**Architecture:** `Arx5BimanualRobot` will solve both arms before sending either command. If the exact pair fails, it will retry common SE(3) progress factors; if every pair fails, it will actively hold both arms and report failure. `ActionDispatcher` will propagate that failure without advancing command tracking, while synchronous and asynchronous inference paths discard stale queued actions and replan from a fresh observation.

**Tech Stack:** Python 3.10+, NumPy, Pydantic, pytest, ARX5 SDK Python bindings.

---

### Task 1: Define and test atomic bimanual IK recovery

**Files:**
- Modify: `Robot/arx5_bimanual.py`
- Modify: `Core/config_schema.py`
- Modify: `Config/NeoVTLA/arx5_bimanual_neovtla.yaml`
- Create: `tests/test_arx5_bimanual_ik.py`

- [ ] **Step 1: Write failing tests**

Create fake solvers and a command capture around `_act_cartesian()` that assert:

```python
def test_bimanual_ik_failure_never_sends_one_sided_solution():
    result = robot._act_cartesian(target, state=state)
    assert result is False
    assert sent[-1][:6] == pytest.approx(state.joint_positions[:6])
    assert sent[-1][7:13] == pytest.approx(state.joint_positions[7:13])

def test_bimanual_ik_uses_same_backoff_for_both_arms():
    result = robot._act_cartesian(target, state=state)
    assert result is True
    assert left_solver.pose_attempts[-1][0] == pytest.approx(0.5)
    assert right_solver.pose_attempts[-1][0] == pytest.approx(0.5)
```

Also assert failed attempts log the side, status, current joints, candidate solution, and configured limits.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_arx5_bimanual_ik.py -q`

Expected: failures because `_act_cartesian()` currently sends the successful side and has no shared backoff/result.

- [ ] **Step 3: Implement minimal recovery**

Add configuration fields:

```python
cartesian_ik_atomic: bool = True
cartesian_ik_backoff_factors: list[float] = Field(
    default_factory=lambda: [0.5, 0.25, 0.125]
)
```

Add a local pose interpolation helper that linearly interpolates XYZ and uses shortest-path quaternion slerp. Solve the exact left/right pair first, then retry every configured factor with the same factor for both arms. Send a command only when both solutions succeed. If all attempts fail, command the observed joints and grippers for both arms and return `False`; otherwise return `True`. Log per-joint limit diagnostics for each failed side without clipping an invalid IK solution.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_arx5_bimanual_ik.py -q`

Expected: all tests pass.

### Task 2: Propagate robot action failure through dispatch

**Files:**
- Modify: `Robot/base.py`
- Modify: `Inference/dispatch.py`
- Modify: `tests/test_inference_action_processing.py`

- [ ] **Step 1: Write failing dispatcher tests**

```python
def test_dispatcher_returns_false_and_does_not_advance_tracking_on_robot_failure():
    robot.result = False
    assert dispatcher.dispatch(action) is False
    assert dispatcher.last_dispatch_succeeded is False
    assert dispatcher._last_arm_cmd is None

def test_dispatcher_skips_external_gripper_when_robot_action_fails():
    robot.result = False
    dispatcher.dispatch(action_with_gripper)
    assert gripper.move_calls == []
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_inference_action_processing.py -k 'robot_failure or action_fails' -q`

Expected: failures because `dispatch()` returns `None` and always advances tracking/dispatches the gripper.

- [ ] **Step 3: Implement result propagation**

Allow `BaseRobot.act()` implementations to return `bool | None`, treating only explicit `False` as failure for backward compatibility. Make `ActionDispatcher.dispatch()` return `bool`, set `last_dispatch_succeeded`, and update `_last_arm_cmd`, timestamp, external gripper, and last published action only after success.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_inference_action_processing.py -q`

Expected: all dispatcher and action-processing tests pass.

### Task 3: Invalidate stale async chunks and replan

**Files:**
- Modify: `Inference/async_worker.py`
- Modify: `Example/robot_inference.py`
- Modify: `tests/test_inference_async_and_mapping.py`

- [ ] **Step 1: Write failing invalidation test**

Block a fake `predict_chunk()` in a thread, call `invalidate_pending_actions()`, release the prediction, and assert the old result is not integrated:

```python
worker.invalidate_pending_actions()
release_prediction.set()
thread.join()
assert smoother.remaining == 0
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_inference_async_and_mapping.py -k invalidate -q`

Expected: failure because `AsyncInferenceWorker` has no invalidation generation.

- [ ] **Step 3: Implement generation-based invalidation**

Capture an invalidation generation before prediction and compare it before integrating. `invalidate_pending_actions()` increments the generation under a lock and clears the smoother. In the async control loop, call it when `dispatcher.dispatch()` returns `False`; in the synchronous loop, clear the smoother and stop consuming the current chunk. Both paths retain the control-cycle log but request/reuse only fresh observations afterward.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_inference_async_and_mapping.py -q`

Expected: all async and mapping tests pass.

### Task 4: Record and verify dispatch outcomes

**Files:**
- Modify: `Example/robot_inference.py`
- Modify: `tests/test_inference_action_processing.py`

- [ ] **Step 1: Add a failing action-trace test**

Assert a published JSONL record contains `dispatch_succeeded: false` for a rejected robot action.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_inference_action_processing.py -k dispatch_succeeded -q`

Expected: failure because action trace records currently omit execution status.

- [ ] **Step 3: Add the outcome field**

Extend `ActionTraceLogger.log_publish_step()` with `dispatch_succeeded: bool = True`, write the field to JSONL, and pass the dispatcher result at both publishing call sites.

- [ ] **Step 4: Run focused and broad verification**

Run:

```bash
python -m pytest tests/test_arx5_bimanual_ik.py tests/test_inference_action_processing.py tests/test_inference_async_and_mapping.py -q
python -m pytest -q
```

Expected: focused tests all pass. The full suite may retain only the six established baseline failures caused by missing `Example/realsense_visualize.py` and `Config/ur_example.yaml`; no new failures are allowed.
