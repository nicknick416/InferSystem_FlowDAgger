# ARX5 Automatic Payload Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional per-arm URDF selection and a single-file, staged ARX5 payload calibration program that defaults to the pip SDK URDF, estimates payload gravity parameters, generates a candidate URDF, and validates it safely.

**Architecture:** `Arx5ArmEndpointConfig.urdf_path` is optional; the bimanual driver only overrides `RobotConfig.urdf_path` when it is present, preserving the SDK factory default otherwise. A standalone calibration program under `Example/arx5/` lazily imports `arx5_interface`, keeps XML generation and numeric fitting in pure testable helpers, and guides the operator through inspection, safe-pose approval, collection, fit, candidate generation, controller restart, and validation.

**Tech Stack:** Python 3.10, `arx5-interface`, NumPy, Python standard-library `argparse`, `xml.etree.ElementTree`, `json`, `tempfile`, pytest.

---

### Task 1: Optional per-arm URDF configuration

**Files:**
- Modify: `Core/config_schema.py`
- Modify: `Robot/arx5_bimanual.py`
- Test: `tests/test_arx5_bimanual_ik.py`

- [ ] **Step 1: Write failing schema and driver tests**

Add tests proving that `urdf_path` defaults to `None`, an omitted path leaves the SDK factory path unchanged, and an explicit existing path replaces it before `Arx5JointController` construction. Use fake SDK factories/controllers so no hardware is accessed.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest tests/test_arx5_bimanual_ik.py -q
```

Expected: failures because `Arx5ArmEndpointConfig` and `Arx5BimanualRobot` do not accept or apply per-arm URDF paths.

- [ ] **Step 3: Implement optional path handling**

Add `urdf_path: str | None = None` to `Arx5ArmEndpointConfig`. Pass left/right values through `_from_config_dict()` and the robot constructor. Resolve explicit paths with `expanduser()`; repository-relative paths resolve from the repository root. Raise `FileNotFoundError` before touching hardware for a configured missing file. Do not assign `robot_cfg.urdf_path` at all when the option is absent.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 1 command and expect all tests in the file to pass.

### Task 2: Pure payload model and fitting helpers

**Files:**
- Create: `Example/arx5/auto_payload_calibration.py`
- Create: `tests/test_arx5_payload_calibration.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing helper tests**

Cover these public helpers:

```python
combine_mass_com(base_mass, base_com, payload_mass, payload_com)
write_payload_urdf(source_path, output_path, payload_mass, payload_com)
fit_payload_parameters(regressor, residual, mode, assumed_com)
resolve_base_urdf(robot_config, configured_path)
```

Tests must verify mass/COM combination, source-file preservation, XML update of `link6`, recovery of synthetic `[m, mx, my, mz]`, mass-only fitting, invalid estimate rejection, and default SDK URDF fallback.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest tests/test_arx5_payload_calibration.py -q
```

Expected: import failure because the calibration module does not exist.

- [ ] **Step 3: Implement minimal pure helpers**

Use `ElementTree` to copy the base URDF and modify only `link6/inertial/mass` and `origin`. Parameterize the gravity regressor by `[m, m*x, m*y, m*z]`. Use scaled NumPy least squares with a short Huber iterative reweighting loop, report condition number and RMS residuals, and reject non-finite, non-positive, out-of-bounds, or ill-conditioned estimates.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 2 command and expect all helper tests to pass.

### Task 3: Single-file staged hardware workflow

**Files:**
- Modify: `Example/arx5/auto_payload_calibration.py`
- Modify: `tests/test_arx5_payload_calibration.py`

- [ ] **Step 1: Write failing workflow tests**

Use injected input/output callables and fake controllers to prove that stage prompts stop on a negative response, `--yes` bypasses prompts, generated local poses remain inside joint limits, and validation never promotes a candidate URDF when acceptance thresholds fail.

- [ ] **Step 2: Run tests and verify RED**

Run the Task 2 test command and expect failures for missing workflow classes/functions.

- [ ] **Step 3: Implement staged calibration workflow**

The single program must print and confirm these stages:

1. Safety and SDK/URDF inspection.
2. Connect one arm and enable position hold at its current pose.
3. Print locally generated or JSON-provided safe poses and request path approval.
4. Move slowly, sample each pose in forward and reverse order, abort on tracking/velocity/torque limits.
5. Build gravity basis solvers from temporary URDFs and fit payload mass/COM.
6. Print estimate and diagnostics, then write a `.candidate.urdf` plus JSON report after confirmation.
7. Require physical support while the old controller is released and recreated with the candidate URDF.
8. Re-run validation poses, compare static error and EEF drift, and promote the candidate to the requested output only after automatic acceptance and final confirmation.

Keep `arx5_interface` as a lazy import so unit tests work without the SDK. Use only stdlib plus NumPy. Default `--urdf-path` to the SDK factory path. Never discover unknown workspace poses automatically; generate small perturbations around the operator-approved current pose or load a supplied JSON list.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 2 command and expect all workflow tests to pass.

### Task 4: Configuration example and verification

**Files:**
- Modify: `Config/NeoVTLA/arx5_bimanual_neovtla.yaml`
- Modify: `Config/README.md`

- [ ] **Step 1: Add commented configuration examples**

Document that omitted `urdf_path` uses the pip SDK model. Add commented `urdf_path` examples under both arms without changing current runtime behavior. Document the calibration command, candidate/report outputs, restart boundary, and rollback by removing the path.

- [ ] **Step 2: Run focused verification**

Run:

```bash
python -m pytest tests/test_arx5_payload_calibration.py tests/test_arx5_bimanual_ik.py tests/test_inference_action_processing.py tests/test_inference_async_and_mapping.py -q
python -m py_compile Example/arx5/auto_payload_calibration.py Robot/arx5_bimanual.py Core/config_schema.py
git diff --check
```

Expected: focused tests pass, compilation exits zero, and `git diff --check` produces no output.

- [ ] **Step 3: Review safety and scope**

Confirm line by line that default URDF behavior is unchanged, no hardware command is issued before explicit confirmation, the program controls only one interface at a time, candidate activation requires physical-support confirmation, failed validation cannot overwrite the final URDF, and existing log/config files remain untouched.
