"""Hardware-free deployment preflight for the ARX FlowDAgger control loop."""
from __future__ import annotations

import argparse
import json

from Core import ActionSpace, load_config
from Inference.client import InferenceClient


CAMERA_KEYS = {
    "observation/image",
    "observation/left_wrist_image",
    "observation/right_wrist_image",
}
PROTOCOL_VERSION = 3
BASE_MODEL_ID = "connect_elevator_pins_arx_0901:20000:648ff0462d1cec61"


def validate_local_config(config, stage: str) -> list[str]:
    errors: list[str] = []
    inference = config.inference
    if inference is None:
        return ["missing inference configuration"]
    flow = inference.flowdagger
    if not flow.enabled:
        errors.append("flowdagger.enabled must be true")
    if inference.action_space.value != ActionSpace.CARTESIAN.value:
        errors.append("action_space must be cartesian")
    if inference.arm_dof != 20:
        errors.append(f"arm_dim/arm_dof must be 20, got {inference.arm_dof}")
    if inference.fps != 30 or flow.expert_fps != 30:
        errors.append("policy and expert rates must both be 30Hz")
    if not inference.async_inference.enabled:
        errors.append("async inference must be enabled")
    enabled = list(inference.enabled_cameras or [])
    if len(enabled) != 3 or len(set(enabled)) != 3:
        errors.append("exactly three distinct cameras must be enabled")
    mapped = set()
    for name in enabled:
        camera = config.cameras.get(name)
        if camera is None:
            errors.append(f"enabled camera {name!r} is missing")
            continue
        mapped.add(camera.mapped_key)
        color = camera.streams.get("color")
        if color is None or (color.width, color.height, color.fps) != (640, 480, 30):
            errors.append(f"camera {name!r} must provide 640x480 RGB at 30Hz")
    if mapped != CAMERA_KEYS:
        errors.append(f"camera mapped keys mismatch: {sorted(str(x) for x in mapped)}")
    return errors


def validate_server(health: dict, stage: str) -> list[str]:
    errors: list[str] = []
    expected_mode = (
        "base_record" if stage in ("demonstration", "baseline") else "flowdagger"
    )
    mode = health.get("runtime_mode")
    if mode != expected_mode:
        errors.append(f"stage {stage} requires server mode {expected_mode}, got {mode}")
    if health.get("protocol_version") != PROTOCOL_VERSION:
        errors.append(
            f"server protocol_version must be {PROTOCOL_VERSION}, "
            f"got {health.get('protocol_version')}"
        )
    if health.get("base_model_id") != BASE_MODEL_ID:
        errors.append(
            f"server base_model_id must be {BASE_MODEL_ID}, "
            f"got {health.get('base_model_id')}"
        )
    if not health.get("server_session_id"):
        errors.append("server_session_id is missing")
    for key, expected in (("action_horizon", 50), ("action_dim", 20), ("state_dim", 20)):
        if health.get(key) != expected:
            errors.append(f"server {key} must be {expected}, got {health.get(key)}")
    if set(health.get("camera_keys", [])) != CAMERA_KEYS:
        errors.append("server camera keys do not match ARX configuration")
    if health.get("active_episode_id") is not None:
        errors.append(f"server already has active episode {health['active_episode_id']}")
    if health.get("training", {}).get("state") == "running":
        errors.append("server is still training")
    if stage in ("shadow", "closed_loop") and int(health.get("policy_version", 0)) <= 0:
        errors.append(f"stage {stage} requires a trained steering checkpoint")
    if stage in ("shadow", "closed_loop") and not bool(
        health.get("steering_eligible", False)
    ):
        errors.append(f"stage {stage} requires an eligible steering checkpoint")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--stage", required=True,
        choices=("demonstration", "baseline", "bootstrap", "shadow", "closed_loop"),
    )
    parser.add_argument(
        "--server",
        help="health-check address override for disconnected protocol testing",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    errors = validate_local_config(config, args.stage)
    inference = config.inference
    server = args.server or inference.server
    try:
        with InferenceClient(
            server,
            recv_timeout_ms=min(inference.recv_timeout_ms, 5000),
            send_timeout_ms=3000,
            max_retries=1,
        ) as client:
            health = client.flowdagger_health()
        errors.extend(validate_server(health, args.stage))
    except Exception as exc:
        errors.append(f"server health check failed: {exc}")
        health = {}
    report = {"stage": args.stage, "server": server, "health": health, "errors": errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
