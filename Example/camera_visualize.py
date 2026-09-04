"""RealSense/触觉视频检测与可视化。

用法:
    # 从配置文件加载 inference.enabled_cameras 中的图像传感器
    python Example/realsense_visualize.py Config/rizon4_example.yaml

    # 从配置文件加载全部 cameras/tactile 图像传感器
    python Example/realsense_visualize.py Config/rizon4_example.yaml --all

    # 自动检测所有相机
    python Example/realsense_visualize.py --auto

    # 不显示深度
    python Example/realsense_visualize.py Config/rizon4_example.yaml --no-depth
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Core import SensorFrame, load_yaml
from Core.logging import setup_run_logger
from Sensor.base import BaseSensor
from Sensor.manager import SensorManager

log = logging.getLogger(__name__)


def _get_realsense_camera_class():
    try:
        from Sensor.rgb_camera import RealSenseCamera
    except ImportError as e:
        raise RuntimeError(
            "RealSenseCamera 不可用，请先安装 pyrealsense2，"
            "或使用配置文件只打开 tactile/opencv 设备。"
        ) from e
    return RealSenseCamera


def print_device_info() -> None:
    RealSenseCamera = _get_realsense_camera_class()
    devices = RealSenseCamera.discover()
    if not devices:
        print("未检测到 RealSense 设备。")
        return
    print(f"检测到 {len(devices)} 台 RealSense 设备:")
    for dev in devices:
        print(f"  - {dev['name']}  serial={dev['serial_number']}  fw={dev['firmware']}")
    print()


def _configured_enabled_cameras(cfg: dict[str, Any]) -> list[str] | None:
    inference_cfg = cfg.get("inference", {}) or {}
    enabled = inference_cfg.get("enabled_cameras")
    if enabled is None:
        enabled = inference_cfg.get("enable_cameras")
    if enabled is None:
        return None
    return list(enabled)


def sensors_from_config(config_path: Path, *, all_sensors: bool = False) -> SensorManager:
    """从 YAML 配置创建图像传感器管理器。

    默认只创建 inference.enabled_cameras 白名单中的 cameras/tactile。
    all_sensors=True 时创建配置中的全部 cameras/tactile。
    """
    cfg = load_yaml(config_path)
    enabled_names = None if all_sensors else _configured_enabled_cameras(cfg)
    return SensorManager.from_config(cfg, strict=True, enabled_names=enabled_names)


def sensors_from_all_connected(enable_depth: bool = True) -> SensorManager:
    """自动检测所有已连接的 RealSense 相机。"""
    RealSenseCamera = _get_realsense_camera_class()
    devices = RealSenseCamera.discover()
    if not devices:
        raise RuntimeError("未检测到 RealSense 设备。")
    manager = SensorManager()
    for i, dev in enumerate(devices):
        manager.add(
            f"realsense_{i}",
            RealSenseCamera(
                name=f"realsense_{i}",
                serial_number=dev["serial_number"],
                enable_depth=enable_depth,
            ),
        )
    return manager


def display_streams_from_frame(
    frame: SensorFrame,
    *,
    show_depth: bool = True,
) -> list[tuple[str, Any]]:
    """提取一帧中需要显示的 OpenCV 图像。"""
    streams = frame.payload.get("streams", {})
    title = frame.sensor_name
    displays: list[tuple[str, Any]] = []

    color_stream = streams.get("color")
    if color_stream is not None and "data" in color_stream:
        displays.append((f"{title} - Color", color_stream["data"]))

    tactile_stream = streams.get("tactile")
    if tactile_stream is not None and "data" in tactile_stream:
        displays.append((f"{title} - Tactile", tactile_stream["data"]))

    depth_stream = streams.get("depth")
    if show_depth and depth_stream is not None and "data" in depth_stream:
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_stream["data"], alpha=0.03),
            cv2.COLORMAP_JET,
        )
        dist = depth_stream.get("center_distance", 0)
        cv2.putText(
            depth_colormap, f"{dist:.2f}m", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        )
        displays.append((f"{title} - Depth", depth_colormap))
    return displays


def _sensor_desc(sensor: BaseSensor) -> str:
    if hasattr(sensor, "serial_number"):
        return f"serial={getattr(sensor, 'serial_number')}"
    if hasattr(sensor, "device_path"):
        return f"device={getattr(sensor, 'device_path')}"
    return f"type={sensor.sensor_type}"


def visualize(sensors: SensorManager, show_depth: bool = True) -> None:
    """打开所有图像传感器并在 OpenCV 窗口中显示实时画面。按 'q' 或 Esc 退出。"""
    sensors.open_all()
    try:
        while True:
            for frame in sensors.read_all().values():
                for title, image in display_streams_from_frame(frame, show_depth=show_depth):
                    cv2.imshow(title, image)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        sensors.close_all()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="RealSense/触觉视频检测与可视化")
    parser.add_argument("config", nargs="?", default=None, help="YAML 配置文件路径")
    parser.add_argument("--auto", action="store_true", help="自动检测所有相机 (无需配置文件)")
    parser.add_argument("--all", action="store_true", help="使用配置文件时打开全部 cameras/tactile，忽略 enabled_cameras")
    parser.add_argument("--no-depth", action="store_true", help="不显示深度流")
    args = parser.parse_args()

    setup_run_logger(__file__, args.config)
    try:
        print_device_info()
    except RuntimeError as e:
        log.info("%s", e)

    show_depth = not args.no_depth

    if args.auto or args.config is None:
        sensors = sensors_from_all_connected(enable_depth=show_depth)
    else:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        sensors = sensors_from_config(config_path, all_sensors=args.all)

    print(f"打开 {len(sensors)} 个图像传感器...")
    for name, sensor in sensors.sensors.items():
        print(f"  - {name} ({_sensor_desc(sensor)})")
    print("按 'q' 或 Esc 退出。\n")

    visualize(sensors, show_depth=show_depth)


if __name__ == "__main__":
    main()
