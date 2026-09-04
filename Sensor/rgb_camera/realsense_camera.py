from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)

from Core import SensorFrame
from Core.config_schema import CameraConfig
from Sensor.rgb_camera.base import BaseRGBCamera


# 统一参数名 → pyrealsense2 option enum
_RS_OPTION_MAP: dict[str, rs.option] = {
    "exposure.auto": rs.option.enable_auto_exposure,
    "exposure.value": rs.option.exposure,
    "gain": rs.option.gain,
    "white_balance.auto": rs.option.enable_auto_white_balance,
    "white_balance.value": rs.option.white_balance,
    "brightness": rs.option.brightness,
    "contrast": rs.option.contrast,
    "gamma": rs.option.gamma,
    "hue": rs.option.hue,
    "saturation": rs.option.saturation,
    "sharpness": rs.option.sharpness,
    "backlight_compensation": rs.option.backlight_compensation,
    "laser_power": rs.option.laser_power,
    "emitter_enabled": rs.option.emitter_enabled,
    "visual_preset": rs.option.visual_preset,
    "power_line_frequency": rs.option.power_line_frequency,
}

# 每个参数属于哪个硬件 sensor: "color" = RGB Camera, "depth" = Stereo Module
_OPTION_SENSOR_MAP: dict[str, str] = {
    "exposure.auto": "color",
    "exposure.value": "color",
    "gain": "color",
    "white_balance.auto": "color",
    "white_balance.value": "color",
    "brightness": "color",
    "contrast": "color",
    "gamma": "color",
    "hue": "color",
    "saturation": "color",
    "sharpness": "color",
    "backlight_compensation": "color",
    "power_line_frequency": "color",
    "laser_power": "depth",
    "emitter_enabled": "depth",
    "visual_preset": "depth",
}


@BaseRGBCamera.register("realsense")
class RealSenseCamera(BaseRGBCamera):
    """Intel RealSense RGB-D 相机驱动。

    支持 D435/D435i/D405 等 librealsense 兼容设备。

    用法::

        cam = RealSenseCamera("main_cam", serial_number="409122273675")
        with cam:
            frame = cam.read_frame()
            color = frame.payload["streams"]["color"]["data"]  # numpy BGR
    """

    def __init__(
        self,
        name: str = "realsense",
        *,
        serial_number: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
        align_depth_to_color: bool = True,
        depth_width: int | None = None,
        depth_height: int | None = None,
        depth_fps: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            width=width,
            height=height,
            fps=fps,
            params=params,
        )
        self._serial_number = serial_number
        self._enable_depth = enable_depth
        self._align_depth_to_color = align_depth_to_color
        self._depth_width = depth_width or width
        self._depth_height = depth_height or height
        self._depth_fps = depth_fps or fps

        self._pipeline: rs.pipeline | None = None
        self._align: rs.align | None = None
        self._device: rs.device | None = None
        self._color_sensor: rs.sensor | None = None
        self._depth_sensor: rs.sensor | None = None
        self._is_d405 = False

    @classmethod
    def _from_config_dict(
        cls,
        name: str,
        cfg: CameraConfig | dict[str, Any],
    ) -> RealSenseCamera:
        if isinstance(cfg, dict):
            cfg = CameraConfig.model_validate(cfg)
        color_stream = cfg.streams.get("color")
        depth_stream = cfg.streams.get("depth")
        return cls(
            name=name,
            serial_number=cfg.serial_number,
            width=color_stream.width if color_stream else 640,
            height=color_stream.height if color_stream else 480,
            fps=color_stream.fps if color_stream else 30,
            enable_depth=cfg.enable_depth,
            align_depth_to_color=cfg.align_depth_to_color,
            depth_width=depth_stream.width if depth_stream else None,
            depth_height=depth_stream.height if depth_stream else None,
            depth_fps=depth_stream.fps if depth_stream else None,
            params=cfg.params or None,
        )

    @property
    def serial_number(self) -> str | None:
        return self._serial_number

    @property
    def device_name(self) -> str | None:
        if self._device is None:
            return None
        return self._device.get_info(rs.camera_info.name)

    @property
    def enable_depth(self) -> bool:
        return self._enable_depth

    def _open_device(self) -> None:
        ctx = rs.context()
        devices = list(ctx.query_devices())
        if not devices:
            raise RuntimeError("未检测到 RealSense 设备。")

        # 按序列号查找设备
        if self._serial_number:
            self._device = None
            for dev in devices:
                if dev.get_info(rs.camera_info.serial_number) == self._serial_number:
                    self._device = dev
                    break
            if self._device is None:
                raise RuntimeError(
                    f"未找到序列号为 {self._serial_number} 的 RealSense 设备。"
                )
        else:
            self._device = devices[0]
            self._serial_number = self._device.get_info(rs.camera_info.serial_number)

        # 检测型号和 sensor
        dev_name = self._device.get_info(rs.camera_info.name)
        self._is_d405 = "D405" in dev_name

        for sensor in self._device.sensors:
            sensor_name = sensor.get_info(rs.camera_info.name)
            if sensor_name == "RGB Camera":
                self._color_sensor = sensor
            elif sensor_name == "Stereo Module":
                self._depth_sensor = sensor

        # D405: color 来自 stereo module
        if self._is_d405:
            self._color_sensor = self._depth_sensor

        # 配置 pipeline
        self._pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(self._serial_number)

        rs_config.enable_stream(
            rs.stream.color,
            self.width, self.height,
            rs.format.bgr8, self.fps,
        )

        if self._enable_depth:
            rs_config.enable_stream(
                rs.stream.depth,
                self._depth_width, self._depth_height,
                rs.format.z16, self._depth_fps,
            )

        self._pipeline.start(rs_config)

        if self._enable_depth and self._align_depth_to_color:
            self._align = rs.align(rs.stream.color)

    def _close_device(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._align = None
        self._device = None
        self._color_sensor = None
        self._depth_sensor = None

    def _get_sensor_for_option(self, key: str) -> rs.sensor | None:
        if self._is_d405:
            return self._depth_sensor
        target = _OPTION_SENSOR_MAP.get(key, "color")
        return self._color_sensor if target == "color" else self._depth_sensor

    def _apply_param_to_device(self, key: str, value: Any) -> None:
        rs_opt = _RS_OPTION_MAP.get(key)
        if rs_opt is None:
            logger.warning("相机 '%s': 未知参数 '%s'，已忽略", self.name, key)
            return
        sensor = self._get_sensor_for_option(key)
        if sensor is None:
            return
        try:
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            sensor.set_option(rs_opt, float(value))
        except RuntimeError as e:
            logger.warning("相机 '%s': 设置参数 '%s=%s' 失败: %s", self.name, key, value, e)

    def _grab_streams(self) -> dict[str, dict[str, Any]]:
        assert self._pipeline is not None

        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)

        streams: dict[str, dict[str, Any]] = {}

        color_frame = frames.get_color_frame()
        if color_frame:
            streams["color"] = {
                "data": np.asanyarray(color_frame.get_data()),
                "encoding": "bgr8",
                "width": color_frame.get_width(),
                "height": color_frame.get_height(),
            }

        depth_frame = frames.get_depth_frame()
        if depth_frame:
            w, h = depth_frame.get_width(), depth_frame.get_height()
            streams["depth"] = {
                "data": np.asanyarray(depth_frame.get_data()),
                "encoding": "z16",
                "width": w,
                "height": h,
                "unit": "meter",
                "center_distance": depth_frame.get_distance(w // 2, h // 2),
            }

        return streams

    def intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        """返回当前 color 流的 (K, D)。

        K: 3x3 相机内参矩阵 [[fx,0,cx],[0,fy,cy],[0,0,1]]
        D: 畸变系数 (Brown-Conrady)，长度 5

        相机必须先 open()，否则抛出 RuntimeError。
        """
        if self._pipeline is None:
            raise RuntimeError(f"相机 '{self.name}' 未打开，无法获取内参")
        prof = self._pipeline.get_active_profile()
        intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = np.array([
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ])
        D = np.array(intr.coeffs)
        return K, D

    @staticmethod
    def discover() -> list[dict[str, str]]:
        """列出所有已连接的 RealSense 设备。"""
        ctx = rs.context()
        return [
            {
                "name": dev.get_info(rs.camera_info.name),
                "serial_number": dev.get_info(rs.camera_info.serial_number),
                "firmware": dev.get_info(rs.camera_info.firmware_version),
            }
            for dev in ctx.query_devices()
        ]
