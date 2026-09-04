from __future__ import annotations

import logging
from typing import Any

from Core import SensorFrame
from Core.config_schema import SystemConfig
from Sensor.base import BaseSensor
from Sensor.rgb_camera.base import BaseRGBCamera
from Sensor.tactile.base import BaseTactileSensor

logger = logging.getLogger(__name__)


class SensorManager:
    """统一传感器管理器。

    从 YAML 配置一次创建并管理所有传感器 (相机 + 触觉)，
    提供统一的生命周期管理和数据读取接口。

    用法::

        # 从配置创建 (strict=True 默认，硬件失败会抛异常)
        config = load_yaml("config.yaml")
        with SensorManager.from_config(config) as sensors:
            # 读取所有传感器数据
            all_frames = sensors.read_all()

            # 只获取图像 (推理用)
            images = sensors.read_images()

            # 只获取指定传感器
            images = sensors.read_images(["main_cam", "wrist_left"])

        # 手动创建
        manager = SensorManager()
        manager.add("main_cam", camera_instance)
        manager.add("wrist_left", tactile_instance)
    """

    def __init__(self, *, strict: bool = True) -> None:
        self._sensors: dict[str, BaseSensor] = {}
        self._strict = strict

    # ── 工厂 ──

    @classmethod
    def from_config(
        cls,
        config: SystemConfig | dict[str, Any],
        *,
        strict: bool = True,
        enabled_names: list[str] | None = None,
    ) -> SensorManager:
        """从 SystemConfig 或原始 dict 创建管理器。

        自动解析 cameras: 和 tactile: 段，创建对应传感器。

        Args:
            config: SystemConfig 或完整 YAML dict
            strict: 严格模式 (默认 True)。任何传感器创建失败立即抛出异常。
                     设为 False 则退化为 warning 并跳过。
            enabled_names: 图像传感器白名单。None 表示创建 cameras/tactile 全部设备。
        """
        if isinstance(config, dict):
            config = SystemConfig.model_validate(config)

        manager = cls(strict=strict)
        enabled_set = set(enabled_names) if enabled_names is not None else None
        if enabled_set is not None:
            all_names = set(config.cameras) | set(config.tactile)
            unknown = sorted(enabled_set - all_names)
            if unknown:
                raise ValueError(
                    "enabled_cameras contains unknown image sensor keys: "
                    f"{unknown}"
                )

        # 创建相机
        for cam_name, cam_cfg in config.cameras.items():
            if enabled_set is not None and cam_name not in enabled_set:
                continue
            try:
                camera = BaseRGBCamera.from_config(cam_name, cam_cfg)
                manager.add(cam_name, camera)
                logger.info("已创建相机: %s (type=%s)", cam_name, cam_cfg.type)
            except Exception:
                if strict:
                    raise
                logger.warning("创建相机 '%s' 失败", cam_name, exc_info=True)

        # 创建触觉传感器
        for sensor_name, sensor_cfg in config.tactile.items():
            if enabled_set is not None and sensor_name not in enabled_set:
                continue
            try:
                sensor = BaseTactileSensor.from_config(sensor_name, sensor_cfg)
                manager.add(sensor_name, sensor)
                logger.info("已创建触觉传感器: %s (type=%s)", sensor_name, sensor_cfg.type)
            except Exception:
                if strict:
                    raise
                logger.warning("创建触觉传感器 '%s' 失败", sensor_name, exc_info=True)

        return manager

    # ── 传感器管理 ──

    def add(self, name: str, sensor: BaseSensor) -> None:
        """添加传感器。"""
        self._sensors[name] = sensor

    def get(self, name: str) -> BaseSensor | None:
        """按名称获取传感器。"""
        return self._sensors.get(name)

    @property
    def sensors(self) -> dict[str, BaseSensor]:
        """所有传感器 (只读视图)。"""
        return dict(self._sensors)

    @property
    def cameras(self) -> dict[str, BaseRGBCamera]:
        """所有 RGB 相机。"""
        return {
            name: s for name, s in self._sensors.items()
            if isinstance(s, BaseRGBCamera)
        }

    @property
    def tactile_sensors(self) -> dict[str, BaseTactileSensor]:
        """所有触觉传感器。"""
        return {
            name: s for name, s in self._sensors.items()
            if isinstance(s, BaseTactileSensor)
        }

    def __len__(self) -> int:
        return len(self._sensors)

    def __contains__(self, name: str) -> bool:
        return name in self._sensors

    # ── 生命周期 ──

    def open_all(self) -> None:
        """打开所有传感器。strict 模式下任一失败立即抛出异常。"""
        for name, sensor in self._sensors.items():
            try:
                sensor.open()
            except Exception:
                if self._strict:
                    raise
                logger.warning("打开传感器 '%s' 失败", name, exc_info=True)

    def close_all(self) -> None:
        """关闭所有传感器 (close 始终尽力释放，不受 strict 影响)。"""
        for name, sensor in self._sensors.items():
            try:
                sensor.close()
            except Exception:
                logger.warning("关闭传感器 '%s' 失败", name, exc_info=True)

    def __enter__(self):
        self.open_all()
        return self

    def __exit__(self, *exc):
        self.close_all()

    # ── 数据读取 ──

    def read_all(self) -> dict[str, SensorFrame]:
        """读取所有已打开传感器的数据。

        Returns:
            {sensor_name: SensorFrame}
        """
        frames: dict[str, SensorFrame] = {}
        for name, sensor in self._sensors.items():
            if not sensor.is_open:
                continue
            try:
                frames[name] = sensor.read()
            except Exception:
                if self._strict:
                    raise
                logger.warning("读取传感器 '%s' 失败", name, exc_info=True)
        return frames

    def read_images(
        self,
        names: list[str] | None = None,
    ) -> dict[str, Any]:
        """获取彩色图像 (推理用便捷方法)。

        遍历相机和触觉传感器，提取图像数据:
          - 相机: 提取 streams["color"]["data"]
          - 触觉: 提取 streams["tactile"]["data"]

        Args:
            names: 要读取的传感器名称列表。None 表示读取所有。

        Returns:
            {sensor_name: image_array}
        """
        images: dict[str, Any] = {}

        targets = (
            {n: self._sensors[n] for n in names if n in self._sensors}
            if names is not None
            else self._sensors
        )
        for name, sensor in targets.items():
            if not sensor.is_open:
                continue
            try:
                frame = sensor.read()
                self._extract_image(name, frame, images)
            except Exception:
                if self._strict:
                    raise
                logger.warning("读取传感器 '%s' 图像失败", name, exc_info=True)

        return images

    @staticmethod
    def _extract_image(
        name: str, frame: SensorFrame, out: dict[str, Any],
    ) -> None:
        """从 SensorFrame 中提取图像数据到 out 字典。"""
        streams = frame.payload.get("streams", {})

        # 相机: 取 color 流
        color = streams.get("color", {})
        if "data" in color:
            out[name] = color["data"]
            return

        # 触觉: 取 tactile 流
        tactile = streams.get("tactile", {})
        if "data" in tactile:
            out[name] = tactile["data"]
