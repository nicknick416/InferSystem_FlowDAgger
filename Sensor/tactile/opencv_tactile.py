from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from Core.config_schema import TactileConfig
from Sensor.tactile.base import BaseTactileSensor

logger = logging.getLogger(__name__)


@BaseTactileSensor.register("opencv")
class OpenCVTactileSensor(BaseTactileSensor):
    """基于 OpenCV VideoCapture 的视触觉传感器驱动。

    通过 USB 摄像头捕捉触觉传感器的接触面图像。
    device_path 推荐使用 /dev/v4l/by-path/... 以保证设备映射稳定。

    用法::

        sensor = OpenCVTactileSensor("wrist_left", device_path="/dev/video0")
        with sensor:
            frame = sensor.read_frame()
            img = frame.payload["streams"]["tactile"]["data"]  # numpy BGR
    """

    def __init__(
        self,
        name: str,
        *,
        device_path: str | int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        super().__init__(
            name=name,
            width=width,
            height=height,
            fps=fps,
        )
        self._device_path = device_path
        self._cap: cv2.VideoCapture | None = None

    @classmethod
    def _from_config_dict(
        cls,
        name: str,
        cfg: TactileConfig | dict[str, Any],
    ) -> OpenCVTactileSensor:
        if isinstance(cfg, dict):
            cfg = TactileConfig.model_validate(cfg)
        return cls(
            name=name,
            device_path=cfg.device_path,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
        )

    @property
    def device_path(self) -> str | int:
        return self._device_path

    def _open_device(self) -> None:
        self._cap = cv2.VideoCapture(self._device_path)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"无法打开触觉传感器 '{self.name}': {self._device_path}"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        logger.info("触觉传感器已打开: %s (%s)", self.name, self._device_path)

    def _close_device(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("触觉传感器已关闭: %s", self.name)

    def _grab_frame(self) -> np.ndarray:
        assert self._cap is not None
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(
                f"触觉传感器 '{self.name}' 读取失败"
            )
        return frame
