from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from Sensor.rgb_camera.base import BaseRGBCamera


@BaseRGBCamera.register("mock")
class MockRGBCamera(BaseRGBCamera):
    """测试用 Mock 相机。不需要真实硬件。"""

    def __init__(
        self,
        name: str = "front_camera",
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        with_depth: bool = False,
        distance_sequence: Sequence[float] = (2.5, 0.5, 1.5, 3.0),
        params: dict[str, Any] | None = None,
        auto_open: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            width=width,
            height=height,
            fps=fps,
            params=params,
        )
        self._with_depth = with_depth
        self._distance_sequence = tuple(distance_sequence)

        if auto_open:
            self.open()

    @classmethod
    def _from_config_dict(
        cls,
        name: str,
        cfg: dict[str, Any],
    ) -> MockRGBCamera:
        return cls(
            name=name,
            width=cfg.get("width", 640),
            height=cfg.get("height", 480),
            fps=cfg.get("fps", 30),
            with_depth=cfg.get("with_depth", False),
            params=cfg.get("params"),
            auto_open=False,
        )

    def _open_device(self) -> None:
        pass

    def _close_device(self) -> None:
        pass

    def _grab_streams(self) -> dict[str, dict[str, Any]]:
        color_data = np.random.randint(
            0, 256, (self.height, self.width, 3), dtype=np.uint8
        )
        streams: dict[str, dict[str, Any]] = {
            "color": {
                "data": color_data,
                "encoding": "rgb8",
                "width": self.width,
                "height": self.height,
                "channels": 3,
            },
        }

        if self._with_depth:
            idx = (self._frame_index - 1) % len(self._distance_sequence)
            distance = self._distance_sequence[idx]
            depth_data = np.full(
                (self.height, self.width), int(distance * 1000), dtype=np.uint16
            )
            streams["depth"] = {
                "data": depth_data,
                "encoding": "z16",
                "width": self.width,
                "height": self.height,
                "unit": "meter",
                "center_distance": distance,
            }

        return streams
