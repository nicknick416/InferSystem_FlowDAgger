from __future__ import annotations

from abc import abstractmethod
from typing import Any

from Core import SensorFrame, utc_now
from Core.config_schema import TactileConfig
from Core.registry import Registrable
from Sensor.base import BaseSensor


class BaseTactileSensor(Registrable["BaseTactileSensor"], BaseSensor):
    """视触觉传感器基类。

    在 BaseSensor 的生命周期和读取接口之上，增加:
      - 分辨率/帧率配置
      - 单路触觉图像流

    视触觉传感器本质是一个摄像头，通过图像捕捉接触面的形变来感知力。
    与 RGB 相机的区别:
      - 只有单路视频流，无 depth
      - 不需要复杂的参数管理 (曝光/白平衡由硬件固定)
      - 语义不同: 输出是触觉信息，不是场景图像

    子类注册:
      使用 @BaseTactileSensor.register("driver_type") 注册，
      通过 BaseTactileSensor.from_config() 自动分发。

    子类必须实现:
      _open_device()  — 打开硬件
      _close_device() — 释放硬件
      _grab_frame()   — 抓取一帧原始图像 (numpy BGR)
    """

    _registry: dict[str, type[BaseTactileSensor]] = {}
    _registry_label: str = "触觉传感器"

    @classmethod
    def from_config(
        cls,
        name: str,
        cfg: TactileConfig | dict[str, Any],
    ) -> BaseTactileSensor:
        """从 typed config 或 dict 创建实例。"""
        if isinstance(cfg, dict):
            cfg = TactileConfig.model_validate(cfg)
        factory = cls._resolve_factory(cfg.type)
        return factory._from_config_dict(name, cfg)

    @classmethod
    def _from_config_dict(
        cls,
        name: str,
        cfg: TactileConfig | dict[str, Any],
    ) -> BaseTactileSensor:
        """从 typed config 创建实例。子类覆盖此方法。"""
        raise NotImplementedError(f"{cls.__name__} 未实现 _from_config_dict")

    def __init__(
        self,
        name: str,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        super().__init__(name=name, sensor_type="tactile")
        self.width = width
        self.height = height
        self.fps = fps

    # ── 读取 ──

    def read(self) -> SensorFrame:
        return self.read_frame()

    def read_frame(self) -> SensorFrame:
        """读取一帧触觉图像，返回 SensorFrame。线程安全。"""
        with self._lock:
            self._ensure_open()
            frame_id = self._next_frame_index()
            data = self._grab_frame()
            return SensorFrame(
                sensor_name=self.name,
                sensor_type=self.sensor_type,
                timestamp=utc_now(),
                payload={
                    "frame_id": frame_id,
                    "streams": {
                        "tactile": {
                            "data": data,
                            "encoding": "bgr8",
                            "width": self.width,
                            "height": self.height,
                        },
                    },
                },
            )

    # ── 子类实现 ──

    @abstractmethod
    def _grab_frame(self) -> Any:
        """抓取一帧原始图像数据 (numpy array, BGR)。"""
