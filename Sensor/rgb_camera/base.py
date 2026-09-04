from __future__ import annotations

from abc import abstractmethod
from typing import Any

from Core import SensorFrame, utc_now
from Core.config_schema import CameraConfig
from Core.registry import Registrable
from Sensor.base import BaseSensor


class BaseRGBCamera(Registrable["BaseRGBCamera"], BaseSensor):
    """RGB 相机基类。

    在 BaseSensor 的生命周期和读取接口之上，增加:
      - 分辨率/帧率配置
      - 设备参数管理 (曝光、白平衡等)
      - 多流抓取 (color + 可选 depth)

    子类注册:
      使用 @BaseRGBCamera.register("driver_type") 注册，
      通过 BaseRGBCamera.from_config() 自动分发。

    子类必须实现:
      _open_device()   — 打开硬件
      _close_device()  — 释放硬件
      _grab_streams()  — 抓取一帧，返回 {"color": {...}, "depth": {...}}
    """

    _registry: dict[str, type[BaseRGBCamera]] = {}
    _registry_label: str = "相机"

    @classmethod
    def from_config(
        cls,
        name: str,
        cfg: CameraConfig | dict[str, Any],
    ) -> BaseRGBCamera:
        """从 typed config 或 dict 创建实例。"""
        if isinstance(cfg, dict):
            cfg = CameraConfig.model_validate(cfg)
        factory = cls._resolve_factory(cfg.type)
        return factory._from_config_dict(name, cfg)

    @classmethod
    def _from_config_dict(
        cls,
        name: str,
        cfg: CameraConfig | dict[str, Any],
    ) -> BaseRGBCamera:
        """从 typed config 创建实例。子类覆盖此方法。"""
        raise NotImplementedError(f"{cls.__name__} 未实现 _from_config_dict")

    def __init__(
        self,
        name: str,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, sensor_type="rgb_camera")
        self.width = width
        self.height = height
        self.fps = fps
        self._params: dict[str, Any] = dict(params or {})

    # ── 生命周期扩展: open 时自动应用参数 ──

    def open(self) -> None:
        super().open()
        if self._is_open:
            self._apply_params()

    # ── 读取 ──

    def read(self) -> SensorFrame:
        return self.read_frame()

    def read_frame(self) -> SensorFrame:
        """读取一帧，返回 SensorFrame。线程安全。"""
        with self._lock:
            self._ensure_open()
            frame_id = self._next_frame_index()
            streams = self._grab_streams()
            return SensorFrame(
                sensor_name=self.name,
                sensor_type=self.sensor_type,
                timestamp=utc_now(),
                payload={
                    "frame_id": frame_id,
                    "streams": streams,
                },
            )

    # ── 参数管理 ──

    def set_param(self, key: str, value: Any) -> None:
        self._params[key] = value
        if self._is_open:
            self._apply_param_to_device(key, value)

    def set_params(self, params: dict[str, Any] | None = None, **kwargs: Any) -> None:
        merged = dict(params or {})
        merged.update(kwargs)
        for key, value in merged.items():
            self.set_param(key, value)

    def get_param(self, key: str, default: Any = None) -> Any:
        return self._params.get(key, default)

    def get_params(self) -> dict[str, Any]:
        return dict(self._params)

    # ── 内部 ──

    def _apply_params(self) -> None:
        """将所有缓存参数写入硬件。open() 后自动调用。"""
        for key, value in self._params.items():
            self._apply_param_to_device(key, value)

    def _apply_param_to_device(self, key: str, value: Any) -> None:
        """将单个参数写入硬件。子类按需覆盖。"""

    # ── 子类实现 ──

    @abstractmethod
    def _grab_streams(self) -> dict[str, dict[str, Any]]:
        """抓取一帧所有流数据。

        返回格式::

            {
                "color": {"data": np.ndarray, "encoding": "bgr8", ...},
                "depth": {"data": np.ndarray, "encoding": "z16", ...},  # 可选
            }
        """
