from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any

from Core import SensorFrame


class BaseSensor(ABC):
    """传感器基类 — 所有传感器的公共接口。

    第一性原理: 传感器做且只做两件事 —— 读取数据、管理生命周期。

    生命周期:
      open()   → 打开硬件设备
      close()  → 释放硬件资源
      支持 with 语句

    数据读取:
      read()   → SensorFrame   统一输出帧

    线程安全:
      所有公开方法通过 _lock 保护，多线程可安全调用。

    子类必须实现:
      _open_device()   打开硬件
      _close_device()  释放硬件
      read()           读取一帧数据 → SensorFrame
    """

    def __init__(
        self,
        name: str,
        sensor_type: str,
    ) -> None:
        self.name = name
        self.sensor_type = sensor_type
        self._is_open = False
        self._frame_index = 0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self._is_open

    # ── 生命周期 ──

    def open(self) -> None:
        """打开传感器硬件。幂等，重复调用无副作用。"""
        with self._lock:
            if self._is_open:
                return
            self._open_device()
            self._is_open = True

    def close(self) -> None:
        """关闭传感器硬件并释放资源。幂等。"""
        with self._lock:
            if not self._is_open:
                return
            try:
                self._close_device()
            finally:
                self._is_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # ── 数据读取 ──

    @abstractmethod
    def read(self) -> SensorFrame:
        """读取一帧传感器数据。"""

    # ── 内部 ──

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError(f"传感器未打开: {self.name}")

    def _next_frame_index(self) -> int:
        """递增并返回帧序号。"""
        self._frame_index += 1
        return self._frame_index

    @abstractmethod
    def _open_device(self) -> None:
        """打开硬件设备。子类实现。"""

    @abstractmethod
    def _close_device(self) -> None:
        """释放硬件资源。子类实现。"""
