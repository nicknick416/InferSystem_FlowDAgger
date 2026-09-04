from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class GripperState:
    """夹爪当前状态。"""
    width: float = 0.0
    max_width: float = 0.0
    force: float = 0.0
    is_moving: bool = False


class BaseGripper(ABC):
    """夹爪基类 — 独立于机器人的设备。

    夹爪在物理上是挂载在机器人上的独立设备，拥有独立的:
      - 通信通道 (可能复用机器人总线，但逻辑独立)
      - 生命周期 (初始化、使能、释放)
      - 状态 (宽度、力、运动中)
      - 控制接口 (开合、移动到宽度)

    通过 robot.create_gripper(config) 创建，获取所需的底层句柄。
    """

    def __init__(self, name: str = "") -> None:
        self.name = name

    @abstractmethod
    def connect(self) -> None:
        """初始化夹爪硬件连接。"""

    @abstractmethod
    def disconnect(self) -> None:
        """释放夹爪资源。"""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    @abstractmethod
    def observe(self) -> GripperState:
        """读取夹爪当前状态。"""

    @abstractmethod
    def set(self, open: bool, *, wait: bool = True) -> None:
        """二值控制。True=打开, False=关闭（抓取）。"""

    @abstractmethod
    def move(
        self,
        width: float,
        *,
        velocity: float | None = None,
        force: float | None = None,
        wait: bool = True,
    ) -> None:
        """移动夹爪到指定宽度 (m)。"""
