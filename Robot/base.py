from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from Core import Action, ActionSpace, ArmState, RobotParams, load_config, load_yaml
from Core.config_schema import SystemConfig
from Core.registry import Registrable
from Robot.gripper import BaseGripper

logger = logging.getLogger(__name__)


class BaseRobot(Registrable["BaseRobot"], ABC):
    """机器人基类。

    第一性原理: 机器人做且只做三件事 —— 观测、执行、生命周期管理。
    一切上层接口都从这三个原语推导而来。

    原语层 (抽象，子类必须实现):
      observe()       → ArmState    原子状态快照，一次读取全部传感器
      act(action)     → bool|None   统一动作入口，非阻塞，一个控制周期

    任务层 (非抽象，有默认实现，子类可覆盖以优化):
      move_joint_position()          阻塞移动到目标关节位置
      move_joint_velocity()          以指定速度运动一段时间
      move_joint_torque()            施加指定力矩一段时间
      move_eef()                     阻塞移动末端到目标位姿
      go_home()                      回到 Home 位置

    末端工具:
      create_gripper()  → BaseGripper | None   创建关联夹爪 (独立设备)

    子类注册:
      使用 @BaseRobot.register("robot_type") 装饰器注册，
      即可通过 BaseRobot.from_config() 自动分发创建。
    """

    _registry: dict[str, type[BaseRobot]] = {}
    _registry_label: str = "机器人"

    def __init__(self, name: str, robot_type: str, *, dof: int = 7) -> None:
        self.name = name
        self.robot_type = robot_type
        self.dof = dof
        self.config: dict[str, Any] = {}  # 完整原始配置，供上层读取
        self.system_config: SystemConfig | None = None  # 验证后的完整配置

    # ── 从配置文件创建 ────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str | Path) -> BaseRobot:
        """从 YAML 配置文件创建机器人实例。

        YAML 结构:
          robot:     机器人配置 (type, serial_number, ...)
          cameras:   相机配置
          tactile:   触觉传感器配置
          inference: 推理参数

        使用 Pydantic schema 验证配置，机器人只消费 robot: 段。
        """
        system_config = load_config(config_path)
        robot_type = system_config.robot.type

        factory = cls._resolve_factory(robot_type)
        robot = factory._from_config_dict(system_config.robot)
        robot.system_config = system_config
        # 保留原始 dict 供尚未迁移的上层代码使用
        robot.config = load_yaml(config_path)
        return robot

    @classmethod
    def _from_config_dict(cls, robot_cfg: Any) -> BaseRobot:
        """从 typed config 创建实例。子类必须覆盖此方法。"""
        raise NotImplementedError(
            f"{cls.__name__} 未实现 _from_config_dict"
        )

    # ================================================================
    #  生命周期 (抽象)
    # ================================================================

    @abstractmethod
    def connect(self) -> None:
        """连接机器人并完成硬件初始化。"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接并释放资源。"""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    def __del__(self):
        """GC 回收时兜底调用 disconnect（确保机械臂安全停机）。"""
        try:
            self.disconnect()
        except Exception:
            logger.error(
                "GC 回收时 disconnect() 异常: %s",
                getattr(self, "name", self.__class__.__name__),
                exc_info=True,
            )

    @abstractmethod
    def enable(self) -> None:
        """使能机器人。"""

    @abstractmethod
    def stop(self) -> None:
        """停止所有运动。"""

    @abstractmethod
    def emergency_stop(self) -> None:
        """紧急停止 — 立即停止所有运动。

        与 stop() 的区别: emergency_stop() 以最高优先级执行，
        不等待减速，适用于危险场景。调用后需要 clear_fault() + enable()
        才能恢复控制。
        """

    @abstractmethod
    def clear_fault(self) -> None:
        """清除故障。"""

    @abstractmethod
    def get_params(self) -> RobotParams:
        """返回机器人硬件参数。"""

    # ── 状态查询 ──

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def is_operational(self) -> bool: ...

    @abstractmethod
    def is_busy(self) -> bool: ...

    @abstractmethod
    def is_fault(self) -> bool: ...

    # ================================================================
    #  核心原语 (抽象)
    # ================================================================

    @abstractmethod
    def observe(self) -> ArmState:
        """原子状态快照 — 一次硬件读取，返回完整状态。

        保证所有字段来自同一时刻的采样。
        """

    @abstractmethod
    def act(self, action: Action, *, state: ArmState | None = None) -> bool | None:
        """执行一步动作 (非阻塞，一个控制周期)。

        参数:
            action: 统一动作指令
            state:  可选，提供已有观测以避免重复硬件读取。
                    用于 JOINT_VELOCITY 等需要当前状态做积分的场景。

        返回:
            显式 ``False`` 表示动作未执行；``True`` 或旧驱动返回的 ``None``
            均表示上层可继续推进，以保持向后兼容。
        """

    # ================================================================
    #  末端工具工厂
    # ================================================================

    def create_gripper(
        self, config: dict[str, Any] | None = None,
    ) -> BaseGripper | None:
        """创建与本机器人关联的夹爪。

        夹爪是独立设备，但可能需要机器人的底层句柄来通信。
        子类覆盖此方法，传入所需的内部句柄。
        默认返回 None (无夹爪)。
        """
        return None

    # ================================================================
    #  便捷方法: 状态读取 (由 observe() 推导)
    # ================================================================

    def get_joint_positions(self) -> list[float]:
        """当前各关节位置 (rad)。便捷方法，内部调用 observe()。"""
        return self.observe().joint_positions

    def get_joint_velocities(self) -> list[float]:
        """当前各关节速度 (rad/s)。"""
        return self.observe().joint_velocities

    def get_joint_torques(self) -> list[float]:
        """当前各关节实际力矩 (Nm)。"""
        return self.observe().joint_torques

    def get_eef_pose(self) -> list[float]:
        """末端位姿。"""
        return self.observe().eef_pose

    # ================================================================
    #  任务层: 阻塞控制 (非抽象，有默认实现，子类可覆盖)
    # ================================================================

    def move_joint_position(
        self,
        positions: Sequence[float],
        *,
        velocity: float | None = None,
        tolerance: float = 0.01,
        timeout_s: float = 30.0,
    ) -> bool:
        """阻塞移动到目标关节位置。

        默认实现: 以控制频率持续发送 act()，轮询 observe() 直到到达。
        子类可覆盖以使用更高效的实现 (如运动规划原语)。

        返回: 是否在超时前到达目标。
        """
        self._validate_length("positions", positions)
        self._validate_joint_positions(positions)

        extra = {"velocity": velocity} if velocity is not None else {}
        action = Action(ActionSpace.JOINT_POSITION, list(positions), extra=extra)

        dt = 1.0 / self.get_params().control_frequency_hz
        deadline = time.monotonic() + timeout_s
        next_time = time.perf_counter()

        while time.monotonic() < deadline:
            state = self.observe()
            if all(
                abs(a - b) < tolerance
                for a, b in zip(state.joint_positions, positions)
            ):
                return True
            self.act(action, state=state)
            next_time += dt
            self._rt_sleep_until(next_time)
        return False

    def move_joint_velocity(
        self,
        velocities: Sequence[float],
        duration: float,
    ) -> None:
        """以指定关节速度运动一段时间，结束后自动停止。

        默认实现: 以控制频率持续发送 act()。
        """
        self._validate_length("velocities", velocities)
        self._validate_joint_velocities(velocities)

        action = Action(ActionSpace.JOINT_VELOCITY, list(velocities))
        dt = 1.0 / self.get_params().control_frequency_hz
        steps = int(duration / dt)
        next_time = time.perf_counter()

        for _ in range(steps):
            state = self.observe()
            self.act(action, state=state)
            next_time += dt
            self._rt_sleep_until(next_time)

        # 停止
        self.act(Action(ActionSpace.JOINT_VELOCITY, [0.0] * self.dof))

    def move_joint_torque(
        self,
        torques: Sequence[float],
        duration: float,
    ) -> None:
        """施加指定关节力矩一段时间，结束后自动停止。

        默认实现: 以控制频率持续发送 act()。
        """
        self._validate_length("torques", torques)
        self._validate_joint_torques(torques)

        action = Action(ActionSpace.JOINT_TORQUE, list(torques))
        dt = 1.0 / self.get_params().control_frequency_hz
        steps = int(duration / dt)
        next_time = time.perf_counter()

        for _ in range(steps):
            self.act(action)
            next_time += dt
            self._rt_sleep_until(next_time)

        self.act(Action(ActionSpace.JOINT_TORQUE, [0.0] * self.dof))

    def move_eef(
        self,
        position: Sequence[float],
        orientation: Sequence[float] | None = None,
        *,
        velocity: float | None = None,
        tolerance: float = 0.005,
        timeout_s: float = 30.0,
    ) -> bool:
        """阻塞移动末端到目标位姿。

        默认实现: 以控制频率持续发送 act()，轮询 observe() 直到到达。
        子类可覆盖以使用更高效的实现。

        返回: 是否在超时前到达目标。
        """
        self._validate_length("position", position, expected=3)
        if orientation is None:
            state = self.observe()
            orientation = state.eef_pose[3:7]  # 保持当前姿态
        pose = list(position) + list(orientation)
        extra = {"velocity": velocity} if velocity is not None else {}
        action = Action(ActionSpace.CARTESIAN, pose, extra=extra)

        dt = 1.0 / self.get_params().control_frequency_hz
        deadline = time.monotonic() + timeout_s
        next_time = time.perf_counter()

        while time.monotonic() < deadline:
            state = self.observe()
            if all(
                abs(a - b) < tolerance
                for a, b in zip(state.eef_pose[:3], position)
            ):
                return True
            self.act(action, state=state)
            next_time += dt
            self._rt_sleep_until(next_time)
        return False

    def go_home(self, *, velocity: float | None = None, timeout_s: float = 60.0) -> bool:
        """回到 Home 位置（阻塞）。

        默认实现: 调用 move_joint_position(home_position)。
        """
        home = self.get_params().home_position
        if not home:
            raise RuntimeError("未配置 Home 位置")
        return self.move_joint_position(
            home, velocity=velocity, timeout_s=timeout_s,
        )

    def wait_until_operational(self, timeout_s: float = 10.0) -> bool:
        """等待机器人进入 operational 状态。"""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.is_operational():
                return True
            time.sleep(0.2)
        return self.is_operational()

    # ================================================================
    #  安全校验
    # ================================================================

    def _validate_length(
        self, name: str, seq: Sequence[float], expected: int | None = None,
    ) -> None:
        """校验序列长度是否与自由度匹配。"""
        n = expected if expected is not None else self.dof
        if len(seq) != n:
            raise ValueError(
                f"{name} 长度 ({len(seq)}) 与期望 ({n}) 不匹配"
            )

    def _validate_joint_positions(self, positions: Sequence[float]) -> None:
        """校验关节位置是否在限位范围内。"""
        params = self.get_params()
        for i, (pos, lo, hi) in enumerate(zip(
            positions, params.joint_position_min, params.joint_position_max,
        )):
            if not (lo <= pos <= hi):
                raise ValueError(
                    f"关节 {i} 位置 {pos:.4f} rad 超出限位 "
                    f"[{lo:.4f}, {hi:.4f}]"
                )

    def _validate_joint_velocities(self, velocities: Sequence[float]) -> None:
        """校验关节速度是否在限制范围内。"""
        params = self.get_params()
        for i, (vel, max_vel) in enumerate(zip(
            velocities, params.joint_velocity_max,
        )):
            if abs(vel) > max_vel:
                raise ValueError(
                    f"关节 {i} 速度 {abs(vel):.4f} rad/s 超出限制 "
                    f"±{max_vel:.4f}"
                )

    def _validate_joint_torques(self, torques: Sequence[float]) -> None:
        """校验关节力矩是否在限制范围内。"""
        params = self.get_params()
        for i, (tau, max_tau) in enumerate(zip(
            torques, params.joint_torque_max,
        )):
            if abs(tau) > max_tau:
                raise ValueError(
                    f"关节 {i} 力矩 {abs(tau):.4f} Nm 超出限制 "
                    f"±{max_tau:.4f}"
                )

    @staticmethod
    def _rt_sleep_until(target_time: float) -> None:
        """高精度等待，用于实时控制循环。

        策略: 剩余 >2ms 时用 sleep 让出 CPU，最后阶段忙等保证精度。
        """
        remaining = target_time - time.perf_counter()
        if remaining > 0.002:
            time.sleep(remaining - 0.001)
        while time.perf_counter() < target_time:
            pass
