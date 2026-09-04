# Robot 模块使用规约

## 1. 设计原则

本模块基于第一性原理设计。一个受控机器人只做三件不可再分解的事：

```
观测    observe()  → ArmState     一次硬件读取，原子状态快照
执行    act()      → None         统一动作入口，非阻塞，一个控制周期
生命周期 connect() / disconnect()  建立/释放硬件连接
```

一切上层接口（阻塞移动、回 Home、夹爪控制）都从这三个原语推导而来，而非基类强制的抽象方法。

## 2. 架构

```
BaseRobot (抽象基类)
│
│  抽象原语 (子类必须实现):
│    observe()        → ArmState
│    act(action)      → None
│    connect / disconnect / enable / stop / ...
│
│  任务便捷方法 (有默认实现，子类可覆盖优化):
│    move_joint_position()    阻塞移动到关节位置
│    move_joint_velocity()    以速度运动一段时间
│    move_joint_torque()      施加力矩一段时间
│    move_eef()               阻塞移动末端位姿
│    go_home()                回 Home
│
│  末端工具工厂:
│    create_gripper()  → BaseGripper | None
│
├── FlexivRobot      飞夕 Rizon 系列
├── (YourRobot)      通过 @BaseRobot.register() 接入
└── ...

BaseGripper (抽象基类, 独立于机器人)
│
│  observe()  → GripperState
│  set(open)  / move(width)
│  connect()  / disconnect()
│
├── FlexivGripper    飞夕配套夹爪
└── ...
```

关键设计决策：

- **观测是原子的** — `observe()` 一次读取全部传感器，返回带时间戳的快照，保证数据时间一致性
- **执行是统一的** — `act(Action)` 是唯一的实时控制入口，上层推理只需产出 `Action` 对象
- **夹爪是独立设备** — 不在基类接口中，通过 `create_gripper()` 按需创建
- **任务层由原语组合** — `move_*` 系列方法有默认实现（act + poll），子类可覆盖以利用特定 SDK 能力
- **硬件概念不泄露** — `switch_mode` 等硬件特有方法留在子类，不污染基类接口

## 3. 生命周期

```python
from Robot import BaseRobot

# 推荐: with 语句
with BaseRobot.from_config("config.yaml") as robot:
    robot.enable()
    robot.wait_until_operational()
    # ... 使用机器人 ...

# 手动管理
robot = BaseRobot.from_config("config.yaml")
robot.connect()     # 建立硬件连接
robot.enable()      # 使能伺服
# ... 使用 ...
robot.disconnect()  # 释放资源
```

约束：
- 构造函数中**不访问硬件**，硬件查询延迟到 `connect()`
- 未调用 `connect()` 就调用其他方法抛出 `RuntimeError`
- `disconnect()` 会尝试停止运动后释放资源

## 4. 核心原语

### 4.1 observe() — 原子观测

```python
state = robot.observe()

# 以下字段来自同一时刻的硬件采样
state.timestamp             # perf_counter 时间戳
state.joint_positions       # rad
state.joint_velocities      # rad/s
state.joint_torques         # Nm
state.eef_pose              # [x, y, z, qw, qx, qy, qz]
state.wrench_in_tcp         # [fx, fy, fz, tx, ty, tz]
# ... 完整字段见 ArmState 定义
```

### 4.2 act(action) — 统一执行

```python
from Core import Action, ActionSpace

# 关节位置
robot.act(Action(ActionSpace.JOINT_POSITION, target_q))

# 关节位置 + 前馈速度
robot.act(Action(ActionSpace.JOINT_POSITION, target_q,
                 extra={"velocities": target_dq}))

# 关节速度
robot.act(Action(ActionSpace.JOINT_VELOCITY, target_dq))

# 关节力矩
robot.act(Action(ActionSpace.JOINT_TORQUE, target_tau))

# 笛卡尔位姿 + 力
robot.act(Action(ActionSpace.CARTESIAN, [x, y, z, qw, qx, qy, qz],
                 extra={"wrench": [fx, fy, fz, tx, ty, tz]}))
```

典型控制循环：

```python
import time
from Core import Action, ActionSpace

dt = 1.0 / robot.get_params().control_frequency_hz
next_time = time.perf_counter()

for step in range(num_steps):
    state = robot.observe()                     # 1. 原子观测
    action_values = policy.predict(state)       # 2. 推理
    action = Action(ActionSpace.JOINT_POSITION, action_values)
    robot.act(action, state=state)              # 3. 执行 (复用已有观测)
    next_time += dt
    BaseRobot._rt_sleep_until(next_time)        # 4. 高精度定时
```

传入 `state=` 参数可避免 `act()` 内部重复读取硬件（用于速度积分等场景）。

## 5. 任务层便捷方法

这些方法有默认实现（基于 `act()` + `observe()` 循环），子类可覆盖以使用 SDK 原生能力：

```python
# 阻塞移动到关节位置，返回是否到达
reached = robot.move_joint_position([0, -0.3, 0, 1.5, 0, 0.3, 0], velocity=0.5)

# 以指定速度运动 2 秒
robot.move_joint_velocity([0.1, 0, 0, 0, 0, 0, 0], duration=2.0)

# 阻塞移动末端
robot.move_eef([0.4, 0.0, 0.3])  # 保持当前姿态

# 回 Home
robot.go_home(velocity=0.5)
```

## 6. 夹爪 (独立设备)

夹爪是挂载在机器人上的**独立设备**，拥有独立的生命周期和接口：

```python
with BaseRobot.from_config("config.yaml") as robot:
    robot.enable()
    robot.wait_until_operational()

    # 通过机器人创建夹爪 (传入所需的底层句柄)
    gripper = robot.create_gripper()
    if gripper is not None:
        with gripper:                     # connect + disconnect
            gripper.set(True)             # 打开
            gripper.set(False)            # 关闭 (抓取)
            gripper.move(0.05)            # 移动到 5cm
            gs = gripper.observe()        # 读取状态
            print(gs.width, gs.force)
```

为什么分离：
- 物理上夹爪是独立设备，有独立的控制板和通信
- 不是所有机器人都有夹爪
- 夹爪可以替换为吸盘、焊枪等其他末端工具
- 不同夹爪型号的接口差异不应影响机器人基类

## 7. 安全机制

### 输入校验

所有控制路径在下发指令前自动校验：
- **长度校验** — 输入序列长度必须匹配自由度
- **关节限位** — 位置、速度、力矩不得超出 `RobotParams` 极限值

校验不通过抛出 `ValueError`，指令不会下发到硬件。

### 紧急停止

```python
robot.emergency_stop()    # 立即停止
# 恢复:
robot.clear_fault()
robot.enable()
```

## 8. 新机器人接入指南

### 步骤一: 创建子类

```python
# Robot/ur.py
from Core import Action, ActionSpace, ArmState, RobotParams
from Robot.base import BaseRobot


@BaseRobot.register("ur")
class URRobot(BaseRobot):
    """Universal Robots 驱动。"""

    def __init__(self, ip: str, *, name: str | None = None) -> None:
        self._ip = ip
        self._handle = None
        super().__init__(name=name or ip, robot_type="ur", dof=6)

    @classmethod
    def _from_config_dict(cls, robot_cfg: dict) -> "URRobot":
        return cls(ip=robot_cfg["ip"], name=robot_cfg.get("name"))

    # ── 生命周期 (必须实现) ──
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def enable(self) -> None: ...
    def stop(self) -> None: ...
    def emergency_stop(self) -> None: ...
    def clear_fault(self) -> None: ...
    def get_params(self) -> RobotParams: ...
    def is_connected(self) -> bool: ...
    def is_operational(self) -> bool: ...
    def is_busy(self) -> bool: ...
    def is_fault(self) -> bool: ...

    # ── 核心原语 (必须实现) ──
    def observe(self) -> ArmState: ...
    def act(self, action: Action, *, state: ArmState | None = None) -> None: ...

    # ── 任务层 (可选覆盖，有默认实现) ──
    # move_joint_position, move_eef, go_home 等
    # 如果 SDK 有更高效的阻塞移动接口，覆盖即可
```

### 步骤二: 在 `__init__.py` 中导入

```python
from .ur import URRobot
```

### 步骤三: 编写 YAML 配置

```yaml
robot:
  type: ur
  ip: "192.168.1.100"
  name: "ur5e_arm"
```

### 必须实现的抽象方法

| 类别 | 方法 |
|------|------|
| 生命周期 | `connect`, `disconnect`, `enable`, `stop`, `emergency_stop`, `clear_fault`, `get_params` |
| 状态查询 | `is_connected`, `is_operational`, `is_busy`, `is_fault` |
| 核心原语 | `observe`, `act` |

### 可选覆盖的任务方法

| 方法 | 默认实现 | 何时覆盖 |
|------|---------|---------|
| `move_joint_position()` | act() 循环 + observe() 轮询 | SDK 有原生 MoveJ |
| `move_eef()` | act() 循环 + observe() 轮询 | SDK 有原生 MoveL |
| `go_home()` | 调用 move_joint_position(home) | SDK 有原生 Home 指令；Flexiv 当前使用 YAML `home_position_deg` |
| `move_joint_velocity()` | act() 循环 | 通常不需要覆盖 |
| `move_joint_torque()` | act() 循环 | 通常不需要覆盖 |
| `create_gripper()` | 返回 None | 有配套夹爪时覆盖 |

## 9. 单位约定

| 物理量 | 单位 | 备注 |
|--------|------|------|
| 关节位置 | rad | YAML 配置中用 deg，代码内部统一 rad |
| 关节速度 | rad/s | |
| 关节力矩 | Nm | |
| 末端位置 | m | |
| 末端姿态 | 四元数 | 标量在前 [qw, qx, qy, qz] |
| 夹爪宽度 | m | |
| 力/力矩 | N / Nm | |
| 速度比例 | [0.0, 1.0] | 0.0 最慢，1.0 最快 |
| 控制频率 | Hz | |
| 时间戳 | float | time.perf_counter() |

## 10. YAML 配置文件格式

```yaml
robot:
  type: flexiv
  serial_number: "Rizon4s-123456"
  name: "arm_0"
  network_interface_whitelist: []
  verbose: true
  lite: false

  gripper:
    name: "Robotiq-2F-85"
    max_width: 0.085
    default_velocity: 0.1
    default_force: 20.0

  control:
    frequency_hz: 1000
    home_position_deg: [0, -20, 0, 90, 0, 20, 0]
    home_velocity_scale: 50

  joint_limits:
    position_min_deg: [-160, -130, -170, -107, -170, -80, -170]
    position_max_deg: [+160, +130, +170, +154, +170, +260, +170]
    velocity_max: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    torque_max: [100, 100, 50, 50, 20, 20, 20]

cameras:
  # ... (机器人模块不消费)

inference:
  # ... (机器人模块不消费)
```
