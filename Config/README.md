# YAML 配置项说明

`Config/` 目录用于保存机器人、传感器和推理运行时配置。所有 YAML 会在加载时由
`Core.config_schema.SystemConfig` 做 Pydantic 校验，因此字段类型错误、必填项缺失、
枚举值不合法会在硬件初始化前直接报错。

## 加载与校验

```python
from Core.config_schema import SystemConfig

cfg = SystemConfig.from_yaml("Config/arx5_example.yaml")
print(cfg.robot.type)
```

校验当前所有示例配置：

```bash
conda run -n infersystem python -c "from Core.config_schema import SystemConfig; [SystemConfig.from_yaml(p) for p in ['Config/rizon4_example.yaml','Config/arx5_example.yaml','Config/arx5_bimanual_example.yaml','Config/aloha_example.yaml','Config/ur_example.yaml']]"
```

## 顶层结构

```yaml
robot:       # 必填；机器人本体配置
cameras:     # 可选；相机字典，key 是逻辑相机名
tactile:     # 可选；触觉/视频设备字典
inference:   # 可选；推理服务器、动作映射、安全和平滑
logging:     # 可选；日志等级
```

`robot.type` 决定使用哪一种机器人配置。当前支持：
`flexiv`、`arx5`、`arx5_bimanual`、`aloha`、`ur`。

## 共享机器人字段

`joint_limits` 必须放在 `robot` 下，不要放到 `robot.control` 下。长度通常等于
单臂 DOF；双臂 ARX5 的共享限位也是 6 维，每侧复用。

```yaml
robot:
  joint_limits:
    position_min_deg: [-170, -120, -170, -180, -120, -360]
    position_max_deg: [170, 120, 170, 180, 120, 360]
    velocity_max: [2.0, 2.0, 2.0, 2.0, 2.5, 3.0]
    acceleration_max: [3.0, 3.0, 3.0, 3.0, 4.0, 4.0]
    torque_max: [60, 60, 45, 30, 20, 10]
```

## Flexiv 配置

示例文件：`Config/rizon4_example.yaml`。

必填/常用字段：
- `type: flexiv`
- `serial_number`: Flexiv 机器人序列号。
- `network_interface_whitelist`: 网络适配器白名单，`[]` 表示使用默认。
- `verbose`、`lite`: 传给 Flexiv RDK 的运行参数。
- `gripper`: Flexiv 夹爪参数。
- `control.default_mode`: 通常为 `NRT_JOINT_POSITION`。
- `control.frequency_hz`: 控制频率。
- `control.home_position_deg`: 7 维 home 关节角，单位度。
- `control.home_velocity_scale`: 回 home 速度百分比。

Flexiv 的夹爪通过独立 gripper 对象分发，所以 joint 模式通常配置：

```yaml
inference:
  arm_dim: 7
  gripper_index: 7
```

## ARX5 单臂配置

示例文件：`Config/arx5_example.yaml`。

必填/常用字段：
- `type: arx5`
- `model`: 通常为 `X5`。
- `interface` 或 `interface_name`: CAN 口，例如 `can0`。
- `canable_serial`: 可选，用于记录 USB-CAN 适配器序列号。
- `dof: 6`

控制字段：
- `background_send_recv`: 建议保持 `true`，使用 SDK 后台通信线程。
- `frequency_hz` 和 `controller_dt`: 常用 `500` 和 `0.002`。
- `over_current_cnt_max`: 过流计数阈值。
- `enable_gripper`: 是否把夹爪作为 robot action 的一部分控制。
- `gripper_width`、`gripper_kp`、`gripper_kd`、`gripper_closed_slack`。
- `home_position_deg`: 6 维 home 关节角，单位度。

ARX5 单臂夹爪是 robot action 的内置维度，不是外部 gripper dispatcher。
模型输出 `[j0..j5, gripper]` 时应配置：

```yaml
inference:
  arm_dim: 7
  gripper_index: -1
```

## ARX5 双臂配置

示例文件：`Config/arx5_bimanual_example.yaml`。

左右臂分别配置端点：

```yaml
robot:
  type: arx5_bimanual
  left_arm:
    model: X5
    interface: can0
    # urdf_path: Config/RobotModels/left_payload.urdf
  right_arm:
    model: X5
    interface: can1
    # urdf_path: Config/RobotModels/right_payload.urdf
```

`urdf_path` 可选。省略时驱动不会覆盖 `RobotConfig.urdf_path`，继续使用
pip `arx5-interface` 随对应型号提供的默认 URDF；只有配置了路径时才加载
仓库内的标定模型。路径相对仓库根目录解析，文件不存在时会在连接 CAN 前报错。

常用控制字段与 ARX5 单臂类似。`home_position_deg_14` 是可选 14 维 home，
格式为 `[Lj0..Lj5, Lg, Rj0..Rj5, Rg]`。`joint_limits` 是 6 维单臂限位，
左右臂共用。

双臂 joint action 格式为 `[Lj0..Lj5, Lg, Rj0..Rj5, Rg]`，推荐：

```yaml
inference:
  arm_dim: 14
  gripper_index: -1
```

### ARX5 末端负载自动标定

`Example/arx5/auto_payload_calibration.py` 是单文件、单臂运行的分阶段工具，
只使用 `arx5-interface`、Python 标准库和项目已有 NumPy。首次使用不要传
`--yes`，程序会在连接、运动路径、候选生成、控制器重启和部署前逐项确认。

```bash
python Example/arx5/auto_payload_calibration.py \
  --model X5 \
  --interface can0 \
  --output-urdf Config/RobotModels/arx1_left_payload.urdf
```

不传 `--urdf-path` 时以上命令以 pip SDK 默认 X5 URDF 为标定基线。程序先在
人工确认的当前姿态附近小范围采样，拟合附加质量和质心，生成
`*.candidate.urdf`，再提示操作员托住末端并重启控制器做验证。验证通过且
最终确认后才提升为 `--output-urdf`；失败会保留候选和 JSON 报告但不会部署。

左右臂必须分别标定，例如右臂改用 `--interface can1` 和不同输出路径。
标定完成后在对应 `left_arm` / `right_arm` 中配置生成的 `urdf_path`。删除该
字段即可回滚到 pip SDK 默认模型。

## Aloha / Piper 配置

示例文件：`Config/aloha_example.yaml`。

字段：
- `type: aloha`
- `can_interface_left`、`can_interface_right`: 左右 Piper CAN 口。
- `speed_percent`: Piper 运动速度百分比。
- `enable_timeout`: 使能超时时间，单位秒。
- `gripper_max_m`: 夹爪最大开口，单位米。
- `control.frequency_hz`: 控制频率。

Aloha joint action 格式为 `[Lj0..Lj5, Lg, Rj0..Rj5, Rg]`，推荐：

```yaml
inference:
  arm_dim: 14
  gripper_index: -1
```

## UR 配置

示例文件：`Config/ur_example.yaml`。

字段：
- `type: ur`
- `robot_ip`: 机器人 IP；兼容别名 `ip`、`host`、`address`。
- `dof`: 通常为 `6`。
- `control.frequency_hz`: 控制频率；兼容别名 `hz`。
- `control.joint_step_limit`: `servoJ` 单步关节限幅。
- `control.servoj_lookahead`、`control.servoj_gain`: RTDE `servoJ` 参数。
- `gripper.enabled`: 无 Robotiq 夹爪时设为 `false`。
- `gripper.port`、`max_width`、`speed`、`force`: Robotiq 参数。

UR 使用外部 Robotiq gripper dispatcher，模型输出 `[j0..j5, gripper]` 时推荐：

```yaml
inference:
  arm_dim: 6
  gripper_index: 6
```

## 相机配置

`cameras` 是字典，key 是逻辑相机名。`inference.enabled_cameras` 使用这些 key
选择要启用并发送给模型的图像传感器。

```yaml
cameras:
  top:
    type: realsense
    serial_number: "346522070312"
    enable_depth: false
    align_depth_to_color: true
    streams:
      color: {width: 640, height: 480, fps: 30, encoding: bgr8}
      depth: {width: 640, height: 480, fps: 30, encoding: z16, aligned_to: color}
    params:
      exposure.auto: false
      exposure.value: 4000
```

字段：
- `type`: 相机驱动类型，例如 `realsense`。
- `serial_number`: 设备序列号。
- `mapped_key`: 可选，推理请求中使用的图像 key；不填则使用 YAML 相机名。
- `calibration_role`: `eye_in_hand`、`eye_to_hand` 或空。
- `enable_depth`: 是否启用深度流。
- `align_depth_to_color`: 深度是否对齐到彩色图。
- `streams`: 各视频流配置。
- `params`: 驱动参数字典，例如曝光、增益、白平衡。

## 触觉/视频设备配置

`tactile` 也是字典，key 是逻辑设备名。

```yaml
tactile:
  wrist_left:
    type: opencv
    mapped_key: observation/tactile_image
    device_path: "/dev/video0"
    width: 640
    height: 480
    fps: 30
```

建议使用稳定的 `/dev/v4l/by-path/...` 路径，避免 USB 枚举顺序变化导致设备交换。
`mapped_key` 可选，只影响发给推理服务的图像 key；不填则使用 YAML 里的触觉设备名。

## 推理配置

基础字段：
- `server`: 推理服务器地址，格式 `host:port`。
- `recv_timeout_ms`: 接收超时，单位毫秒。
- `fps`: 本地控制循环频率。
- `n_execute`: 每个 action chunk 最多执行多少帧。
- `jpeg_quality`: 图像 JPEG 压缩质量。
- `enabled_cameras`: 启用并发送给模型的图像传感器 key 列表，可引用 `cameras`
  和 `tactile` 下的 key；不填则启用全部图像传感器。
- `inference_home`: 推理开始前移动到的关节位置。
- `latency_compensation`: 是否根据推理耗时自动跳过过期 action。

动作映射字段：
- `action_space`: 正常推理使用 `joint_position` 或 `cartesian`。
- `policy_format`: `normal` 使用有效维度；`canonical` 仅支持 `cartesian`，state 补 0
  到 `canonical_dim`，action chunk 也要求每帧为 `canonical_dim`。
- `canonical_dim`: canonical 格式固定维度，默认 `32`。
- `arm_dim` / `arm_dof`: 发送给 `robot.act()` 的动作维度。
- `gripper_index` / `gripper_action_index`: 外部 gripper 的动作下标；如果夹爪
  已经在 robot action 内部，设为 `-1`。
- `gripper_mode`: `binary` 按阈值开合；`width` 将 gripper action 值作为目标宽度（米）。
- `gripper_threshold`: 判断开合的阈值。

推荐维度：

| Robot | Joint action | `arm_dim` | `gripper_index` |
|---|---:|---:|---:|
| Flexiv + 外部夹爪 | `[j0..j6, g]` | `7` | `7` |
| ARX5 单臂 | `[j0..j5, g]` | `7` | `-1` |
| ARX5 双臂 | `[Lj0..Lj5,Lg,Rj0..Rj5,Rg]` | `14` | `-1` |
| Aloha/Piper | `[Lj0..Lj5,Lg,Rj0..Rj5,Rg]` | `14` | `-1` |
| UR + 外部夹爪 | `[j0..j5, g]` | `6` | `6` |

## EEF / Cartesian 模式

通过配置切换：

```yaml
inference:
  action_space: cartesian
```

模型输出使用 rot6d：
- 单臂：`[x,y,z,R00,R10,R20,R01,R11,R21,gripper]`
- 双臂：`[left_10_values, right_10_values]`

推理层会把 rot6d 转成库内统一 pose7 quaternion 格式：
`[x,y,z,qw,qx,qy,qz]`。安全检查会比较当前 `ArmState.eef_pose` 和 chunk
首帧 EEF action 的平移/旋转差。

## Canonical Policy Format

`canonical` 是面向训练/推理 server 的固定宽度协议，只支持 `cartesian`：

```yaml
inference:
  action_space: cartesian
  policy_format: canonical
  canonical_dim: 32
```

启用后，客户端发送给 server 的 `state` 总是 32 维，不足部分补 0。
server 返回的 action chunk 也必须是 `[chunk_size, 32]`：
- 单臂只使用前 10 维：`[x,y,z,rot6d,gripper]`。
- 双臂只使用前 20 维：`[left_10_values, right_10_values]`。
- 其余维度会被忽略，不会下发给机器人。

`policy_format: canonical` 与 `action_space: joint_position` 组合会在配置加载时直接报错。

## Temporal Smoothing

Temporal smoothing 用于融合新旧 action chunk 的重叠区，减少 chunk 接缝跳变。

```yaml
inference:
  latency_compensation: true
  smooth:
    enabled: true
    overlap_steps: 8
  async:
    enabled: false
    obs_fps: null
    max_latency_steps: 8
```

字段：
- `latency_compensation`: 根据推理耗时自动跳过已经过期的 action。
- `enabled`: 总开关。
- `overlap_steps`: 新旧 action chunk 融合的重叠长度。
- `async.enabled`: 是否启用后台推理线程；默认 `false`，保持同步推理。
- `async.obs_fps`: 后台发送 obs 的频率；`null` 表示使用 `inference.fps`。
- `async.max_latency_steps`: 异步新 chunk 到达时最多丢弃的前缀步数。

平滑逻辑只在 `Example/robot_inference.py` 控制循环中生效；直接调用 `robot.act()`
不会自动平滑。同步模式下平滑发生在每次阻塞推理返回之后；异步模式下后台线程持续
请求新 chunk，主线程按 `fps` 从平滑队列取 action 下发。

## 安全限制

```yaml
inference:
  max_joint_delta_rad: 0.5
  max_eef_delta_m: 0.10
  max_eef_rotation_delta_rad: 0.8
  max_joint_velocity_rad_s: 1.2
```

含义：
- `max_joint_delta_rad`: chunk 首帧与当前关节角的最大允许偏差。
- `max_eef_delta_m`: EEF 首帧最大平移偏差。
- `max_eef_rotation_delta_rad`: EEF 首帧最大旋转偏差。
- `max_joint_velocity_rad_s`: dispatcher 对连续 joint 命令做速度裁切；设为
  `null` 可关闭。

## 日志配置

```yaml
logging:
  console_level: INFO
  file_level: DEBUG
```

脚本调用统一日志初始化时会使用这些等级。

## 常见错误

- `joint_limits` 放错位置：应在 `robot.joint_limits`，不是 `robot.control.joint_limits`。
- ARX5/Aloha 的夹爪已经在 robot action 中，应使用 `gripper_index: -1`。
- Flexiv/UR 的外部夹爪需要设置正确的 `gripper_index`。
- `enabled_cameras` 必须引用 `cameras` 或 `tactile` 下存在的 key。
- `cartesian` 模式要求 robot 的 `observe()` 能提供 `eef_pose`。
- 同一个 action chunk 内所有 action 维度必须一致，否则 smoothing 无法融合。
