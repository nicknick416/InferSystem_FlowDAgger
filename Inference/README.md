# Inference — 推理通信层

## 设计原则

推理通信层解决一个问题: **机器人端的观测 → 网络 → GPU 端的模型 → 网络 → 机器人端的动作**。

不做模型推理本身，不依赖机器人/传感器代码。Server 可以直接部署到任何有 GPU 的机器上。

## 架构

```
InferenceServer (server.py)   — GPU 机器上运行，接收观测，调用模型，返回 action chunk
InferenceClient (client.py)   — 机器人端运行，发送观测，获取 action
```

## 协议 (ZMQ REP/REQ, msgpack)

```
客户端 → 服务器:
  {"cmd": "predict", "state": [float, ...], "<cam_name>": RGB uint8 CHW list, ...}
  {"cmd": "reset"}

服务器 → 客户端:
  {"status": "ok", "actions": [[float, ...], ...], "infer_time_ms": float}
  {"status": "ok"}
  {"status": "error", "message": str}
```

图像字段优先使用当前 client 格式：`RGB uint8 CHW`，通常为
`(3, 224, 224)` 的 msgpack list。Server 仍兼容旧的 `bytes(JPEG)`，
会在 server 侧转 RGB、resize/pad 到 `224x224`，再转为 CHW；如果传
HWC list/array，应按 RGB 排列。

## 服务器用法

```python
from Inference import InferenceServer

# 函数式
def my_predict(images: dict[str, np.ndarray], state: list[float]) -> list[list[float]]:
    return my_model.infer(images, state)

server = InferenceServer("tcp://*:5555", predict_fn=my_predict)
server.run()

# 子类式
class MyServer(InferenceServer):
    def predict(self, images, state):
        return self.model(images, state)

    def on_reset(self):
        self.model.reset()
```

## 客户端用法

### 方式 1: 获取完整 chunk

一次拿到全部 action，自行控制执行节奏:

```python
from Inference import InferenceClient

client = InferenceClient("192.168.50.225:5555")
with client:
    client.reset()
    actions = client.predict_chunk(images, state)  # list[list[float]]
    for action in actions[:n_execute]:
        robot.send_joint_position(action[:7])
```

### 方式 2: 逐帧获取 (推荐)

每次返回一帧 action，队列空时自动用最新观测请求新 chunk:

```python
with InferenceClient("192.168.50.225:5555") as client:
    client.reset()
    while running:
        images = cameras.read()
        state = robot.get_state()
        action = client.get_action(images, state)   # list[float], 单帧
        robot.send_joint_position(action[:7])
```

**方式 2 的关键**: 每次调用 `get_action()` 都会存储最新的 `images` 和 `state`。
当队列用完时，会用**最新存储的观测**（而不是旧观测）去请求新 chunk。
这对于视觉伺服等需要持续更新观测的场景至关重要。

## 动作空间与平滑

`inference.action_space` 控制推理 action 的执行方式:

- `joint_position`: 关节位置。单臂常见为 `[j0..jN, gripper]`，双臂 ARX5 为 `[Lj0..Lj5, Lg, Rj0..Rj5, Rg]`。
- `cartesian`: EEF 位姿。模型输出使用 6D rotation: `[x,y,z,R00,R10,R20,R01,R11,R21,gripper]`；双臂则左右各一段，共 20 维。

`cartesian` action 会先转换为库内统一 pose7 格式 `[x,y,z,qw,qx,qy,qz]`，再交给各机器人驱动执行。

推理 I/O 可以切换为固定 32 维 canonical 协议：

```yaml
inference:
  action_space: cartesian
  policy_format: canonical
  canonical_dim: 32
```

该模式只支持 `cartesian`。客户端发送的 state 会补 0 到 32 维；server 返回的
action chunk 必须是 `[chunk_size, 32]`。单臂只取前 10 维，双臂只取前 20 维，
再继续走现有 rot6d → pose7 转换和安全检查。

可选开启 chunk 接缝平滑:

```yaml
inference:
  action_space: cartesian   # 或 joint_position
  latency_compensation: true
  smooth:
    enabled: true
    overlap_steps: 8
  async:
    enabled: false
    obs_fps: null
    max_latency_steps: 8
```

`latency_compensation` 会根据推理耗时跳过过期 action；平滑会在新旧 action chunk 的重叠区线性融合，减少 chunk 切换时的跳变。

默认是同步推理: 控制循环阻塞等待 server 返回 chunk，再执行其中一段 action。
设置 `async.enabled: true` 后，后台线程按 `obs_fps` 持续发送最新 obs 并把返回
chunk 融合进平滑队列，主线程按 `fps` 下发 action。

相机和触觉视频设备都可以在 YAML 中配置 `mapped_key`，只影响发给 server 的图像 key。
`inference.enabled_cameras` 是本地图像传感器白名单，可引用 `cameras` 和 `tactile` 下的 key；
未配置时默认发送全部图像传感器。

```yaml
cameras:
  third_view:
    type: realsense
    mapped_key: observation/image

tactile:
  wrist_tactile:
    type: opencv
    mapped_key: observation/tactile_image
```

## 网络错误处理

- 发送/接收超时自动重试 (默认 3 次)
- ZMQ REQ socket 超时后状态损坏时自动重建连接
- 服务器返回 `{"status": "error"}` 时抛出 `RuntimeError`
- 所有重试耗尽后抛出 `ConnectionError`

## 依赖

Server 和 Client 只依赖: `zmq`, `msgpack`, `numpy`, `cv2` — 常见 ML 环境默认可用。
