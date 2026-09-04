# Sensor

传感器抽象与硬件驱动。

## 架构

```
BaseSensor (base.py)              — 顶层传感器契约: name + sensor_type + read() → SensorFrame
  ├── BaseRGBCamera (rgb_camera/) — RGB(-D) 相机: 生命周期 + 工厂 + 参数管理
  │     ├── RealSenseCamera       — Intel RealSense 驱动
  │     └── MockRGBCamera         — 测试用 Mock
  └── BaseTactileSensor (tactile/)— 视触觉传感器: 单路视频流 + OpenCV 读取
        └── OpenCVTactileSensor   — cv2.VideoCapture 驱动
```

## 核心契约

所有传感器输出 `SensorFrame`，推理层不关心数据来自哪个厂商或传输方式:

```python
@dataclass
class SensorFrame:
    sensor_name: str
    sensor_type: str          # "rgb_camera" | "tactile" | ...
    timestamp: datetime
    payload: dict[str, Any]   # 传感器特定数据，图像在 payload["streams"][stream_name]["data"]
```

## 统一模式

RGB 相机和触觉传感器共享相同的设计模式:

| 维度 | BaseRGBCamera | BaseTactileSensor |
|------|---------------|-------------------|
| 注册工厂 | `@register("type")` + `from_config()` | `@register("type")` + `from_config()` |
| 上下文管理器 | `with cam:` | `with sensor:` |
| 子类实现 | `_open_device` / `_close_device` / `_grab_streams` | `_open_device` / `_close_device` / `_grab_frame` |
| 参数管理 | `set_param(key, value)` (简单 dict) | 无 (硬件固定) |

## 目录

| 目录 | 说明 |
|------|------|
| `base.py` | 顶层 BaseSensor |
| `rgb_camera/` | RGB(-D) 相机基类与驱动 |
| `tactile/` | 视触觉传感器基类与驱动 |
