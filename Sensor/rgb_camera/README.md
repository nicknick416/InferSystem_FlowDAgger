# RGB Camera — RGB(-D) 相机

## 设计原则

相机在推理系统中的本质: **打开设备 → 读取图像帧 → 关闭设备**。
配置 (分辨率、帧率、设备参数) 在构造时确定，运行时基本不变。
基类只管生命周期和统一输出，所有硬件差异封装在驱动子类中。

## 架构

```
BaseSensor (Sensor/base.py)
  └── BaseRGBCamera (rgb_camera/base.py)       — 生命周期 + 工厂 + 参数管理
        ├── RealSenseCamera (realsense_camera.py) — Intel RealSense D435/D405 驱动
        └── MockRGBCamera (mock_camera.py)        — 测试用 Mock
```

## 配置 (YAML)

在系统配置文件的 `cameras:` 段声明:

```yaml
cameras:
  main_cam:
    type: realsense                   # 驱动类型，对应 @BaseRGBCamera.register("realsense")
    serial_number: "409122273675"
    enable_depth: true
    align_depth_to_color: true
    streams:
      color:
        width: 640
        height: 480
        fps: 30
      depth:
        width: 640
        height: 480
        fps: 30
    params:
      exposure.auto: false
      exposure.value: 200
      gain: 64
```

## 用法

```python
from Sensor.rgb_camera import RealSenseCamera, BaseRGBCamera

# 直接构造
cam = RealSenseCamera("main_cam", serial_number="409122273675")
with cam:
    frame = cam.read_frame()
    color = frame.payload["streams"]["color"]["data"]   # numpy BGR array

# 从配置字典创建 (上层系统调用)
cam = BaseRGBCamera.from_config("main_cam", cfg_dict)
```

## SensorFrame 输出格式

```python
SensorFrame(
    sensor_name="main_cam",
    sensor_type="rgb_camera",
    timestamp=...,
    payload={
        "frame_id": 1,
        "streams": {
            "color": {
                "data": np.ndarray,     # (H, W, 3) BGR uint8
                "encoding": "bgr8",
                "width": 640,
                "height": 480,
            },
            "depth": {                  # 仅 enable_depth=True 时存在
                "data": np.ndarray,     # (H, W) uint16
                "encoding": "z16",
                "width": 640,
                "height": 480,
                "unit": "meter",
                "center_distance": 1.23,
            },
        },
    },
)
```

## 添加新驱动

```python
from Sensor.rgb_camera.base import BaseRGBCamera

@BaseRGBCamera.register("usb_camera")
class USBCamera(BaseRGBCamera):
    @classmethod
    def _from_config_dict(cls, name, cfg):
        return cls(name=name, ...)

    def _open_device(self) -> None: ...
    def _close_device(self) -> None: ...
    def _grab_streams(self) -> dict:
        return {"color": {"data": ..., "encoding": "bgr8", ...}}
```

注册后即可在 YAML 中使用 `type: usb_camera`。

## 规约

1. **生命周期**: `open()` → `read_frame()` ... → `close()`，支持 `with` 语句
2. **输出统一**: 所有驱动输出 `SensorFrame`，图像在 `payload["streams"]["color"]["data"]`
3. **注册机制**: `@BaseRGBCamera.register("type")` 注册，`from_config()` 分发
4. **子类三件事**: 实现 `_open_device` / `_close_device` / `_grab_streams`
5. **参数简洁**: `set_param(key, value)` 直接写入，硬件 SDK 自行校验
6. **无语义泄漏**: 驱动只提供原始数据，不解释 "障碍物" 等业务含义
