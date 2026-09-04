# Tactile — 视触觉传感器

## 设计原则

视触觉传感器本质是一个摄像头，捕捉弹性接触面的形变图像来感知力/接触。
因此底层使用 OpenCV VideoCapture 读取，与 RGB 相机共享 `BaseSensor → SensorFrame` 数据通路。

与 RGB 相机的区别:
- 单路视频流 (`tactile`)，无 depth
- 不需要复杂参数管理 (曝光/白平衡由传感器硬件固定)
- 语义不同: payload 中的图像是触觉信息，不是场景图像

## 架构

```
BaseSensor (Sensor/base.py)
  └── BaseTactileSensor (tactile/base.py)     — 触觉传感器通用生命周期 + 工厂
        └── OpenCVTactileSensor (tactile/opencv_tactile.py) — cv2.VideoCapture 驱动
```

## 配置 (YAML)

在系统配置文件的 `tactile:` 段声明:

```yaml
tactile:
  wrist_left:
    type: opencv                    # 驱动类型，对应 @BaseTactileSensor.register("opencv")
    device_path: "/dev/v4l/by-path/pci-0000:08:00.3-usb-0:1:1.0-video-index0"
    width: 640
    height: 480
    fps: 30
```

`device_path` 推荐使用 `/dev/v4l/by-path/...`，保证 USB 重插后设备映射稳定。

## 用法

```python
from Sensor.tactile import OpenCVTactileSensor

# 直接构造
sensor = OpenCVTactileSensor("wrist_left", device_path="/dev/video0")
with sensor:
    frame = sensor.read_frame()
    img = frame.payload["streams"]["tactile"]["data"]   # numpy BGR array

# 从配置字典创建 (上层系统调用)
sensor = BaseTactileSensor.from_config("wrist_left", cfg_dict)
```

## SensorFrame 输出格式

```python
SensorFrame(
    sensor_name="wrist_left",
    sensor_type="tactile",
    timestamp=...,
    payload={
        "frame_id": 1,
        "streams": {
            "tactile": {
                "data": np.ndarray,     # (H, W, 3) BGR uint8
                "encoding": "bgr8",
                "width": 640,
                "height": 480,
            },
        },
    },
)
```

## 添加新驱动

```python
from Sensor.tactile.base import BaseTactileSensor

@BaseTactileSensor.register("gelsight")
class GelSightSensor(BaseTactileSensor):
    @classmethod
    def _from_config_dict(cls, name, cfg):
        return cls(name=name, ...)

    def _open_device(self) -> None: ...
    def _close_device(self) -> None: ...
    def _grab_frame(self): ...    # 返回 numpy BGR array
```

注册后即可在 YAML 中使用 `type: gelsight`。

## 规约

1. **生命周期**: `open()` → `read_frame()` ... → `close()`，支持 `with` 语句
2. **输出统一**: 所有驱动输出 `SensorFrame`，触觉图像在 `payload["streams"]["tactile"]["data"]`
3. **注册机制**: `@BaseTactileSensor.register("type")` 注册，`from_config()` 分发
4. **子类三件事**: 实现 `_open_device` / `_close_device` / `_grab_frame`
5. **设备路径**: 使用 `/dev/v4l/by-path/...` 稳定路径，不用 `/dev/video*`
