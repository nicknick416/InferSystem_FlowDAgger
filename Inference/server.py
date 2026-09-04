"""独立推理服务器。

无 InferSystem 依赖，可直接部署到任何有 GPU 的机器上。

协议 (ZMQ REP, msgpack):

  请求:
    {"cmd": "predict", "state": [float, ...], "prompt": str, "extra": {...},
     "<cam_name>": [[...]], ...}
    {"cmd": "reset"}

  图像字段:
    - 当前 client: RGB uint8 CHW list, shape=(3, 224, 224)
    - 兼容旧 client: bytes(JPEG)，server 会转 RGB、resize/pad 到 224x224，再转 CHW
    - 也兼容 RGB HWC list/array，server 会转为 CHW

  响应:
    {"status": "ok", "actions": [[float, ...], ...], "infer_time_ms": float}
    {"status": "ok"}
    {"status": "error", "message": str}

用法::

    from Inference.server import InferenceServer

    def my_predict(images, state, prompt="", extra=None):
        # 你的模型推理
        return actions_chunk

    server = InferenceServer("tcp://*:5555", predict_fn=my_predict)
    server.run()  # 阻塞运行

    # 或者子类方式
    class MyServer(InferenceServer):
        def predict(self, images, state, prompt="", extra=None):
            return my_model(images, state, prompt)

        def on_reset(self):
            my_model.reset()
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any, Callable

import cv2
import msgpack
import numpy as np
import zmq

logger = logging.getLogger(__name__)

PredictFn = Callable[..., list[list[float]]]
IMAGE_SKIP_KEYS = {"cmd", "state", "prompt", "extra"}


def resize_with_pad(
    img: np.ndarray,
    target_h: int = 224,
    target_w: int = 224,
) -> np.ndarray:
    """Resize an HWC image with zero padding while preserving aspect ratio."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid image shape: {img.shape}")
    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def decode_jpeg_image(buf: bytes | bytearray) -> np.ndarray:
    """JPEG bytes -> HWC uint8 BGR ndarray."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("JPEG 解码失败")
    return img


def _as_array(value: Any) -> np.ndarray | None:
    try:
        return np.asarray(value)
    except (TypeError, ValueError):
        return None


def is_image_payload(value: Any) -> bool:
    """Return True when a msgpack value looks like an image payload."""
    if isinstance(value, (bytes, bytearray)):
        return True

    arr = _as_array(value)
    if arr is None or arr.ndim != 3:
        return False
    return arr.shape[0] == 3 or arr.shape[-1] == 3


def decode_image_payload(value: Any, *, strict: bool = False) -> np.ndarray | None:
    """Decode one image payload to RGB uint8 CHW.

    Supported inputs are legacy JPEG bytes, current client CHW lists, and RGB
    HWC lists/arrays. Invalid payloads return None unless strict=True.
    """
    if isinstance(value, (bytes, bytearray)):
        try:
            bgr = decode_jpeg_image(value)
        except ValueError:
            if strict:
                raise
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = resize_with_pad(rgb, 224, 224)
        return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.uint8)

    arr = _as_array(value)
    if arr is None or arr.ndim != 3:
        return None

    if arr.shape[0] == 3:
        return np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.shape[-1] == 3:
        return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.uint8)
    return None


def image_keys_from_msg(
    msg: dict[str, Any],
    *,
    skip_keys: Iterable[str] = IMAGE_SKIP_KEYS,
) -> list[str]:
    """Return top-level keys that will be treated as image payloads."""
    skip = set(skip_keys)
    return [
        key
        for key, value in msg.items()
        if key not in skip and is_image_payload(value)
    ]


def decode_images_from_msg(
    msg: dict[str, Any],
    *,
    skip_keys: Iterable[str] = IMAGE_SKIP_KEYS,
    strict: bool = False,
) -> dict[str, np.ndarray]:
    """Decode top-level image fields from a client msgpack payload."""
    skip = set(skip_keys)
    images: dict[str, np.ndarray] = {}
    for key, value in msg.items():
        if key in skip:
            continue
        image = decode_image_payload(value, strict=strict)
        if image is not None:
            images[key] = image
    return images


class InferenceServer:
    """ZMQ 推理服务器。

    Args:
        addr: ZMQ 绑定地址，如 "tcp://*:5555"
        predict_fn: 推理回调。签名:
            (images: dict[str, ndarray], state: list[float],
             prompt: str, extra: dict | None) -> list[list[float]]
            images 的 ndarray 为 RGB uint8 CHW。JPEG 旧协议会在 server 侧
            resize/pad 到 224x224 后转为该格式。
            也可通过子类覆盖 predict() 方法替代。
    """

    def __init__(
        self,
        addr: str = "tcp://*:5555",
        predict_fn: PredictFn | None = None,
    ) -> None:
        self._addr = addr
        self._predict_fn = predict_fn
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None

    def predict(
        self,
        images: dict[str, np.ndarray],
        state: list[float],
        prompt: str = "",
        extra: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        """推理入口。子类覆盖此方法，或构造时传入 predict_fn。

        Args:
            images: {cam_name: RGB uint8 CHW ndarray}
            state: 机器人状态向量
            prompt: 语言指令 (可为空)
            extra: 扩展输入 (可为 None)

        Returns:
            action chunk: N x action_dim
        """
        if self._predict_fn is not None:
            return self._predict_fn(images, state, prompt, extra)
        raise NotImplementedError("请传入 predict_fn 或覆盖 predict() 方法")

    def on_reset(self) -> None:
        """收到 reset 命令时调用。子类按需覆盖。"""

    def run(self) -> None:
        """阻塞运行，处理请求直到 KeyboardInterrupt。"""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(self._addr)
        logger.info("推理服务器启动: %s", self._addr)

        try:
            while True:
                raw = self._socket.recv()
                reply = self._handle_request(raw)
                self._socket.send(msgpack.packb(reply))
        except KeyboardInterrupt:
            logger.info("服务器收到中断信号，停止。")
        finally:
            self.close()

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def _handle_request(self, raw: bytes) -> dict[str, Any]:
        try:
            msg = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            return {"status": "error", "message": f"msgpack 解码失败: {e}"}

        cmd = msg.get("cmd", "predict")

        if cmd == "reset":
            self.on_reset()
            logger.info("策略已重置")
            return {"status": "ok"}

        if cmd == "predict":
            return self._handle_predict(msg)

        return {"status": "error", "message": f"未知命令: {cmd}"}

    def _handle_predict(self, msg: dict[str, Any]) -> dict[str, Any]:
        try:
            state = msg.get("state", [])
            prompt = msg.get("prompt", "")
            extra = msg.get("extra")
            images = self._decode_images(msg)

            t0 = time.perf_counter()
            actions = self.predict(images, state, prompt=prompt, extra=extra)
            infer_ms = (time.perf_counter() - t0) * 1000

            return {
                "status": "ok",
                "actions": actions,
                "infer_time_ms": infer_ms,
            }
        except Exception as e:
            logger.error("推理失败: %s", e, exc_info=True)
            return {"status": "error", "message": str(e)}

    @staticmethod
    def _decode_images(msg: dict[str, Any]) -> dict[str, np.ndarray]:
        """从 msgpack 消息中解码图像字段为 RGB uint8 CHW。"""
        return decode_images_from_msg(msg)
