"""推理客户端。

两种使用方式:

  方式 1 — 获取完整 chunk::

      client = InferenceClient("192.168.50.225:5555")
      with client:
          actions = client.predict_chunk(images, state)  # list[list[float]]
          for action in actions:
              robot.act(Action(ActionSpace.JOINT_POSITION, action[:7]))

  方式 2 — 逐帧获取 (自动管理队列)::

      client = InferenceClient("192.168.50.225:5555")
      with client:
          while running:
              action = client.get_action(images, state, prompt="pick up the cup")
              robot.act(Action(ActionSpace.JOINT_POSITION, action[:7]))

  方式 2 每次调用都存储最新观测。队列用完时自动用最新观测请求新 chunk。

  也可以直接传入 Observation 对象::

      from Core import Observation
      obs = Observation(images=images, state=state, prompt="pick up the cup")
      actions = client.predict_chunk_from_obs(obs)
"""
from __future__ import annotations

import logging
import uuid
from collections import deque
from typing import Any

import cv2
import msgpack
import numpy as np
import zmq

logger = logging.getLogger(__name__)
FLOWDAGGER_PROTOCOL_VERSION = 3


def _resize_with_pad(img: np.ndarray, target_h: int = 224, target_w: int = 224) -> np.ndarray:
    """Match openpi_client.image_tools.resize_with_pad for one HWC image."""
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


class InferenceClient:
    """ZMQ 推理客户端。

    Args:
        server_addr: 服务器地址，如 "192.168.50.225:5555"
        jpeg_quality: JPEG 压缩质量 (0-100)
        recv_timeout_ms: 接收超时 (ms)，首次推理较慢建议 ≥ 30000
        send_timeout_ms: 发送超时 (ms)
        max_retries: 网络错误自动重试次数
    """

    def __init__(
        self,
        server_addr: str,
        *,
        jpeg_quality: int = 90,
        recv_timeout_ms: int = 30000,
        send_timeout_ms: int = 5000,
        max_retries: int = 3,
    ) -> None:
        self._server_addr = server_addr
        self._jpeg_quality = jpeg_quality
        self._recv_timeout_ms = recv_timeout_ms
        self._send_timeout_ms = send_timeout_ms
        self._max_retries = max_retries

        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._connected = False

        # 方式 2 的队列和存储
        self._action_queue: deque[list[float]] = deque()
        self._latest_images: dict[str, np.ndarray] = {}
        self._latest_state: list[float] = []
        self._latest_prompt: str = ""
        self._latest_extra: dict[str, Any] = {}
        self._flow_client_session_id = uuid.uuid4().hex
        self._flow_server_session_id: str | None = None
        self._flow_base_model_id: str | None = None
        self._flow_policy_version: int | None = None
        self._flow_request_generation = 0
        self._flow_episode_id: str | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── 生命周期 ──

    def connect(self) -> None:
        """连接到推理服务器。"""
        if self._connected:
            return
        self._context = zmq.Context()
        self._create_socket()
        self._connected = True
        logger.info("已连接到推理服务器: %s", self._server_addr)

    def close(self) -> None:
        """断开连接并释放资源。"""
        if not self._connected:
            return
        try:
            if self._socket is not None:
                self._socket.close()
                self._socket = None
            if self._context is not None:
                self._context.term()
                self._context = None
        finally:
            self._connected = False
            self._action_queue.clear()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    # ── 方式 1: 获取完整 chunk ──

    def predict_chunk(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray | list[float],
        *,
        prompt: str = "",
        extra: dict[str, Any] | None = None,
        episode_id: str | None = None,
        step_id: int | None = None,
        policy_version: int | None = None,
        request_generation: int | None = None,
    ) -> list[list[float]]:
        """发送观测，获取完整 action chunk。

        Args:
            images: {cam_name: BGR numpy (H,W,3)}
            state: 机器人状态向量
            prompt: 语言指令 (VLA 等模型使用)
            extra: 扩展输入 (点云、力等)

        Returns:
            action chunk: N x action_dim
        """
        self._ensure_connected()
        payload = self._encode_observation(images, state, prompt=prompt, extra=extra)
        if episode_id is not None:
            payload["episode_id"] = episode_id
        if step_id is not None:
            payload["step_id"] = int(step_id)
        if policy_version is not None:
            payload["policy_version"] = int(policy_version)
        if request_generation is not None:
            payload["request_generation"] = int(request_generation)
            self._flow_request_generation = int(request_generation)
        self._add_flow_context(payload)
        resp = self._request(payload)
        return resp["actions"]

    def predict_chunk_response(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray | list[float],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return the full FlowDAgger response for stale-result validation."""
        payload = self._encode_observation(
            images, state,
            prompt=kwargs.pop("prompt", ""),
            extra=kwargs.pop("extra", None),
        )
        for key in ("episode_id", "step_id", "policy_version", "request_generation"):
            value = kwargs.pop(key, None)
            if value is not None:
                payload[key] = int(value) if key != "episode_id" else value
                if key == "request_generation":
                    self._flow_request_generation = int(value)
        if kwargs:
            raise TypeError(f"unexpected predict arguments: {sorted(kwargs)}")
        self._add_flow_context(payload)
        return self._request(payload)

    def predict_chunk_from_obs(self, obs: Any) -> list[list[float]]:
        """从 Observation 对象获取完整 action chunk。

        Args:
            obs: Core.Observation 实例

        Returns:
            action chunk: N x action_dim
        """
        return self.predict_chunk(
            images=obs.images,
            state=obs.state,
            prompt=obs.prompt,
            extra=obs.extra if obs.extra else None,
        )

    # ── 方式 2: 逐帧获取 ──

    def get_action(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray | list[float],
        *,
        prompt: str = "",
        extra: dict[str, Any] | None = None,
    ) -> list[float]:
        """获取单帧 action。

        每次调用都存储最新观测。队列为空时，
        用最新存储的观测自动请求新 chunk。

        Args:
            images: {cam_name: BGR numpy (H,W,3)}
            state: 机器人状态向量
            prompt: 语言指令
            extra: 扩展输入

        Returns:
            单帧 action: action_dim 维列表
        """
        # 始终存储最新观测
        self._latest_images = images
        self._latest_state = (
            state.tolist() if isinstance(state, np.ndarray) else list(state)
        )
        self._latest_prompt = prompt
        self._latest_extra = extra or {}

        if not self._action_queue:
            actions = self.predict_chunk(
                self._latest_images,
                self._latest_state,
                prompt=self._latest_prompt,
                extra=self._latest_extra or None,
            )
            self._action_queue.extend(actions)

        return self._action_queue.popleft()

    @property
    def actions_remaining(self) -> int:
        """队列中剩余的 action 数量。"""
        return len(self._action_queue)

    def clear_queue(self) -> None:
        """清空 action 队列，下次 get_action 时会请求新 chunk。"""
        self._action_queue.clear()

    # ── 控制命令 ──

    def reset(self) -> dict[str, Any]:
        """通知服务器重置策略。同时清空本地 action 队列。"""
        self._ensure_connected()
        resp = self._request({"cmd": "reset"})
        self._action_queue.clear()
        logger.info("策略已重置")
        return resp

    def flowdagger_episode_start(
        self,
        episode_id: str,
        *,
        prompt: str = "",
        shadow_mode: bool = False,
        run_stage: str = "demonstration",
    ) -> dict[str, Any]:
        health = self.flowdagger_health()
        self._flow_server_session_id = str(health["server_session_id"])
        self._flow_base_model_id = str(health["base_model_id"])
        requested_policy_version = int(health.get("policy_version", 0))
        reply = self._request({
            "cmd": "episode_start",
            "episode_id": episode_id,
            "prompt": prompt,
            "shadow_mode": bool(shadow_mode),
            "run_stage": run_stage,
            "protocol_version": FLOWDAGGER_PROTOCOL_VERSION,
            "client_session_id": self._flow_client_session_id,
            "server_session_id": self._flow_server_session_id,
            "base_model_id": self._flow_base_model_id,
            "requested_policy_version": requested_policy_version,
        })
        if str(reply.get("server_session_id")) != self._flow_server_session_id:
            raise RuntimeError("FlowDAgger server restarted during episode_start")
        self._flow_episode_id = episode_id
        self._flow_policy_version = int(reply["policy_version"])
        return reply

    def flowdagger_intervention(
        self, episode_id: str, *, active: bool,
        step_id: int | None = None,
        images: dict[str, np.ndarray] | None = None,
        state: np.ndarray | list[float] | None = None,
        prompt: str = "",
    ) -> dict[str, Any]:
        payload = {
            "cmd": "intervention_start" if active else "intervention_stop",
            "episode_id": episode_id,
        }
        if active:
            if step_id is None or images is None or state is None:
                raise ValueError("intervention_start requires boundary observation")
            raw_images: dict[str, bytes] = {}
            for key, image in images.items():
                ok, encoded = cv2.imencode(
                    ".jpg", np.asarray(image),
                    [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
                )
                if not ok:
                    raise ValueError(f"failed to encode boundary image {key!r}")
                raw_images[key] = encoded.tobytes()
            payload.update({
                "step_id": int(step_id),
                "state": state.tolist() if isinstance(state, np.ndarray) else list(state),
                "prompt": prompt,
                "raw_images": raw_images,
                "request_generation": self._flow_request_generation,
            })
        self._add_flow_context(payload)
        return self._request(payload)

    def flowdagger_expert_step(
        self,
        episode_id: str,
        step_id: int,
        images: dict[str, np.ndarray],
        state: np.ndarray | list[float],
        *,
        prompt: str = "",
        gripper_event: str | None = None,
    ) -> dict[str, Any]:
        raw_images: dict[str, bytes] = {}
        for key, image in images.items():
            value = np.asarray(image)
            if value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError(f"expert image {key!r} must be HWC BGR, got {value.shape}")
            ok, encoded = cv2.imencode(
                ".jpg",
                value,
                [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
            )
            if not ok:
                raise ValueError(f"failed to encode expert image {key!r}")
            raw_images[key] = encoded.tobytes()
        payload: dict[str, Any] = {
            "cmd": "expert_step",
            "episode_id": episode_id,
            "step_id": int(step_id),
            "state": state.tolist() if isinstance(state, np.ndarray) else list(state),
            "prompt": prompt,
            "raw_images": raw_images,
        }
        if gripper_event:
            payload["gripper_event"] = gripper_event
        self._add_flow_context(payload)
        return self._request(payload)

    def flowdagger_episode_end(
        self,
        episode_id: str,
        task_outcome: str,
        *,
        control_metrics: dict[str, int | float] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cmd": "episode_end",
            "episode_id": episode_id,
            "task_outcome": task_outcome,
        }
        if control_metrics is not None:
            payload["control_metrics"] = dict(control_metrics)
        self._add_flow_context(payload)
        reply = self._request(payload)
        self._flow_episode_id = None
        return reply

    def flowdagger_train_status(self) -> dict[str, Any]:
        return self._request({"cmd": "train_status"})

    def flowdagger_health(self) -> dict[str, Any]:
        """Return server mode/specification without opening an episode."""
        return self._request({"cmd": "health"})

    def flowdagger_demo_start(self, steering_version: str | int = "active") -> dict[str, Any]:
        """Pin a steering checkpoint for inference without opening an episode."""
        return self._request({
            "cmd": "demo_start",
            "steering_version": str(steering_version),
        })

    def flowdagger_demo_stop(self) -> dict[str, Any]:
        return self._request({"cmd": "demo_stop"})

    def flowdagger_heartbeat(self, episode_id: str) -> dict[str, Any]:
        payload = {"cmd": "control_heartbeat", "episode_id": episode_id}
        self._add_flow_context(payload)
        reply = self._request(payload)
        if str(reply.get("server_session_id")) != self._flow_server_session_id:
            raise RuntimeError("FlowDAgger server session changed")
        return reply

    def _add_flow_context(self, payload: dict[str, Any]) -> None:
        if self._flow_episode_id is None and payload.get("cmd") == "predict":
            return
        if self._flow_client_session_id:
            payload["client_session_id"] = self._flow_client_session_id
        if self._flow_server_session_id:
            payload["server_session_id"] = self._flow_server_session_id

    # ── 内部实现 ──

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "未连接到推理服务器，请先调用 connect() 或使用 with 语句"
            )

    def _create_socket(self) -> None:
        """创建新的 REQ socket。"""
        assert self._context is not None
        if self._socket is not None:
            self._socket.close()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._send_timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self._server_addr}")

    def _reconnect(self) -> None:
        """重建 socket 连接 (ZMQ REQ 在超时后状态会损坏)。"""
        logger.warning("重建连接: %s", self._server_addr)
        self._create_socket()

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """发送请求并接收响应，含重试逻辑。"""
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                assert self._socket is not None
                self._socket.send(msgpack.packb(payload))
                raw = self._socket.recv()
                resp = msgpack.unpackb(raw, raw=False)

                if resp.get("status") == "error":
                    raise RuntimeError(
                        f"服务器返回错误: {resp.get('message', '未知')}"
                    )
                return resp

            except zmq.Again as e:
                last_error = e
                logger.warning(
                    "请求超时 (第 %d/%d 次): %s",
                    attempt, self._max_retries, e,
                )
                self._reconnect()

            except zmq.ZMQError as e:
                last_error = e
                logger.warning(
                    "ZMQ 错误 (第 %d/%d 次): %s",
                    attempt, self._max_retries, e,
                )
                self._reconnect()

        raise ConnectionError(
            f"推理请求失败，已重试 {self._max_retries} 次: {last_error}"
        )

    def _encode_observation(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray | list[float],
        *,
        prompt: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """将观测编码为 msgpack 可序列化的字典。"""
        state_list = np.asarray(state, dtype=np.float32).reshape(-1).tolist()
        payload: dict[str, Any] = {
            "cmd": "predict",
            "state": state_list,
        }

        # 语言指令
        if prompt:
            payload["prompt"] = prompt

        # 扩展输入 (确保可序列化)
        if extra:
            payload["extra"] = extra

        # Match the old websocket client: BGR -> RGB, resize_with_pad(224,224), then HWC -> CHW.
        # msgpack cannot serialize ndarray directly, so send nested uint8 lists.
        for cam_name, img_bgr in images.items():
            rgb = cv2.cvtColor(np.asarray(img_bgr), cv2.COLOR_BGR2RGB)
            rgb = _resize_with_pad(rgb, 224, 224)
            chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.uint8)
            payload[cam_name] = chw.tolist()

        return payload
