"""手眼标定 — 单脚本同时处理 eye-in-hand (腕部) 与 eye-to-hand (静态第三视角)。

每台相机在 YAML 里通过 ``calibration_role`` 指定身份:
    - ``eye_in_hand``   —— 装在夹爪上，求 T_EC (cam → gripper)
    - ``eye_to_hand``   —— 静态俯视，求 T_BC (cam → base)

流程:
    1. 在工作区固定放置棋盘格标定板
    2. 运行此脚本，三路相机画面会拼接显示
    3. 手动拖动机械臂换姿态，平移 > 阈值 或 旋转 > 阈值 且 任一相机检测到棋盘格 → 自动采集
    4. 建议采 15~25 个样本，覆盖不同角度/距离，并尽量保证两台第三视角也能多次看到板
    5. 按 q 结束，自动按角色分别求解并保存

输出:
    T_EC_<cam>.json/.npy  对应每台 eye_in_hand 相机
    T_BC_<cam>.json/.npy  对应每台 eye_to_hand 相机

用法:
    # 棋盘格 11x8 方格、方格 30mm
    python Example/calibrate_hand_eye.py Config/rizon4_example.yaml \
        --cols 11 --rows 8 --square 0.030

    # 仅生成可打印棋盘格图片
    python Example/calibrate_hand_eye.py --print-board board.png \
        --cols 11 --rows 8

机器人位姿走 ``robot.observe().eef_pose`` ([x,y,z,qw,qx,qy,qz]，对 Flexiv / ARX5 通用)。
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Core import load_config, setup_run_logger
from Robot import BaseRobot
from Sensor.rgb_camera.realsense_camera import RealSenseCamera

log = logging.getLogger(__name__)

_running = True


def _sig(_s, _f):
    global _running
    _running = False


signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)


# ═══════════════════════════════════════════════════════════════
#  几何
# ═══════════════════════════════════════════════════════════════


def quat_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = float(np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz))
    w, x, y, z = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def pose_delta(p1: list[float], p2: list[float]) -> tuple[float, float]:
    """两个 EEF 位姿的平移距离 (m) 和旋转角度 (°)。"""
    dt = float(np.linalg.norm(np.array(p1[:3]) - np.array(p2[:3])))
    R1, R2 = quat_to_R(*p1[3:7]), quat_to_R(*p2[3:7])
    cos_a = float(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1))
    return dt, float(np.rad2deg(np.arccos(cos_a)))


def eef_to_T(pose: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """[x,y,z,qw,qx,qy,qz] → (R, t) 表示 gripper→base。"""
    R = quat_to_R(pose[3], pose[4], pose[5], pose[6])
    t = np.array(pose[:3]).reshape(3, 1)
    return R, t


# ═══════════════════════════════════════════════════════════════
#  棋盘格检测
# ═══════════════════════════════════════════════════════════════


def detect_chessboard(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
    square_size: float,
    K: np.ndarray,
    D: np.ndarray,
):
    """检测棋盘格角点并估计板子位姿。返回 (success, rvec, tvec, corners)。"""
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not found or corners is None:
        return False, None, None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    obj_pts = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    obj_pts[:, :2] = (
        np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    )

    ok, rvec, tvec = cv2.solvePnP(obj_pts, corners, K, D)
    if not ok:
        return False, None, None, corners
    return True, rvec, tvec, corners


# ═══════════════════════════════════════════════════════════════
#  求解
# ═══════════════════════════════════════════════════════════════


_HE_METHODS = {
    "PARK":       cv2.CALIB_HAND_EYE_PARK,
    "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
    "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def solve_camera(samples: list[dict], cam_name: str, role: str) -> np.ndarray | None:
    """对单台相机的样本求 T_EC (eye_in_hand) 或 T_BC (eye_to_hand)。

    samples 中每条至少包含: ``eef_pose`` 和 ``rvec``/``tvec``（target → cam）。
    eye_to_hand 走 OpenCV 标准技巧: 把 base↔gripper 翻转后调同一函数，
    解出来的 cam2gripper 在我们的语境里就是 cam2base = T_BC。
    """
    if len(samples) < 3:
        log.warning("%s: 样本不足 (%d < 3)，跳过求解", cam_name, len(samples))
        return None

    R_h, t_h, R_t, t_t = [], [], [], []
    for s in samples:
        R_gb, t_gb = eef_to_T(s["eef_pose"])
        if role == "eye_to_hand":
            # 反演: 喂 base→gripper，结果即 cam→base
            R_h.append(R_gb.T)
            t_h.append((-R_gb.T @ t_gb).reshape(3, 1))
        else:
            R_h.append(R_gb)
            t_h.append(t_gb.reshape(3, 1))

        R_tc, _ = cv2.Rodrigues(np.array(s["rvec"]))
        R_t.append(R_tc)
        t_t.append(np.array(s["tvec"]).reshape(3, 1))

    results: dict[str, dict] = {}
    for name, method in _HE_METHODS.items():
        try:
            R, t = cv2.calibrateHandEye(R_h, t_h, R_t, t_t, method=method)
            det = float(np.linalg.det(R))
            tn = float(np.linalg.norm(t))
            if abs(det - 1.0) > 0.01:
                log.warning("  [%s] %s: det(R)=%.4f 无效", cam_name, name, det)
                continue
            if tn > 2.0:  # eye_to_hand 平移可能 > 0.5m，放宽到 2m
                log.warning("  [%s] %s: |t|=%.3fm 过大", cam_name, name, tn)
                continue
            angle = float(np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.ravel()
            results[name] = {"T": T, "angle": angle, "trans": tn}
            log.info("  [%s] %s: 旋转 %.1f°  平移 [%.4f, %.4f, %.4f]m",
                     cam_name, name, angle, *t.ravel())
        except Exception as e:
            log.warning("  [%s] %s 失败: %s", cam_name, name, e)

    if not results:
        log.error("[%s] 所有方法均失败", cam_name)
        return None

    best = "PARK" if "PARK" in results else min(results, key=lambda k: results[k]["trans"])
    log.info("  [%s] 选择: %s", cam_name, best)
    return results[best]["T"]


# ═══════════════════════════════════════════════════════════════
#  生成棋盘格图片
# ═══════════════════════════════════════════════════════════════


def generate_board(path: str | Path, cols: int, rows: int, dpi: int = 300) -> None:
    """生成可打印的棋盘格图片 (方格 30mm)。"""
    sq = int(dpi * 0.030 / 0.0254)
    margin = sq
    w = cols * sq + 2 * margin
    h = rows * sq + 2 * margin
    img = np.ones((h, w), dtype=np.uint8) * 255
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y0 = margin + r * sq
                x0 = margin + c * sq
                img[y0:y0 + sq, x0:x0 + sq] = 0
    cv2.imwrite(str(path), img)
    log.info("棋盘格已保存: %s (%dx%d 方格, 内角点 %dx%d)",
             path, cols, rows, cols - 1, rows - 1)


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════


class _CamReader:
    """每台相机一个后台线程，持续 drain pipeline 并缓存最近一帧彩色图。

    用顺序阻塞 read_frame() 同时读多台时，慢相机会让快相机的 pipeline
    队列溢出 → 整体失速。每相机独立线程把这个耦合解开。
    """

    def __init__(self, camera: RealSenseCamera) -> None:
        self._cam = camera
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frames = 0
        self._errors = 0
        self._thread = threading.Thread(
            target=self._loop, name=f"camreader-{camera.name}", daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._cam.read_frame()
            except RuntimeError as e:
                self._errors += 1
                log.debug("相机 %s 帧读取异常: %s", self._cam.name, e)
                time.sleep(0.05)
                continue
            color = frame.payload.get("streams", {}).get("color", {})
            data = color.get("data")
            if data is not None:
                with self._lock:
                    self._latest = data
                    self._frames += 1

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    @property
    def stats(self) -> tuple[int, int]:
        return self._frames, self._errors

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _open_calibration_cameras(config) -> dict[str, dict]:
    """实例化并打开所有带 calibration_role 的相机，并为每台启动后台读线程。

    返回 name → {cam, reader, role, K, D}。
    """
    cams: dict[str, dict] = {}
    for name, cfg in config.cameras.items():
        if cfg.calibration_role is None:
            continue
        cam = RealSenseCamera.from_config(name, cfg)
        cam.open()
        K, D = cam.intrinsics()
        reader = _CamReader(cam)
        reader.start()
        cams[name] = {
            "cam": cam, "reader": reader,
            "role": cfg.calibration_role, "K": K, "D": D,
        }
        log.info("相机 %-15s 角色=%s 内参 fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                 name, cfg.calibration_role, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    if not cams:
        raise RuntimeError(
            "配置中无任何相机带 calibration_role 字段，"
            "请在 cameras.<name>.calibration_role 设为 eye_in_hand 或 eye_to_hand"
        )
    # 等首帧到位 (最多 3s)，并报告每台的状态。
    log.info("等待相机首帧...")
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if all(info["reader"].latest() is not None for info in cams.values()):
            break
        time.sleep(0.05)
    for n, info in cams.items():
        f, e = info["reader"].stats
        ok = "OK" if info["reader"].latest() is not None else "NO FRAME"
        log.info("  %-15s frames=%d errors=%d  %s", n, f, e, ok)
    return cams


def _annotate(
    bgr: np.ndarray,
    pattern_size: tuple[int, int],
    K: np.ndarray, D: np.ndarray,
    ok: bool, rvec, tvec, corners,
    *,
    cam_name: str, role: str, n_seen: int, square_size: float,
) -> np.ndarray:
    disp = bgr.copy()
    if ok:
        cv2.drawChessboardCorners(disp, pattern_size, corners, True)
        cv2.drawFrameAxes(disp, K, D, rvec, tvec, square_size * 3)
    color_bg = (0, 255, 0) if ok else (0, 0, 255)
    cv2.putText(disp, f"{cam_name} [{role}]", (10, 25), 0, 0.6, (255, 255, 0), 2)
    cv2.putText(disp, f"seen: {n_seen}  {'OK' if ok else '--'}",
                (10, 50), 0, 0.55, color_bg, 2)
    return disp


def run_calibration(
    config_path: str,
    *,
    cols: int = 11,
    rows: int = 8,
    square_size: float = 0.030,
    min_trans: float = 0.02,
    min_rot: float = 5.0,
    save_dir: str = "./calib_data",
    output_dir: str = ".",
) -> None:
    global _running

    config = load_config(config_path)
    pattern_size = (cols - 1, rows - 1)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 1. 机器人 ──
    log.info("[1/3] 创建机器人 (只读位姿)...")
    robot = BaseRobot.from_config(config_path)
    robot.connect()

    try:
        if robot.is_fault():
            log.info("检测到故障，正在清除...")
            robot.clear_fault()
            time.sleep(2.0)
        robot.enable()
        if not robot.wait_until_operational(timeout_s=30.0):
            raise RuntimeError("机器人未能在超时时间内变为 operational")

        # ── 2. 相机 ──
        log.info("[2/3] 打开标定相机...")
        cams = _open_calibration_cameras(config)
        cam_names = list(cams.keys())  # 固定顺序
        n_seen_per_cam: dict[str, int] = {n: 0 for n in cam_names}

        try:
            samples: list[dict] = []
            last_eef: list[float] | None = None

            roles_summary = ", ".join(f"{n}={cams[n]['role']}" for n in cam_names)
            log.info("")
            log.info("=" * 64)
            log.info("  手眼标定 (多相机)  %s", roles_summary)
            log.info("  棋盘格 %dx%d (内角点 %dx%d, 方格 %.0fmm)",
                     cols, rows, *pattern_size, square_size * 1000)
            log.info("")
            log.info("  自动采样: 平移 > %.0fcm 或 旋转 > %.0f° 且 任一相机检测到板",
                     min_trans * 100, min_rot)
            log.info("  s = 强制采集    q = 结束并求解")
            log.info("=" * 64)

            # ── 3. 采集循环 ──
            while _running:
                try:
                    eef = list(robot.observe().eef_pose)
                except Exception as e:
                    log.warning("机器人位姿读取失败: %s", e)
                    time.sleep(0.05)
                    continue
                if len(eef) < 7:
                    log.warning("eef_pose 长度异常 (%d)，跳过该帧", len(eef))
                    time.sleep(0.05)
                    continue

                per_cam: dict[str, dict] = {}
                for name in cam_names:
                    info = cams[name]
                    bgr = info["reader"].latest()
                    if bgr is None:
                        per_cam[name] = {"bgr": None, "ok": False}
                        continue
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    ok, rvec, tvec, corners = detect_chessboard(
                        gray, pattern_size, square_size, info["K"], info["D"],
                    )
                    per_cam[name] = {
                        "bgr": bgr, "ok": ok,
                        "rvec": rvec, "tvec": tvec, "corners": corners,
                    }

                moved = True
                dt_text = ""
                if last_eef is not None:
                    dt, dr = pose_delta(last_eef, eef)
                    dt_text = f"delta: {dt*100:.1f}cm / {dr:.1f}deg"
                    moved = (dt > min_trans) or (dr > min_rot)
                any_seen = any(p["ok"] for p in per_cam.values())

                key = cv2.waitKey(1) & 0xFF
                force = (key == ord("s"))

                if any_seen and (moved or force):
                    sample_id = len(samples) + 1
                    detections: dict[str, dict] = {}
                    for name in cam_names:
                        p = per_cam[name]
                        if p.get("ok"):
                            detections[name] = {
                                "rvec": p["rvec"].tolist(),
                                "tvec": p["tvec"].tolist(),
                            }
                            n_seen_per_cam[name] += 1
                            cv2.imwrite(
                                str(save_path / f"{name}_sample_{sample_id:03d}.png"),
                                p["bgr"],
                            )
                    samples.append({"eef_pose": eef, "detections": detections})
                    last_eef = eef
                    seen_str = ",".join(n for n in cam_names if per_cam[n]["ok"])
                    log.info("样本 #%d EEF=[%.3f,%.3f,%.3f]  cams=[%s]  %s",
                             sample_id, *eef[:3], seen_str,
                             "手动" if force else "自动")

                # 拼接显示
                tiles = []
                for name in cam_names:
                    p = per_cam[name]
                    if p.get("bgr") is None:
                        tile = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(tile, f"{name}: NO FRAME", (10, 240),
                                    0, 0.7, (0, 0, 255), 2)
                    else:
                        info = cams[name]
                        tile = _annotate(
                            p["bgr"], pattern_size, info["K"], info["D"],
                            p["ok"], p.get("rvec"), p.get("tvec"), p.get("corners"),
                            cam_name=name, role=info["role"],
                            n_seen=n_seen_per_cam[name],
                            square_size=square_size,
                        )
                    tiles.append(tile)
                # 高度对齐再 hstack
                h = min(t.shape[0] for t in tiles)
                tiles = [t if t.shape[0] == h
                         else cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h))
                         for t in tiles]
                canvas = np.hstack(tiles)

                cv2.putText(canvas, f"Total samples: {len(samples)}  {dt_text}",
                            (10, canvas.shape[0] - 15), 0, 0.6, (0, 255, 255), 2)
                cv2.putText(canvas, "q=quit  s=capture",
                            (canvas.shape[1] - 240, canvas.shape[0] - 15),
                            0, 0.6, (200, 200, 200), 2)
                cv2.imshow("Calibration", canvas)

                if key == ord("q"):
                    break
            cv2.destroyAllWindows()
        finally:
            # 先停后台读线程，再关 pipeline。
            for info in cams.values():
                try:
                    info["reader"].stop()
                except Exception:
                    log.warning("停止相机线程失败", exc_info=True)
            for info in cams.values():
                try:
                    info["cam"].close()
                except Exception:
                    log.warning("关闭相机失败", exc_info=True)
    finally:
        robot.disconnect()

    # ── 落盘原始数据 ──
    with open(save_path / "samples.json", "w") as f:
        json.dump(samples, f, indent=2)
    log.info("采集数据: %s/samples.json (%d 样本)", save_path, len(samples))

    # ── 求解 & 保存 ──
    log.info("")
    log.info("=" * 64)
    log.info("  求解 (按相机角色)")
    log.info("=" * 64)

    for name in cam_names:
        info = cams[name]
        role = info["role"]
        cam_samples = []
        for s in samples:
            d = s["detections"].get(name)
            if d is None:
                continue
            cam_samples.append({
                "eef_pose": s["eef_pose"],
                "rvec": d["rvec"],
                "tvec": d["tvec"],
            })
        log.info("[%s]  role=%s  有效样本=%d", name, role, len(cam_samples))

        T = solve_camera(cam_samples, name, role)
        if T is None:
            continue

        R = T[:3, :3]
        t = T[:3, 3]
        angle = float(np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

        prefix = "T_EC" if role == "eye_in_hand" else "T_BC"
        log.info("  → %s (%s):", prefix, "cam→gripper" if role == "eye_in_hand" else "cam→base")
        log.info("    旋转: %.1f°", angle)
        log.info("    平移: [%.4f, %.4f, %.4f] m", *t)
        for row in T:
            log.info("    [%s]", ", ".join(f"{v:8.4f}" for v in row))

        out_json = out_path / f"{prefix}_{name}.json"
        result = {
            f"{prefix}": T.tolist(),
            "role": role,
            "camera": name,
            "rotation_deg": angle,
            "translation_m": t.tolist(),
            "n_samples": len(cam_samples),
            "board": {"cols": cols, "rows": rows, "square_size_m": square_size},
        }
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        np.save(str(out_json.with_suffix(".npy")), T)
        log.info("  已保存: %s / %s", out_json, out_json.with_suffix(".npy"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="手眼标定 (多相机, 棋盘格)")
    p.add_argument("config", nargs="?", help="YAML 配置文件")
    p.add_argument("--print-board", metavar="PATH", help="生成棋盘格图片")
    p.add_argument("--output-dir", default=".",
                   help="结果输出目录 (默认当前目录)")
    p.add_argument("--save-dir", default="./calib_data",
                   help="原始样本图像与 samples.json 保存目录")
    p.add_argument("--cols", type=int, default=11, help="方格列数")
    p.add_argument("--rows", type=int, default=8, help="方格行数")
    p.add_argument("--square", type=float, default=0.030, help="方格边长 (米)")
    p.add_argument("--min-trans", type=float, default=0.02, help="采样平移阈值 (米)")
    p.add_argument("--min-rot", type=float, default=5.0, help="采样旋转阈值 (度)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.print_board:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
        generate_board(args.print_board, args.cols, args.rows)
        return

    if not args.config:
        sys.exit("错误: 请提供 YAML 配置文件，或用 --print-board 生成棋盘格")

    setup_run_logger(__file__, args.config)
    run_calibration(
        args.config,
        cols=args.cols,
        rows=args.rows,
        square_size=args.square,
        min_trans=args.min_trans,
        min_rot=args.min_rot,
        save_dir=args.save_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
