#!/usr/bin/env python
"""flexiv 单臂 replay —— 最原本的逐帧回放:读一条采集 episode 的 eef 动作(10维),
按采集的 30Hz 逐帧原样发给机器人,不重采样/不匀速/不插值。

复用 ActionDispatcher(rot6d->pose7 + width夹爪 自动处理)。
只保留两样必要的:①首帧从当前位姿 slerp 平滑走到起点(否则首帧猛冲)②空格暂停。

夹爪坑:actions.eef_pose 的 gripper 是 10×米 -> 自动 ×0.1;state 的是正常米 ×1。

用法(手握急停):
  python flexiv_replay.py --episode <ep> --config Config/NeoVTLA/new_vtla_flexiv_example.yaml --source state --execute
"""
import argparse
import csv
import os
import select
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/home/xinzhi/InferSystem")
from Core import Action, ActionSpace, load_config  # noqa: E402
from Robot import BaseRobot  # noqa: E402
from Inference.dispatch import ActionDispatcher  # noqa: E402
from Inference.action_processing import canonicalize_action_values  # noqa: E402

_pause = threading.Event()
_quit = threading.Event()


def _kbd_listener():
    try:
        import termios
        import tty
    except Exception:
        return
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except Exception:
        return
    try:
        tty.setcbreak(fd)
        while not _quit.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if ch == " ":
                if _pause.is_set():
                    _pause.clear(); print("\n▶ 继续", flush=True)
                else:
                    _pause.set(); print("\n⏸ 暂停(空格=继续, q=退出)", flush=True)
            elif ch in ("q", "Q"):
                _quit.set(); print("\n■ 退出中...", flush=True)
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass


def _wait_if_paused():
    """返回是否曾暂停(供继续后重置时间基准,不追赶暂停时间)。"""
    was = False
    while _pause.is_set() and not _quit.is_set():
        was = True
        time.sleep(0.05)
    return was


def _slerp(q0, q1, t):
    """四元数球面插值(标量在前 [w,x,y,z]),大角度也平滑。"""
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return (r / (np.linalg.norm(r) + 1e-9)).tolist()
    th0 = np.arccos(dot)
    s0 = np.sin(th0)
    return ((np.sin((1 - t) * th0) / s0) * q0 + (np.sin(t * th0) / s0) * q1).tolist()


def load_eef(ep, source, grip_scale):
    """读 10 维 eef 动作 [xyz3, rot6d6, grip1]。列名自适应:action=tcp.x/gripper.pos, state=x/gripper。"""
    sub = "actions.eef_pose" if source == "action" else "observation.state.eef_pose"
    f = os.path.join(ep, sub, "data.csv")
    if not os.path.isfile(f):
        raise SystemExit(f"找不到文件: {f}")
    rows = list(csv.DictReader(open(f)))
    if not rows:
        raise SystemExit("空数据")
    hdr = rows[0]
    pre = "tcp." if "tcp.x" in hdr else ""
    cols = [f"{pre}x", f"{pre}y", f"{pre}z"] + [f"{pre}r{i}" for i in range(1, 7)]
    grip_col = "gripper.pos" if "gripper.pos" in hdr else "gripper"
    ts_col = "timestamp_ms" if "timestamp_ms" in hdr else list(hdr.keys())[0]
    missing = [c for c in cols + [grip_col] if c not in hdr]
    if missing:
        raise SystemExit(f"{sub} 缺列 {missing}; 实际列: {list(hdr.keys())}")
    out, ts = [], []
    for r in rows:
        v = [float(r[c]) for c in cols]
        v.append(float(r[grip_col]) * grip_scale)
        out.append(v)
        ts.append(float(r[ts_col]))  # 采集时间戳(ms)
    return out, ts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="/home/xinzhi/InferSystem/Config/NeoVTLA/new_vtla_flexiv_example.yaml")
    p.add_argument("--episode", required=True)
    p.add_argument("--source", choices=("action", "state"), default="state",
                   help="action=遥操指令, state=实测态(默认,更贴合实际轨迹)")
    p.add_argument("--grip-scale", type=float, default=-1.0, help="夹爪缩放(-1=自动:action×0.1,state×1)")
    p.add_argument("--rate", type=float, default=30.0, help="发送帧率Hz(采集是30,原样播就用30)")
    p.add_argument("--approach-s", type=float, default=4.0, help="首帧从当前位姿平滑走到起点的秒数(防猛冲)")
    p.add_argument("--max-steps", type=int, default=0, help="只回放前 N 帧(0=全部)")
    p.add_argument("--execute", action="store_true", help="真动机器人(默认 dry-run)")
    a = p.parse_args()

    grip_scale = a.grip_scale if a.grip_scale >= 0 else (0.1 if a.source == "action" else 1.0)
    cmds, ts = load_eef(a.episode, a.source, grip_scale)
    if a.max_steps > 0:
        cmds = cmds[: a.max_steps]
        ts = ts[: a.max_steps]
    _dur = (ts[-1] - ts[0]) / 1000.0 if len(ts) > 1 else 0.0
    print(f"[时序] 采集时长 {_dur:.1f}s, 平均 {len(cmds)/max(_dur,1e-6):.1f}Hz(按真实 timestamp 回放)")
    arr = np.asarray(cmds, dtype=float)
    print(f"[数据源] source={a.source} grip_scale={grip_scale}  {len(cmds)} 步(原样逐帧,不处理)")
    print(f"[首帧] {[round(float(x), 4) for x in cmds[0]]}")
    print(f"[范围] xyz x[{arr[:,0].min():.3f},{arr[:,0].max():.3f}] "
          f"y[{arr[:,1].min():.3f},{arr[:,1].max():.3f}] z[{arr[:,2].min():.3f},{arr[:,2].max():.3f}]  "
          f"夹爪(米)[{arr[:,9].min():.4f},{arr[:,9].max():.4f}]")

    if not a.execute:
        print("\n=== DRY-RUN:未动机器人。加 --execute 真回放。===")
        return

    print(f"\n[连接] {a.config}")
    config = load_config(a.config)
    robot = BaseRobot.from_config(a.config)
    robot.connect()
    if hasattr(robot, "is_fault") and robot.is_fault():
        robot.clear_fault()
    gripper = robot.create_gripper()      # gripper.connect 需 IDLE,放 go_home 之前
    if gripper is not None:
        gripper.connect()
    dispatcher = ActionDispatcher.from_config(robot, gripper, config)

    print("[起始] go_home + inference_home")
    robot.go_home()
    robot.act(Action(ActionSpace.JOINT_POSITION, list(config.inference.inference_home)))
    time.sleep(2.0)

    dt = 1.0 / a.rate
    kbd = threading.Thread(target=_kbd_listener, daemon=True)
    kbd.start()
    print("[键盘] 空格=暂停/继续, q=退出")
    try:
        # 首帧平滑逼近:xyz smoothstep + quat slerp,从当前位姿慢慢走到起点(防猛冲)
        if a.approach_s > 0:
            c0 = list(canonicalize_action_values(cmds[0], ActionSpace.CARTESIAN))  # 8D [pose7, grip]
            tgt_xyz, tgt_quat, tgt_grip = c0[:3], c0[3:7], float(c0[7])
            cur = list(robot.observe().eef_pose)
            cur_xyz, cur_quat = cur[:3], cur[3:7]
            if gripper is not None:
                try:
                    gripper.move(tgt_grip)
                except Exception:
                    pass
            n_app = max(1, int(a.approach_s * a.rate))
            print(f"[逼近] {a.approach_s}s 平滑走到首帧(手握急停!)...")
            for t in range(n_app):
                _wait_if_paused()
                if _quit.is_set():
                    break
                al = (t + 1) / n_app
                al = al * al * (3 - 2 * al)  # smoothstep
                xyz = [cur_xyz[i] + al * (tgt_xyz[i] - cur_xyz[i]) for i in range(3)]
                quat = _slerp(cur_quat, tgt_quat, al)
                robot.act(Action(ActionSpace.CARTESIAN, xyz + quat))
                time.sleep(dt)

        # 原样逐帧回放:每帧 sleep 到"下一帧的采集间隔",某帧慢了绝不追赶后面(核心:不加快猛冲)
        print(f"[回放] {len(cmds)} 步 · 按采集 timestamp 逐帧(慢了不追赶) ...")
        for k, c in enumerate(cmds):
            _wait_if_paused()
            if _quit.is_set():
                print(f"[中断] step {k}/{len(cmds)}")
                break
            st = robot.observe()
            dispatcher.dispatch(list(c), state=st)  # rot6d->pose7 + width夹爪 自动
            if k % 60 == 0:
                print(f"  step {k}/{len(cmds)}")
            if k + 1 < len(cmds):
                gap = (ts[k + 1] - ts[k]) / 1000.0  # 到下一帧的采集间隔
                gap = min(max(gap, 0.0), 0.2)       # 限合理范围,防异常大/负间隔一次跳太多
                time.sleep(gap)
        print("[完成] go_home")
        robot.go_home()
    finally:
        _quit.set()
        robot.disconnect()


if __name__ == "__main__":
    main()
