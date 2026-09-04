# Example

硬件示例脚本，按机器人平台组织。

## 目录

```
Example/
└── flexiv/                         # Flexiv Rizon 系列机械臂
    ├── infer_client.py             # 配置驱动的推理控制循环 (核心)
    ├── go_home.py                  # 回 Home 位置
    ├── gripper.py                  # 夹爪开合控制
    ├── probe.py                    # 连接探针 — 轮询状态
    ├── replay_parquet.py           # 从 parquet 文件回放动作
    └── realsense_visualize.py      # RealSense 相机可视化
```

## 快速开始

```bash
# 推理控制循环 (核心用法)
python Example/flexiv/infer_client.py Config/rizon4_example.yaml
python Example/flexiv/infer_client.py Config/rizon4_example.yaml --dry-run

# 基础操作
python Example/flexiv/probe.py Rizon4-063609 --enable
python Example/flexiv/go_home.py Rizon4-063609
python Example/flexiv/gripper.py Rizon4-063609 open close

# 数据回放
python Example/flexiv/replay_parquet.py Rizon4-063609 data.parquet

# 相机调试
python Example/flexiv/realsense_visualize.py
```

## 手眼标定 (棋盘格)

`Example/calibrate_hand_eye.py` 通过棋盘格标定板求解 `T_EC`（相机相对夹爪的位姿）。
脚本走通用接口：`load_config` 加载 YAML，`BaseRobot.from_config` 创建机械臂并读取
`observe().eef_pose`（[x,y,z,qw,qx,qy,qz]，对 Flexiv / ARX5 通用），
`RealSenseCamera.intrinsics()` 取相机内参。

```bash
# 1) 生成可打印棋盘格（11x8 方格、方格 30mm）
python Example/calibrate_hand_eye.py --print-board board.png --cols 11 --rows 8

# 2) 固定棋盘格，运行标定（采集 + 求解一体）
python Example/calibrate_hand_eye.py Config/rizon4_example.yaml \
    --cols 11 --rows 8 --square 0.030

# 默认使用配置中名为 main_cam 的相机；可用 --camera <name> 指定
```

交互：手动拖动机械臂换姿态，平移 > 2cm 或旋转 > 5° 且检测到棋盘格即自动采样；
按 `s` 强制采样、`q` 结束。建议采 15~25 个样本，覆盖不同角度与距离。

输出：

| 文件 | 含义 |
|------|------|
| `T_EC_calibration.json` | 求解结果（含 4×4 矩阵、旋转/平移、样本数、棋盘参数） |
| `T_EC_calibration.npy`  | 同一个 4×4 `T_EC`，便于 `numpy.load` 直接读取 |
| `calib_data/sample_*.png` | 采样原始图像 |
| `calib_data/samples.json` | 采样 EEF 位姿 + 板子位姿原始数据 |

