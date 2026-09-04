# ARX5 Examples

ARX5 示例脚本，支持单臂 (`arx5`) 与双臂 (`arx5_bimanual`)。
命名与 `DataCollectionSystem` 对齐：优先使用 `interface: can0/can1` 与 `canable_serial`。

## 目录

```text
Example/arx5/
├── go_home.py               # 回 Home
├── probe.py                 # 连接探针，轮询状态
├── gripper.py               # 夹爪开合/目标开口控制
├── replay_parquet.py        # 从 parquet 回放动作
├── replay_eef.py            # 从 DCSv2 episode 回放双臂 EEF 动作
├── single_arm_smoke_test.py # 单臂联调 smoke test
└── bimanual_smoke_test.py   # 双臂联调 smoke test
```

## 配置文件

- 单臂: `Config/arx5_example.yaml`
- 双臂: `Config/arx5_bimanual_example.yaml`

> 运行前请确认 `robot.type`、`model`、`interface`、`canable_serial`、相机序列号、推理地址已按现场设备修改。

### 与 `probe.py` 对应的配置

- **`python Example/arx5/probe.py Config/arx5_example.yaml ...`** 走的是 **单臂** `robot.type: arx5`，读 **`arx5_example.yaml`**，不会加载 `arx5_bimanual_example.yaml`。
- 单臂若 **不** 在 `control` 里写 `gripper_open_readout`，与 DataCollectionSystem 一样使用 **`RobotConfigFactory` 出厂夹爪标定**；`observe()` 仍会返回 7 维（末维为 SDK 的 `gripper_pos`）。`enable_gripper: false` 只表示默认不下发夹爪力控/目标，不改变状态里夹爪读数的来源。

## CAN 接口与命名（同 DataCollectionSystem）

- 控制器连接参数是 CAN 接口名：`can0` / `can1`（不是 `/dev/ttyACM*`）。
- `canable_serial` 是 USB-CAN 盒序列号（如 `207433A55743`），用于稳定左右臂绑定。
- 推荐用 udev 固定设备名（如 `/dev/canable_left`、`/dev/canable_right`），再用 `slcand` 映射到 `can0/can1`。

示例（1Mbps）：

```bash
./scripts/start_arx5_can.sh

# 或手动执行
sudo killall slcand 2>/dev/null
sudo slcand -o -c -s8 /dev/canable_left can0
sudo slcand -o -c -s8 /dev/canable_right can1
sudo ip link set can0 up
sudo ip link set can1 up
ip link show can0
ip link show can1
```

## 快速开始

```bash
# 1) 连接探针（推荐先跑）
python Example/arx5/probe.py Config/arx5_example.yaml --polls 5 --enable
python Example/arx5/probe.py Config/arx5_bimanual_example.yaml --polls 5 --enable

# 2) 回 Home
python Example/arx5/go_home.py Config/arx5_example.yaml
python Example/arx5/go_home.py Config/arx5_bimanual_example.yaml

# 3) 夹爪控制
python Example/arx5/gripper.py Config/arx5_example.yaml open close
python Example/arx5/gripper.py Config/arx5_bimanual_example.yaml open close --both

# 4) smoke test（不依赖推理服务）
python Example/arx5/single_arm_smoke_test.py Config/arx5_example.yaml
python Example/arx5/bimanual_smoke_test.py Config/arx5_bimanual_example.yaml

# 5) parquet 回放
python Example/arx5/replay_parquet.py Config/arx5_example.yaml /path/to/data.parquet
python Example/arx5/replay_parquet.py Config/arx5_bimanual_example.yaml /path/to/data.parquet

# 5.1) DCSv2 双臂 EEF 回放（先 dry-run）
python Example/arx5/replay_eef.py \
  Config/NeoVTLA/arx5_bimanual_neovtla.yaml \
  /path/to/episode_0002

# 6) client-server测试
# GPU 机器: 启动 mock echo server
python Example/test_inference_server.py --port 5555
# 机器人机器: 连接远程 server，运行 observe → predict → dispatch 回环
python Example/test_inference_client.py Config/arx5_example.yaml --server 192.168.50.81:5555
python Example/test_inference_client.py Config/arx5_bimanual_example.yaml --server 192.168.50.81:5555 --steps 60
# 无硬件测试 (两个终端同一台机器)
python Example/test_inference_client.py Config/arx5_bimanual_example.yaml --dry-run --server 127.0.0.1:5555
```

## 双臂 EEF episode replay

`replay_eef.py` 读取 `actions.eef_pose/data.csv` 的绝对动作，磁盘布局为左右臂
各 10 维 `[xyz3, rot6d6, gripper1]`。程序复用推理链路的 column-major
rot6d → quaternion 转换，向 ARX5 驱动下发 16 维双臂 Cartesian action。

默认只做离线预检，包括列名、数值、时间戳、相邻帧位移/转角和夹爪范围，
不会连接机器人。抽取部分帧可使用 `--start 100 --end 300`。确认 dry-run
结果后再执行：

```bash
python Example/arx5/replay_eef.py \
  Config/NeoVTLA/arx5_bimanual_neovtla.yaml \
  /path/to/episode_0002 \
  --execute
```

实机模式会先回 Home，再用默认 5 秒平滑逼近首帧，并按 CSV 的真实时间戳
逐帧回放；`--speed 0.5` 表示半速。控制周期变慢时不会加速追赶。空格暂停/
继续，`q` 退出。任一侧 IK 精确目标和缩步恢复均失败时，驱动保持双臂当前
位置，replay 立即终止，不会执行后续 action。首次实机运行不要使用 `--yes`。

默认连续性限制为单帧 30 mm / 0.2 rad；如预检失败，应先检查数据，不建议
直接放宽。`--max-gap-s 0.2` 仅截断异常长的记录时间间隔。

## 动作维度约定

- 单臂 (`arx5`)
  - `6` 维: `[q0..q5]`
  - `7` 维: `[q0..q5, gripper]`
- 双臂 (`arx5_bimanual`)
  - `14` 维: `[left_q0..q5, left_gripper, right_q0..q5, right_gripper]`

## 安全建议

- 首次联调先用小幅动作（`--amp-deg 1~3`）和较低频率（`--hz 10~30`）。
- 先运行 `probe.py`，确认状态读取稳定后再做动作脚本。
- 现场建议预留急停手段；若出现通信阻塞，先停止脚本并检查 CAN/网卡连接。
