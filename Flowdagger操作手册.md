# ARX 双臂 FlowDAgger 实机操作手册

更新时间：2026-09-04。本文对应当前 arx-ubuntu 控制端代码与 robot 侧 `0901` 基座。

所有命令和数据只涉及：

- robot-ubuntu：`/home/ubuntu/yzy`
- arx-ubuntu：`/home/xinzhi/InferSystem_FlowDAgger`

## 0. 最短操作路径

每次实机使用两个 SSH 终端。先启动 robot 服务并确认就绪，再启动 ARX 控制程序；退出顺序相反。

当前现场默认流程：

```text
demonstration → bootstrap → closed_loop
```

`baseline`、`shadow` 仍可运行，但 **不再是 closed-loop 的软件门禁**。ARX 启动脚本只检查：服务模式、协议、base model、动作规格，以及 closed-loop/shadow 是否已有 eligible `ACTIVE`。

### 采集示教数据

robot-ubuntu：

```bash
cd /home/ubuntu/yzy/FlowDAgger
./run_arx_flowdagger_server.sh --record-only
```

arx-ubuntu：

```bash
cd /home/xinzhi/InferSystem_FlowDAgger
./run_arx_flowdagger.sh demonstration
```

现场顺序：`Enter` 开始 → `Space` 进入拖动 → 手动示教 → `Space` 返回策略。成功按 `1`；失败按 `3`；异常或主动中止按 `r`。服务端根据是否发生过接管，自动把成功回合分成 `autonomous_success` 或 `assisted_success`。

每个回合结束后，确认 arx 终端出现 `FlowDAgger 数据保存成功`，再开始下一回合。

### 从 Base Policy 启动在线学习

示教完成后，先在 arx 端按 `q` 退出，再在 robot 服务终端按 `Ctrl+C`。确认服务停止后，启动完整服务：

```bash
cd /home/ubuntu/yzy/FlowDAgger
./run_arx_flowdagger_server.sh
```

arx-ubuntu：

```bash
cd /home/xinzhi/InferSystem_FlowDAgger
./run_arx_flowdagger.sh bootstrap
```

bootstrap 执行纯 base policy。成功统一按 `1`；只有发生过接管、且服务端反演后窗口足够的回合才会从零训练 steering。生成 eligible `version_000001` 并出现 `ACTIVE` 后，退出 bootstrap，直接运行：

```bash
./run_arx_flowdagger.sh closed_loop
```

## 1. 安全要求

1. 必须有人站在机械臂旁，急停保持可触达。
2. 启动、回 Home、退出 EXPERT、closed-loop 执行时，双臂工作空间内不得有人或障碍物。
3. 控制程序显示“按 Enter”时才允许开始回合。
4. 出现抖动、跳变、异常声音、碰撞风险或通信异常时，先按急停，再处理软件。
5. `--protocol-only` 固定返回零动作，只能在机械臂断开时测试网络，严禁用于实机控制。
6. 远程 SSH 不要加 `--show-cameras`。无图形显示时 Qt 会退出；相机数据仍会采集和保存。

## 2. 固定配置

以 arx-ubuntu 的 `flowdagger_preflight.py` 为准。当前必须匹配：

| 项 | 当前值 |
|---|---|
| 服务地址 | `192.168.50.124:5557` |
| 协议 | 3 |
| `base_model_id` | `connect_elevator_pins_arx_0901:20000:648ff0462d1cec61` |
| 相机 | 第三视角、左腕、右腕，`640×480 @ 30 Hz` |
| 外部 state / action | 20D 双臂 EEF |
| action horizon | 50 |
| 控制频率 | 30 Hz |
| EEF 平移限幅 | 5 mm / 周期 |
| 首帧安全阈值 | 0.10 m / 0.8 rad |
| 夹爪示教开口 / 闭合 | `0.070 m` / `0.000 m` |

robot 侧 campaign 名由服务启动参数决定，**不在 ARX preflight 里硬编码**。以本回合保存路径为准。2026-09-04 现场训练目录示例：

```text
/home/ubuntu/yzy/flowdagger_runs/arx_connect_elevator_pins_0901_base_retrain_v6/episodes/
```

服务启动日志必须包含 `0901` 和 step `20000`。如果仍出现 `0829_332`、`0828`、checkpoint identity mismatch、asset mismatch 或 traceback，立即停止，不要启动 ARX 端。

ARX 客户端回合 ID 形如 `episode_20260904_104034_0001`；robot 落盘目录可能把序号和时间对调，例如 `episode_20260904_0004_104338`。以 `FlowDAgger 数据保存成功` 打印的完整路径为准。按 `1` / `3` / `r` 后如果没有这行，不要开始下一回合。

## 3. 阶段与服务模式

| 阶段 | robot 服务 | 实机执行 | 是否训练 | ARX preflight 额外要求 |
|---|---|---|---|---|
| `demonstration` | `--record-only` | base policy，可接管 | 否 | 无 ACTIVE |
| `baseline` | `--record-only` | 纯 base policy | 否 | 无 ACTIVE |
| `bootstrap` | 完整服务 | 纯 base policy，可接管 | 首个合格 assisted-success 起训 | 允许 `policy_version=0` |
| `shadow` | 完整服务 | 执行 base；steering 只记录 | 否 | `policy_version>0` 且 `steering_eligible=true` |
| `closed_loop` | 完整服务 | 执行 steering，可接管 | 合格 assisted-success 会更新 | 同上，且当前无活动回合、无训练 |
| `demo` | 完整服务 | 执行指定 steering | 否，也不落训练 episode | 走 `run_arx_flowdagger_demo.sh`，不经 preflight |

启动 ARX 控制程序：

```bash
cd /home/xinzhi/InferSystem_FlowDAgger
./run_arx_flowdagger.sh demonstration
./run_arx_flowdagger.sh bootstrap
./run_arx_flowdagger.sh closed_loop
```

脚本会先跑 `flowdagger_preflight.py`，通过后才连接机械臂和相机。看到 JSON 报告且没有 `errors` 才继续。随后程序回 Home、移动到训练起始位，并停在：

```text
硬件已就绪，按 Enter 开始推理
```

## 4. 启动 robot-ubuntu 服务

```bash
cd /home/ubuntu/yzy/FlowDAgger
```

示教和 baseline：

```bash
./run_arx_flowdagger_server.sh --record-only
```

bootstrap / shadow / closed-loop / demo：

```bash
./run_arx_flowdagger_server.sh
```

看到下面的信息才表示服务就绪：

```text
ARX FlowDAgger server listening on tcp://*:5557
```

同时确认 health / 日志中的 `base_model_id` 与第 2 节完全一致。

## 5. 回合内按键

按键在运行控制程序的 arx-ubuntu 终端中输入。除 `Enter` 外，不要在字符后再按回车。

| 按键 | 功能 |
|---|---|
| `Enter` | 从等待状态开始下一回合 |
| `Space` | 在 POLICY 与 EXPERT 拖动示教之间切换 |
| `1` | 标记 `task_outcome=success`；服务端自动判断 assisted / autonomous |
| `3` | 标记 `failure` 并结束回合 |
| `r` | 标记 `abort`、归档诊断数据、回 Home，不训练 |
| `q` | 安全退出程序 |
| `Ctrl+C` | 紧急软件退出并尝试回 Home；有危险时仍以物理急停优先 |
| `2` | 无功能 |

按键是单字符即时生效。按键后先看终端状态是否已经切换，避免重复按 `Space` 导致刚进入 EXPERT 又立即退出。

### 夹爪说明

在 EXPERT 中，双臂关节和两个夹爪都进入手动示教：Kp=0，保留 Kd 阻尼。直接用手平稳推动夹爪到需要的宽度，系统会连续记录实测夹爪宽度。`1` 是成功结果按键，不控制夹爪。不要快速推拉或对机械限位持续施力。当前示教开口 `0.070 m`，闭合 `0.000 m`。YAML 中夹爪机械上限是 `0.085 m`，启动位刻意停在 `0.070 m`，避免顶住张开限位触发过流。

## 6. demonstration：拖动示教

```bash
./run_arx_flowdagger.sh demonstration
```

1. 等待双臂到训练起始位。
2. 确认环境安全后按 `Enter`。
3. 回合先进入 POLICY，由基座策略控制。
4. 需要纠正时按 `Space` 进入 EXPERT。
5. 手动完成需要纠正的一小段；夹爪也直接用手平稳控制。
6. 再按 `Space` 返回 POLICY。接管后的第一个新策略 chunk 会以最新实测双臂 EEF 为起点重对齐，不会沿用接管前旧动作。
7. 一个回合内可以多次切换。
8. 成功按 `1`，失败按 `3`，异常按 `r`。
9. 可以在 EXPERT 中直接按 `1/3/r`：程序会先退出拖动、保持实时位置，再结束回合和回 Home。

示教结束时 `training_queued=False` 是正常的。demonstration 只采集，首次训练在 bootstrap 的合格 assisted 回合结束后触发。不要因为该字段为 false 而重复采集同一个回合。

建议至少采集若干带连续 EXPERT 数据的 `assisted_success` 回合，供 bootstrap 首次训练使用。`autonomous_success` 只用于评估；failure 和 abort 仅归档诊断。

快速查看已保存回合：

```bash
ROOT=/home/ubuntu/yzy/flowdagger_runs/arx_connect_elevator_pins_0901_base_retrain_v6
find "$ROOT/episodes" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
tail -n 100 "$ROOT/logs/service.log"
```

每个可训练成功回合至少应包含 `metadata.json`、`steps.jsonl` 和三路相机图像。不要手工修改这些文件，也不要把旧 campaign 的 episode 复制进当前目录。`ROOT` 以本回合 `数据保存成功` 路径为准。

## 7. bootstrap：从零训练 steering

新 campaign 初始应为 `policy_version=0`、`steering_eligible=false`，且不存在 `steering_checkpoints/ACTIVE`。完整服务启动后：

```bash
./run_arx_flowdagger.sh bootstrap
```

- 实机执行 base policy，允许 `Space` 接管。
- 成功统一按 `1`。
- 只有发生过接管、反演后有效窗口足够的回合才会启动从零 steering 训练。
- 不恢复任何旧 steering / optimizer / normalization。
- 训练返回 `no_improvement`、`rejected` 或 `failed` 时不会创建新版本，也不会卡住客户端。
- 成功时应生成 `version_000001`、该版本专属 eligibility，以及 `ACTIVE=1`。

生成 eligible ACTIVE 后退出 bootstrap。可以直接进入 closed-loop；shadow 只用于对照，不是必须步骤。

训练排队后不要按 Enter 开始下一回合。终端出现 `FlowDAgger 更新完成: policy_version=...` 或明确的 `未发布新版本` / `更新失败` 后，才会回到 Enter 等待。

## 8. closed-loop：steering 实机闭环

启动条件：

- `policy_version > 0`
- `steering_eligible=true`
- 当前没有活动 episode，也没有正在进行的训练

```bash
cd /home/xinzhi/InferSystem_FlowDAgger
./run_arx_flowdagger.sh closed_loop
```

此阶段会真实下发 steering。保持 5 mm/周期 EEF 平移限幅。只有自动分类为 `assisted_success` 且包含有效 EXPERT transition 的回合，会在回 Home 后触发 steering BC 更新；更新完成前不能开始下一回合。`autonomous_success`、failure、abort 和安全异常均不训练。

启动时 health 必须显示 `protocol_version=3`、正确的 `0901` `base_model_id`、非空 `server_session_id` 和 `steering_eligible=true`。

## 9. 可选阶段

### baseline

继续使用 `--record-only` 服务：

```bash
./run_arx_flowdagger.sh baseline
```

记录纯基座策略，不训练 steering。成功按 `1`，失败按 `3`，中止按 `r`。

### shadow

完整服务下：

```bash
./run_arx_flowdagger.sh shadow
```

实际执行基座策略；steering 只在后台计算并记录差异，不下发。用于对照，不是 closed-loop 的软件前置条件。

### demo：执行 steering，不采集、不训练

用于展示或人工验收某个 steering 版本。连接完整服务后：

```bash
cd /home/xinzhi/InferSystem_FlowDAgger
./run_arx_flowdagger_demo.sh
```

默认使用 `ACTIVE`。指定版本可追加参数，例如 `--steering-version 11`。demo 会执行 steering，但不打开 FlowDAgger episode，也不保存训练数据。仍需有人值守急停。

## 10. 安全停止与 SAFE_HOLD

策略动作超过首帧安全阈值，或被 IK / dispatcher 拒绝时，程序会：

1. 停止当前动作并保持实测关节位置；
2. 将当前回合归档为 `abort`；
3. 回 Home，再回到训练起始位；
4. 要求重新按 `Enter`。

终端会出现 `FlowDAgger SAFE_HOLD: ... 当前回合 abort`。这不是“原地继续等接管”：该回合已经结束，需要重新开始。

服务重启、响应 generation / step / version 不匹配、chunk 超龄或通信超时，同样进入 SAFE_HOLD。

## 11. 暂停与退出

- 回合中临时停止：按 `r`。当前回合归档为 abort，恢复增益并回 Home，随后停在 Enter。
- 完全退出：在 Enter 等待状态按 `q`，确认双臂完成安全退出和 CAN 断开。
- 最后在 robot-ubuntu 服务终端按 `Ctrl+C`，释放 GPU。

退出后检查不得有遗留控制进程：

```bash
# arx-ubuntu
pgrep -af '/home/xinzhi/InferSystem_FlowDAgger'

# robot-ubuntu
pgrep -af '/home/ubuntu/yzy/FlowDAgger'
```

`pgrep` 只显示它自身的查询命令时表示没有遗留项目进程。正常退出日志应包含双臂回零、`Set to damping before exit`、接收线程退出、`CAN socket destroyed` 和 `已断开 ARX5 双臂`。不要直接关闭 SSH 窗口来代替 `q`。

## 12. 常见问题

### 手臂可以拖动，但夹爪仍动不了

EXPERT 会同时将夹爪 Kp 置零并保留 Kd。如果仍无法平稳手动，不要强行推拉；按 `r` 结束回合，检查终端日志中的 gripper kp/kd 和夹爪机械状态。

### preflight 报 server mode 错误

- demonstration / baseline：robot-ubuntu 必须使用 `--record-only`。
- bootstrap / shadow / closed-loop：必须使用完整服务，不能带 `--record-only`。

### preflight 报 base_model_id 不匹配

当前必须是 `connect_elevator_pins_arx_0901:20000:648ff0462d1cec61`。旧手册里的 `0829_332:19999` 已经作废。先停 ARX，核对 robot 服务启动的 checkpoint。

### preflight 报需要 trained / eligible steering

shadow 和 closed-loop 都要求已有 eligible `ACTIVE`。先跑 bootstrap 生成 version 1，不要在 `policy_version=0` 时启动这两类阶段。

### 出现 gripper torque is too large

不要继续向机械限位施力。先按 `r` 或 `q` 回 Home，检查夹爪是否受阻，再重新开始。启动位开口是 `0.070 m`。

### 回合结束出现 ffmpeg stdin 写入失败

如果同时出现 `视频已保存`、非零帧数、`数据记录已关闭` 和 `FlowDAgger 数据保存成功`，该警告只影响 arx 本机预览视频收尾，不代表 robot 侧训练 episode 丢失。仍应检查对应 MP4；若缺少数据保存成功日志，则不要开始下一回合。

### 成功后显示 training_queued=False

- demonstration、baseline、shadow、demo：正常，不进行在线训练。
- bootstrap / closed_loop：只有 success、发生过接管且存在连续 EXPERT transition 才会排队训练。
- `autonomous_success`：只用于成功率评估，永远不训练。
- failure 和 abort：只用于诊断，默认不训练。
