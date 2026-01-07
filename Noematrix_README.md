# Noematrix RLinf 部署指南

本文档介绍如何在 Noematrix 环境中部署和使用 RLinf 框架，包括 Server Ray 节点启动、Client 节点启动以及代码同步流程。

---

## 目录

- [1. 环境准备](#1-环境准备)
- [2. Server Ray 节点启动](#2-server-ray-节点启动)
- [3. Client 节点启动](#3-client-节点启动)
- [4. 运行例程](#4-运行例程)
- [5. 同步远程 main 分支代码](#5-同步远程-main-分支代码)
- [6. 常用命令速查](#6-常用命令速查)

---

## 1. 环境准备

确保已安装以下依赖：
- Docker & Docker Compose
- Ray (建议版本与镜像一致)
- Git

---

## 2. Server Ray 节点启动

Server 节点作为 Ray 集群的 Head 节点，负责协调整个分布式计算集群。

### 2.1 启动 Head 节点

在 Server 机器上运行以下命令：

```bash
# 设置环境变量
export RLINF_NODE_RANK=0
export RANK=0

# 启动 Ray Head 节点
bash ray_utils/start_ray.sh
```

**启动脚本说明 (`ray_utils/start_ray.sh`):**

- **Head 节点 (RANK=0)**: 自动获取 IP 地址并启动 Ray Head 服务
- **Worker 节点 (RANK>0)**: 自动连接到 Head 节点

**关键配置参数:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RAY_PORT` | 8400 | Ray 通信端口 |
| `PASSWORD` | `Noematrix_tmp123!` | Redis 认证密码 |
| `--dashboard-port` | 8265 | Ray Dashboard 端口 |
| `--memory` | 10GB | 分配给 Ray 的内存 |

### 2.2 连接 VPN 加入局域网

在验证 Head 节点状态之前，需要先连接 VPN 加入 Noematrix 内网：

```bash
# 安装 VPN 客户端
cd openvpn_client
bash install_openvpn_client.sh
# 启动 VPN 客户端（根据实际 VPN 类型选择）
# OpenVPN 示例:
cd openvpn_client
bash start_openssh.sh
```

**验证 VPN 连接：**

```bash
# 确认已获取内网 IP
ip addr show

# 测试与 Server 的连通性
ping 10.8.0.1
```

> ⚠️ **注意**: 确保 VPN 连接成功后再进行后续操作，否则无法访问 Ray 集群。

### 2.3 验证 Head 节点状态

```bash
# 查看 Ray 集群状态
ray status

# 访问 Ray Dashboard
# 浏览器打开: http://<SERVER_IP>:8265，目前火山云ip固定为10.8.0.1
```

### 2.4 停止 Ray 服务

```bash
bash ray_utils/stop_ray.sh
```

此脚本会：
1. 停止 Ray 集群
2. 清理残留进程 (raylet, gcs_server, dashboard)
3. 清理临时文件和共享内存

---

## 3. Client 节点启动

Client 节点作为 Worker 加入 Ray 集群，通过openvpn搭建局域网连接到 Server。新增节点请联系@丁俊峰进行注册。

### 3.1 配置 Client

进入 `rlinf_client` 目录：

```bash
cd rlinf_client
```

#### 3.1.1 配置 `.env` 文件

创建或修改 `.env` 文件，设置客户端ip,端口及密码（避免与其他客户端冲突），ip需与新增节点分配ip一致：

```bash
RAY_CLIENT_IP=10.8.0.3
RAY_CLIENT_PORT=28103
```


### 3.2 启动 Client

```bash
# 添加执行权限
chmod +x run.sh

# 启动 Client
./run.sh start
```

### 3.3 Client 管理命令

```bash
# 启动
./run.sh start

# 停止
./run.sh stop

# 重启
./run.sh restart

# 查看状态
./run.sh status
```

### 3.4 Docker Compose 服务说明

`docker-compose.yaml` 包含两个服务：

| 服务 | 说明 |
|------|------|
| `ray_client` | Ray Worker 节点，加入 Ray 集群 |

---

## 4. 运行例程

在 Server 和 Client 节点都启动并成功加入 Ray 集群后，可以运行 embodied RL 例程进行测试。

### 4.1 运行 Real Robot PPO OpenPI π₀.₅ 例程

```bash
# 在 Server 节点上运行
bash examples/embodiment/eval_embodiment.sh real_robot_ppo_openpi_pi05
```

**例程说明：**

该例程使用 π₀.₅ 模型在真实机器人环境下进行 PPO 评估，配置文件位于：
`examples/embodiment/config/real_robot_ppo_openpi_pi05.yaml`

**关键配置参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `cluster.num_nodes` | 2 | 集群节点数量 |
| `actor.model.model_name` | openpi | 使用的模型 |
| `algorithm.max_episode_steps` | 480 | 每个 episode 最大步数 |
| `rollout.model_dir` | `/workspace/RLinf/model/openpi` | 模型路径 |

**节点配置：**

| 节点组 | 节点 | 用途 |
|--------|------|------|
| `RLinf` | 0 | Actor、Rollout 节点 (GPU 4-7) |
| `flexiv_R1` | 1 | 真实机器人环境节点 |

### 4.2 运行其他例程

可以通过传递不同的配置名称运行其他例程：

```bash
# 查看可用配置
ls examples/embodiment/config/

# 运行指定配置
bash examples/embodiment/eval_embodiment.sh <CONFIG_NAME>
```

**常用配置示例：**

| 配置名称 | 说明 |
|----------|------|
| `real_robot_ppo_openpi_pi05` | 真实机器人 + π₀.₅ 模型 |
| `maniskill_ppo_openvlaoft` | ManiSkill 仿真 + OpenVLA-OFT |
| `libero_ppo_openvlaoft` | LIBERO 仿真 + OpenVLA-OFT |

### 4.3 查看运行日志

运行日志会自动保存在 `logs/` 目录下：

```bash
# 查看最新日志
ls -lt logs/

# 实时查看日志
tail -f logs/<TIMESTAMP>/eval_embodiment.log
```

---

## 5. 同步远程 main 分支代码

### 5.1 首次设置 upstream

如果尚未添加上游仓库，先执行：

```bash
git remote add upstream https://github.com/RLinf/RLinf.git
```

### 5.2 同步 main 分支

```bash
# 拉取上游最新代码
git fetch upstream

# 切换到本地 remote_main 分支
git checkout remote_main

# 合并上游 main 分支到本地
git merge upstream/main --no-edit

# 推送到私有仓库
git push origin remote_main
```

### 5.3 合并到自定义分支

**方式一：使用 merge**

```bash
git checkout custom
git merge remote_main
```

**方式二：使用 rebase（推荐，保持提交历史整洁）**

```bash
git checkout custom
git rebase remote_main
```


## 6. 常用命令速查

### Ray 集群管理

| 命令 | 说明 |
|------|------|
| `ray status` | 查看集群状态 |
| `ray stop` | 停止本地 Ray |
| `ray stop --force` | 强制停止 Ray |
| `bash ray_utils/start_ray.sh` | 启动 Ray 节点 |
| `bash ray_utils/stop_ray.sh` | 停止并清理 Ray |

### Client 管理

| 命令 | 说明 |
|------|------|
| `./run.sh start` | 启动 Client |
| `./run.sh stop` | 停止 Client |
| `./run.sh restart` | 重启 Client |
| `./run.sh status` | 查看 Client 状态 |

### Git 同步

| 命令 | 说明 |
|------|------|
| `git fetch upstream` | 获取上游更新 |
| `git merge upstream/main --no-edit` | 合并上游 main |
| `git rebase main` | 变基到 main |
| `git push origin main` | 推送到远程 |

---

## 附录

### 网络架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Noematrix Cloud                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Server (Ray Head Node)                      │    │
│  │  IP: 14.103.157.86
                                       │    │
│  │  Ray Port: 8400                                          │    │
│  │  Dashboard: 8265                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ OpenVPN代理，server局域网ip为10.8.0.1
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Local Environment                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Client (Ray Worker Node)                    │    │
│  │                                  │    │
│  │  Ray Worker → 加入 Ray 集群                              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 故障排查

1. **Ray 连接失败**
   - 检查防火墙设置
   - 确认 `ray_head_ip.txt` 文件内容正确

2. **openvpn 连接异常**
   - 确认开启vpn加入内网

3. **端口冲突**
   - 修改 `.env` 中的 `RAY_CLIENT_PORT`
