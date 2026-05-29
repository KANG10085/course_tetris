# course_tetris

录制：
python aiplay.py \
  --record-placement-dataset \
  --record-actors player \
  --record-frame board


训练：
python train_afterstate_bc.py \
  --data-root screenshot_datasets \
  --actor player \
  --epochs 40 \
  --batch-size 8 \
  --output-dir afterstate_bc_player_test \
  --split-by session


运行：
python aiplay.py --ai rl --checkpoint afterstate_bc_player_test/best.pt

python aiplay.py --ai rl --checkpoint afterstate_model2_player_v2/best.pt




这次强化学习用的是 基于 DQN 思路的 afterstate value learning，不是 PPO、A2C 这类策略梯度模型。

更准确地说，是这套：

模型本体：一个 MLP 价值网络 AfterstateValueNet
训练方式：experience replay + target network + epsilon-greedy
目标函数：reward + gamma * max(next_afterstate_value)
损失：smooth_l1_loss
动作定义：不是按键动作，而是“当前方块所有合法最终落点”里的一个
网络学的是：每个最终落点 afterstate 的价值
所以它不是标准“Q(state, key)”那种按键 DQN，而是更适合俄罗斯方块的：

Afterstate DQN / Afterstate Value Learning

代码位置对应是：

网络：rl_value_net.py
训练循环：train_rl_shared.py
环境：rl_shared_env.py
一句话总结：
当前强化学习用的是“基于 DQN 思路的 afterstate 价值网络”，底层网络是 PyTorch 的 MLP。


## 环境要求

Python 3.10 或更新版本。主要依赖：

- `pygame`
- `torch`

如果只运行启发式 AI，理论上不需要 PyTorch；如果运行默认的人机模式和 RL AI，需要安装 PyTorch，并保留模型文件：

```text
shared_model3_stage2_v1/best.pt
```

## 运行项目

启动主菜单：

```bash
python menu.py
```

主菜单中可以选择：

- `AI Easy`
- `AI Normal`
- `AI Hard`
- `Two Players`

## 直接进入人机模式

默认使用 RL AI 和 `shared_model3_stage2_v1/best.pt`：

```bash
python aiplay.py
```

指定难度：

```bash
python aiplay.py --difficulty easy
python aiplay.py --difficulty normal
python aiplay.py --difficulty hard
```

使用启发式 AI：

```bash
python aiplay.py --ai heuristic --difficulty normal
```

指定 RL 模型：

```bash
python aiplay.py --ai rl --checkpoint shared_model3_stage2_v1/best.pt --difficulty hard
```

## 双人模式

```bash
python towplayer.py
```

## 操作方式

玩家 vs AI：

```text
A: 左移
D: 右移
S: 软降一格
W: 硬降
E: 旋转
P: 暂停
R: 重新开始
Q: 返回菜单
```

双人模式：

```text
P1: A/D/S/W/E
P2: J/L/K/I/O
P: 暂停
R: 重新开始
Q: 返回菜单
```

## AI 难度说明

难度配置在 `aiplay.py` 的 `DIFFICULTY_SETTINGS` 中：

```text
Easy:   动作间隔 0.55 秒，落点失误率 15%
Normal: 动作间隔 0.35 秒，落点失误率 6%
Hard:   动作间隔 0.22 秒，落点失误率 0%
```

动作间隔越短，AI 反应越快；落点失误率越高，AI 越容易把目标落点随机偏移。

## 强化学习训练

项目包含一个可运行的强化学习训练入口：

```bash
python train_rl.py \
  --episodes 300 \
  --max-steps 800 \
  --batch-size 64 \
  --buffer-size 50000 \
  --warmup 1000 \
  --target-update 500 \
  --log-every 10 \
  --save-every 50 \
  --best-window 20 \
  --best-min-episodes 20 \
  --lr 1e-4 \
  --checkpoint-dir afterstate_model3_rl_v1 \
  --init-checkpoint shared_model3_stage2_v1/best.pt
```

断点续训：

```bash
python train_rl.py \
  --episodes 200 \
  --checkpoint-dir afterstate_model3_rl_v1 \
  --resume-checkpoint afterstate_model3_rl_v1/latest.pt
```

训练输出默认是本地 checkpoint 文件，例如：

```text
afterstate_model3_rl_v1/latest.pt
afterstate_model3_rl_v1/best.pt
```





