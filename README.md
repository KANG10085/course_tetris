# course_tetris

录制：
cd /home/wzk/Tetris
/home/wzk/anaconda3/envs/py310/bin/python aiplay.py \
  --record-placement-dataset \
  --record-actors player \
  --record-frame board


训练：
cd /home/wzk/Tetris
/home/wzk/anaconda3/envs/py310/bin/python train_afterstate_bc.py \
  --data-root screenshot_datasets \
  --actor player \
  --epochs 40 \
  --batch-size 8 \
  --output-dir afterstate_bc_player_test \
  --split-by session


运行：
cd /home/wzk/Tetris
/home/wzk/anaconda3/envs/py310/bin/python aiplay.py --ai rl --checkpoint afterstate_bc_player_test/best.pt

cd /home/wzk/Tetris
/home/wzk/anaconda3/envs/py310/bin/python aiplay.py --ai rl --checkpoint afterstate_model2_player_v2/best.pt




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
