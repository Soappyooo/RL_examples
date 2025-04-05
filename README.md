# 强化学习项目

## 0. 项目介绍

国科大2025春季学期强化学习课程作业一。  
  
本项目实现了两种强化学习算法（DQN和PPO）用于解决两个环境问题：倒立摆(Pendulum)控制和冰壶(Curling)游戏。项目包含完整的训练框架、环境定义、智能体实现以及可视化工具。

主要特点：
- 支持DQN（Deep Q-Network）和PPO（Proximal Policy Optimization）算法
- 包含两个环境：倒立摆(Pendulum)和冰壶(Curling)
- 提供训练、评估和可视化工具
- 使用YAML配置文件进行参数管理

## 1. 环境安装

请按照以下步骤安装所需环境：

```bash
# 克隆仓库（如果通过git获取）
git clone <repository-url>
cd <repository-directory>

# 安装依赖
pip install -r requirements.txt
```

## 2. 运行演示

### 查看环境演示

项目提供了两个playground文件用于直观展示环境和已训练的智能体：

```bash
# 运行倒立摆环境演示
python pendulum_playground.py

# 运行冰壶环境演示
python curling_playground.py
```

### 查看结果

- **图像结果**：在`images/`目录中包含了各个算法在不同环境中的训练和评估曲线图，如`pendulum_dqn_train.png`、`curling_ppo_eval.png`等。

- **可视化视频**：在`visualization/`目录中包含了训练过程中智能体行为的视频记录，分别存放在`pendulum/`和`curling/`子目录中。

## 3. 训练智能体

使用`train.py`脚本来训练智能体，支持以下命令行参数：

```bash
python train.py [--config CONFIG_PATH]
```

### 命令行参数

- `--config`：配置文件路径（可选，默认为"./config/curling/ppo.yml"）

配置文件包含了环境设置、智能体参数和训练参数。系统会根据配置文件中的training.mode值选择训练模式（"on-policy"用于PPO，"off-policy"用于DQN）。

### 示例

```bash
# 使用默认配置文件训练
python train.py

# 使用自定义配置文件训练PPO智能体解决冰壶问题
python train.py --config ./config/curling/ppo.yml

# 使用自定义配置文件训练DQN智能体解决倒立摆问题
python train.py --config ./config/pendulum/dqn.yml
```

配置文件中已包含了训练模式、检查点路径、日志路径等必要设置，系统会自动创建日志和检查点目录。  
  
  

---

*以上内容由GitHub Copilot生成。*
