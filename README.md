# 🚀 DQN Agent for LunarLander-v2

This project implements a **Deep Q-Network (DQN)** to train an agent to solve the [LunarLander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) environment from OpenAI Gym. The agent learns to land a spacecraft smoothly between two flags using reinforcement learning.

---

## 📌 Project Overview

- **Environment:** LunarLander-v2 (Discrete action space)
- **Algorithm:** Deep Q-Learning (DQN)
- **Framework:** PyTorch
- **Goal:** Train an agent to maximize rewards by safely landing the lunar module.

---

## 🎯 Problem Definition

The agent must learn to:
- Land on the target pad between two flags.
- Avoid crashing.
- Minimize fuel usage while keeping the lander stable.

**Reward Structure (as per OpenAI Gym):**
- Landing between flags: **+100 to +140** points.
- Crash: **-100 points**.
- Each timestep uses fuel: **small negative reward**.

---

## 🧠 Algorithm Details

The **DQN** approach used here involves:
1. **Q-Network** – A neural network mapping states to Q-values for each action.
2. **Target Network** – For stable training, updated slowly towards Q-Network weights.
3. **Replay Buffer** – Stores experience tuples `(state, action, reward, next_state, done)` for sampling.
4. **Epsilon-Greedy Policy** – Balances exploration and exploitation.

**Network Architecture:**
```
Input Layer: state_size (8)
Hidden Layer 1: 128 neurons, ReLU
Hidden Layer 2: 64 neurons, ReLU
Output Layer: action_size (4) -> Q-values
```

---

## 📂 Project Structure

```
.
├── dqn_agent.py           # Agent class, replay buffer, training step
├── model.py               # Q-network architecture
├── Deep_Q_Network.ipynb   # Training notebook
├── checkpoint.pth         # Saved trained model weights
└── README.md              # Project description
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/dqn-lunar-lander.git
cd dqn-lunar-lander
```

**2. Create environment & install dependencies**
```bash
conda create --name lunarlander python=3.8
conda activate lunarlander
pip install -r requirements.txt
```

**Requirements (`requirements.txt`):**
```
torch
numpy
matplotlib
gym[box2d]
```

---

## 🚀 Training the Agent

Run the notebook:
```bash
jupyter notebook Deep_Q_Network.ipynb
```

Or run a training script (if you make one):
```bash
python train.py
```

---

## 📊 Results

- The agent successfully lands with an average score > 200 in under **X episodes**.




---

## 💾 Using the Trained Agent

You can load and watch the trained agent:
```python
from dqn_agent import Agent
import torch

agent = Agent(state_size=8, action_size=4, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
```

---

## 📜 References

- [Deep Q-Learning Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [OpenAI Gym LunarLander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/)
- Udacity Deep Reinforcement Learning Nanodegree
