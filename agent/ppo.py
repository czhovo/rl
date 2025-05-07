import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from typing import Tuple, Dict

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        # 策略头（输出均值和标准差）
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        # 价值头
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        mean = self.actor_mean(shared_out)
        logstd = self.actor_logstd.expand_as(mean)
        return mean, logstd, self.critic(shared_out)
    
    def act(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """选择动作（连续空间），返回动作、log概率和状态价值"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            mean, logstd, value = self.forward(state_tensor)
            dist = Normal(mean, logstd.exp())
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        return action.numpy(), log_prob.item(), value.item()

class PPOBuffer:
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.pos = 0
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer_size = buffer_size

    def add(self, state: np.ndarray, action: np.ndarray, log_prob: float, reward: float, value: float, done: bool):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.dones[self.pos] = done
        self.pos += 1

    def compute_advantages(self):
        """计算GAE优势"""
        last_value = self.values[-1] if not self.dones[-1] else 0
        last_gae = 0
        for t in reversed(range(self.buffer_size)):
            delta = self.rewards[t] + self.gamma * last_value * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * last_gae
            self.advantages[t] = last_gae
            last_value = self.values[t]
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size: int):
        """生成随机批次"""
        indices = np.random.permutation(self.buffer_size)
        for start in range(0, self.buffer_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield (
                torch.FloatTensor(self.states[batch_idx]),
                torch.FloatTensor(self.actions[batch_idx]),
                torch.FloatTensor(self.log_probs[batch_idx]),
                torch.FloatTensor(self.advantages[batch_idx]),
                torch.FloatTensor(self.values[batch_idx])
            )

class PPOAgent:
    def __init__(self, env, lr: float = 3e-4, gamma: float = 0.99, 
                 clip_eps: float = 0.2, ent_coef: float = 0.01, 
                 n_epochs: int = 10, batch_size: int = 64, 
                 buffer_size: int = 2048):
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.buffer = PPOBuffer(buffer_size, state_dim, action_dim, gamma)
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def update(self):
        """PPO更新逻辑"""
        self.buffer.compute_advantages()
        
        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                states, actions, old_log_probs, advantages, old_values = batch
                
                # 计算新策略的概率和值
                mean, logstd, values = self.actor_critic(states)
                dist = Normal(mean, logstd.exp())
                new_log_probs = dist.log_prob(actions).sum(-1)
                entropy = dist.entropy().mean()
                
                # 策略损失（Clipped Surrogate Objective）
                ratios = (new_log_probs - old_log_probs).exp()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.clip_eps, 1+self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 值函数损失
                value_loss = F.mse_loss(values.flatten(), old_values + advantages)
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy
                
                # 梯度更新
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        self.buffer.pos = 0  # 清空缓冲区

def train_ppo(env, max_episodes: int = 1000):
    agent = PPOAgent(env)
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action, log_prob, value = agent.actor_critic.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.buffer.add(state, action, log_prob, reward, value, done)
            episode_reward += reward
            state = next_state
            
            if agent.buffer.pos == agent.buffer.buffer_size:
                agent.update()
            
            if done:
                break
        
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")

# 使用示例
if __name__ == "__main__":
    env = FDTDEnv()  # 你的环境实例
    train_ppo(env)