import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import time

from env import FDTDEnv

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
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
        hidden = self.shared(x)
        mean = torch.sigmoid(self.actor_mean(hidden))
        logstd = self.actor_logstd.squeeze(0).expand_as(mean)
        value = self.critic(hidden)
        return mean, logstd, value
    
class PPOBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, gamma=0.99, gae_lambda=0.95):
        self.buffer_size = buffer_size
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

    def add(self, state, action, log_prob, reward, value, done):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.buffer_size 

    def is_full(self):
        return self.pos == self.buffer_size - 1
    
    def compute_advantages(self, last_value, last_done):
        last_gae = 0
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t+1]
                next_value = self.values[t+1]
            
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size):
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
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 clip_eps=0.2, ent_coef=0.01, n_epochs=4, batch_size=64, 
                 buffer_size=2048):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.buffer = PPOBuffer(buffer_size, state_dim, action_dim, gamma)
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=1e-5
        )

    def act(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            mean, logstd, value = self.actor_critic(state_tensor)
            dist = Normal(mean, logstd.exp())
            action = dist.sample()
            action = torch.clamp(action, 0, 1)
            log_prob = dist.log_prob(action).sum(-1)
        return action.numpy(), log_prob.item(), value.item()

    def update(self):
        print('update agent')
        # 计算优势
        last_state = self.buffer.states[-1]
        with torch.no_grad():
            last_value = self.actor_critic(torch.FloatTensor(last_state))[2]
        last_done = self.buffer.dones[-1] if self.buffer.pos > 0 else True
        self.buffer.compute_advantages(last_value, last_done)
        
        # 多epoch更新
        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                states, actions, old_log_probs, advantages, old_values = batch
                
                means, logstds, values = self.actor_critic(states)
                dist = Normal(means, logstds.exp())
                log_probs = dist.log_prob(actions).sum(-1)
                
                # 策略损失
                ratios = torch.exp(log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.clip_eps, 1+self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 值函数损失
                value_loss = F.mse_loss(values.flatten(), old_values + advantages)
                
                # 熵正则化
                entropy_loss = -dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss + self.ent_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
        
        self.buffer.pos = 0
        

def train_ppo():
    env = FDTDEnv()
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        buffer_size=16,
        batch_size=4,
        n_epochs=4,
        ent_coef=0.01
    )
    
    for episode in range(10000):
        state = env.reset()
        episode_rewards = []
        
        while True:
            action, log_prob, value = agent.act(state)
            next_state, reward, done, info = env.step(action)

            print(f'[{time.time()-env._start_time:.2f}]', next_state, reward, done, info)
            
            agent.buffer.add(state, action, log_prob, reward, value, done)
            episode_rewards.append(reward)
            state = next_state
            
            if agent.buffer.is_full():
                agent.update()
            
            if done:    
                break
        
        print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards):.2f}, max: {np.max(episode_rewards):.2f}")
        
        # 只关心训练过程中找到的参数组合，模型直接丢弃

if __name__ == "__main__":
    train_ppo()