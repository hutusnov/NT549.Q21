import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingQNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
    def __init__(self, state_dim=8, action_dim=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.max_memory = 10000
        self.target_update_freq = 100
        self.replay_count = 0

        self.memory = []
        self.priorities = []
        self.per_alpha = 0.6
        self.per_beta = 0.4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online network (cập nhật mỗi step)
        self.model = DuelingQNetwork(state_dim, action_dim).to(self.device)
        # Target network (cập nhật định kỳ) — ổn định Q-target
        self.target_model = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss, ổn định hơn MSE

    def get_action(self, state, explore=True):
        """
        Chọn action. Set explore=False khi chạy production (inference only).
        """
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values[0]).item()

    def remember(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_prio)
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
            self.priorities.pop(0)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        prios = np.array(self.priorities)
        probs = prios ** self.per_alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        minibatch = [self.memory[i] for i in indices]

        # Importance Sampling weights
        N = len(self.memory)
        weights = (N * probs[indices]) ** (-self.per_beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(self.device)

        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(self.device)

        # Double DQN: online network CHỌN action, target network ĐÁNH GIÁ
        self.optimizer.zero_grad()
        q_values = self.model(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)

        # KEY: khi done=False, gamma * Q(s') được tính → agent học hậu quả dài hạn
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        td_errors = torch.abs(targets - q_values).detach()
        loss = (weights * (q_values - targets) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Cập nhật priority
        td_errors_np = td_errors.cpu().numpy()
        for i, idx in enumerate(indices):
            self.priorities[idx] = float(td_errors_np[i]) + 1e-5

        # Đồng bộ target network định kỳ
        self.replay_count += 1
        if self.replay_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_path="dqn_model.pth"):
        torch.save({
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'epsilon': self.epsilon,
            'replay_count': self.replay_count,
        }, file_path)

    def load_model(self, file_path="dqn_model.pth"):
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path, map_location=self.device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                self.target_model.load_state_dict(
                    checkpoint.get('target_model', checkpoint['model'])
                )
                self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
                self.replay_count = checkpoint.get('replay_count', 0)
            else:
                try:
                    self.model.load_state_dict(checkpoint)
                    self.target_model.load_state_dict(checkpoint)
                    self.epsilon = self.epsilon_min
                except RuntimeError:
                    print("⚠️ Model cũ không tương thích (state_dim đã đổi). Học từ đầu.")
                    return False
            self.model.eval()
            print(f"✅ Đã load model từ {file_path} (ε={self.epsilon:.3f})")
            return True
        print("🆕 Không tìm thấy model. Agent sẽ học từ đầu.")
        return False
