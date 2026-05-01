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
            nn.Linear(input_dim, 24),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class DQNAgent:
    def __init__(self, state_dim=6, action_dim=2):
        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.gamma = 0.95        
        self.epsilon = 1.0       
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.max_memory = 2000
        
        self.memory = []
        self.priorities = [] 
        self.per_alpha = 0.6 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values[0]).item()

    def remember(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities) if self.memory else 1.0
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
        
        states = torch.FloatTensor(np.array([transition[0] for transition in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([transition[1] for transition in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([transition[2] for transition in minibatch])).to(self.device)
        
        next_states = torch.FloatTensor(np.array([transition[3] for transition in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([transition[4] for transition in minibatch])).to(self.device)

        self.optimizer.zero_grad()
        q_values = self.model(states).gather(1, actions).squeeze(1)
        next_q_values = self.model(next_states).detach().max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        loss = self.criterion(q_values, targets)
        loss.backward()
        self.optimizer.step()

        td_errors = torch.abs(targets - q_values).detach().cpu().numpy()
        for i, idx in enumerate(indices):
            self.priorities[idx] = float(td_errors[i]) + 1e-5

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, file_path="dqn_model.pth"):
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path="dqn_model.pth"):
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path, map_location=self.device))
            self.model.eval()
            self.epsilon = self.epsilon_min 
            print(f"Đã gắn Model Dueling + PER từ {file_path}!")
            return True
        print("AI sẽ học từ đầu bằng Dueling + PER.")
        return False
