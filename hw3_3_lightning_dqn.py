import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset
import copy
import random
from collections import deque
from Gridworld import Gridworld
import warnings
warnings.filterwarnings("ignore")

class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size=200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        if len(self.buffer) < self.sample_size:
            # Not enough samples, just yield empty or small
            for b in self.buffer:
                yield b
        else:
            samples = random.sample(self.buffer, self.sample_size)
            for b in samples:
                yield b

class DQNLightning(pl.LightningModule):
    def __init__(self, L1=64, L2=150, L3=100, L4=4, lr=1e-3, gamma=0.9):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(L1, L2),
            nn.ReLU(),
            nn.Linear(L2, L3),
            nn.ReLU(),
            nn.Linear(L3, L4)
        )
        self.target_model = copy.deepcopy(self.model)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.lr = lr

        self.mem_size = 1000
        self.replay_buffer = deque(maxlen=self.mem_size)
        self.epsilon = 1.0
        self.game = Gridworld(size=4, mode='random')
        self.state = self.get_state()
        self.action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        self.max_moves = 50
        self.mov = 0
        
        # Populate buffer initially
        for _ in range(200):
            self.play_step()

    def get_state(self):
        state_ = self.game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        return torch.from_numpy(state_).float()

    def forward(self, x):
        return self.model(x)

    def play_step(self):
        qval = self.model(self.state)
        if random.random() < self.epsilon:
            action_ = np.random.randint(0, 4)
        else:
            action_ = torch.argmax(qval).item()
        
        self.game.makeMove(self.action_set[action_])
        next_state = self.get_state()
        reward = self.game.reward()
        done = True if reward != -1 else False
        
        exp = (self.state.squeeze(), action_, np.float32(reward), next_state.squeeze(), np.float32(done))
        self.replay_buffer.append(exp)
        self.state = next_state
        self.mov += 1

        if abs(reward) == 10 or self.mov > self.max_moves:
            self.game = Gridworld(size=4, mode='random')
            self.state = self.get_state()
            self.mov = 0
            
        return reward

    def training_step(self, batch, batch_idx):
        self.play_step()
        
        if len(self.replay_buffer) < 200:
            return None 

        states, actions, rewards, next_states, dones = batch
        
        Q1 = self.model(states)
        with torch.no_grad():
            Q2 = self.target_model(next_states)
        
        max_q2 = torch.max(Q2, dim=1)[0]
        Y = rewards + self.gamma * ((1 - dones) * max_q2)
        X = Q1.gather(1, actions.long().unsqueeze(1)).squeeze()
        
        loss = self.loss_fn(X, Y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Bonus: LR Scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        return [optimizer], [scheduler]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.epsilon > 0.1:
            self.epsilon -= (1 / 5000)
        if self.global_step % 500 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def train_dataloader(self):
        dataset = RLDataset(self.replay_buffer, sample_size=200)
        return DataLoader(dataset, batch_size=200)

def run_hw3_3():
    model = DQNLightning()
    # Bonus: Gradient clipping
    trainer = pl.Trainer(max_epochs=1, max_steps=5000, gradient_clip_val=1.0)
    trainer.fit(model)
    print("HW3-3 Finished Training with PyTorch Lightning, LR Scheduling, and Gradient Clipping!")

if __name__ == "__main__":
    run_hw3_3()
