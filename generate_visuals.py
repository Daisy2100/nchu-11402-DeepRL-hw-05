import numpy as np
import torch
import matplotlib.pyplot as plt
from Gridworld import Gridworld
from collections import deque
import random
import os
import copy
from hw3_2_dueling_dqn import DuelingDQN

# Set plotting style
plt.style.use('ggplot')

def plot_and_save_grid(board_array, filename, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(title, fontsize=16)
    
    color_map = {' ': [1,1,1], 'P': [0.2,0.4,1], '+': [0.2,0.8,0.2], '-': [0.8,0.2,0.2], 'W': [0.2,0.2,0.2]}
    img = np.zeros((4, 4, 3))
    for i in range(4):
        for j in range(4):
            img[i, j] = color_map[board_array[i, j]]
            
    ax.imshow(img)
    
    for i in range(4):
        for j in range(4):
            char = board_array[i, j]
            if char != ' ':
                ax.text(j, i, char, ha='center', va='center', color='white', fontsize=24, fontweight='bold')
                
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

def train_and_visualize():
    print("Training Naive DQN (Static)...")
    model_naive = torch.nn.Sequential(
        torch.nn.Linear(64, 150), torch.nn.ReLU(),
        torch.nn.Linear(150, 100), torch.nn.ReLU(),
        torch.nn.Linear(100, 4)
    )
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model_naive.parameters(), lr=1e-3)
    gamma, epsilon = 0.9, 1.0
    epochs = 800
    replay = deque(maxlen=1000)
    losses_naive = []
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='static')
        state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
        status = 1
        mov = 0
        while status == 1:
            mov += 1
            qval = model_naive(state)
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval.data.numpy())
            game.makeMove({0: 'u', 1: 'd', 2: 'l', 3: 'r'}[action_])
            next_state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
            reward = game.reward()
            done = True if reward != -1 else False
            replay.append((state, action_, np.float32(reward), next_state, np.float32(done)))
            state = next_state
            
            if len(replay) > 200:
                minibatch = random.sample(replay, 200)
                state1_batch = torch.cat([s[0] for s in minibatch])
                action_batch = torch.Tensor([s[1] for s in minibatch])
                reward_batch = torch.Tensor([s[2] for s in minibatch])
                state2_batch = torch.cat([s[3] for s in minibatch])
                done_batch = torch.Tensor([s[4] for s in minibatch])
                
                Q1 = model_naive(state1_batch)
                with torch.no_grad():
                    Q2 = model_naive(state2_batch)
                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses_naive.append(loss.item())
                
            if reward != -1 or mov > 50:
                status = 0
        if epsilon > 0.1:
            epsilon -= (1/epochs)

    plt.figure(figsize=(8, 4))
    plt.plot(losses_naive, alpha=0.7, color='blue')
    plt.title('HW3-1 Naive DQN Loss Curve')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.savefig('loss_hw3_1.png', bbox_inches='tight')
    plt.close()
    
    game = Gridworld(size=4, mode='static')
    plot_and_save_grid(game.display(), 'grid_static_start.png', 'Static Mode - Start')
    status = 1
    moves = 0
    while status == 1 and moves < 15:
        state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
        qval = model_naive(state)
        action_ = np.argmax(qval.data.numpy())
        game.makeMove({0: 'u', 1: 'd', 2: 'l', 3: 'r'}[action_])
        if game.reward() == 10:
            plot_and_save_grid(game.display(), 'grid_static_win.png', 'Static Mode - Win!')
            break
        moves += 1

    print("Training Dueling DQN (Player Mode)...")
    model_dueling = DuelingDQN(64, 150, 100, 4)
    optimizer = torch.optim.Adam(model_dueling.parameters(), lr=1e-3)
    epsilon = 1.0
    epochs = 1200
    replay = deque(maxlen=1000)
    losses_dueling = []
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='player')
        state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
        status = 1
        mov = 0
        while status == 1:
            mov += 1
            qval = model_dueling(state)
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval.data.numpy())
            game.makeMove({0: 'u', 1: 'd', 2: 'l', 3: 'r'}[action_])
            next_state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
            reward = game.reward()
            done = True if reward != -1 else False
            replay.append((state, action_, np.float32(reward), next_state, np.float32(done)))
            state = next_state
            
            if len(replay) > 200:
                minibatch = random.sample(replay, 200)
                state1_batch = torch.cat([s[0] for s in minibatch])
                action_batch = torch.Tensor([s[1] for s in minibatch])
                reward_batch = torch.Tensor([s[2] for s in minibatch])
                state2_batch = torch.cat([s[3] for s in minibatch])
                done_batch = torch.Tensor([s[4] for s in minibatch])
                
                Q1 = model_dueling(state1_batch)
                with torch.no_grad():
                    Q2 = model_dueling(state2_batch)
                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses_dueling.append(loss.item())
                
            if reward != -1 or mov > 50:
                status = 0
        if epsilon > 0.1:
            epsilon -= (1/epochs)

    plt.figure(figsize=(8, 4))
    plt.plot(losses_dueling, alpha=0.7, color='green')
    plt.title('HW3-2 Dueling DQN Loss Curve')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.savefig('loss_hw3_2.png', bbox_inches='tight')
    plt.close()
    
    game = Gridworld(size=4, mode='player')
    plot_and_save_grid(game.display(), 'grid_player_start.png', 'Player Mode - Start')
    status = 1
    moves = 0
    while status == 1 and moves < 15:
        state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
        qval = model_dueling(state)
        action_ = np.argmax(qval.data.numpy())
        game.makeMove({0: 'u', 1: 'd', 2: 'l', 3: 'r'}[action_])
        if game.reward() == 10:
            plot_and_save_grid(game.display(), 'grid_player_win.png', 'Player Mode - Win!')
            break
        moves += 1

    print("Generating Random Mode Grids...")
    game = Gridworld(size=4, mode='random')
    plot_and_save_grid(game.display(), 'grid_random_start.png', 'Random Mode - Example')
    print("Done!")

if __name__ == '__main__':
    train_and_visualize()
