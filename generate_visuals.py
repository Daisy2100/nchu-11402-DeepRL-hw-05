import numpy as np
import torch
import matplotlib.pyplot as plt
from Gridworld import Gridworld
from collections import deque
import random
import copy
from hw3_2_dueling_dqn import DuelingDQN
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'

def plot_grid(board_array, filename, title, subtitle=""):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    color_map = {
        ' ': [0.15, 0.15, 0.25],
        'P': [0.22, 0.45, 0.99],  # blue - player
        '+': [0.18, 0.78, 0.45],  # green - goal
        '-': [0.90, 0.25, 0.28],  # red - pit
        'W': [0.35, 0.35, 0.42],  # grey - wall
    }
    label_map = {' ': '', 'P': 'P\nPlayer', '+': '+\nGoal', '-': '-\nPit', 'W': 'W\nWall'}

    img = np.zeros((4, 4, 3))
    for i in range(4):
        for j in range(4):
            img[i, j] = color_map[board_array[i, j]]

    ax.imshow(img, aspect='equal')

    # Draw grid lines
    for x in range(5):
        ax.axhline(x - 0.5, color='#4a4a6a', linewidth=1.5)
        ax.axvline(x - 0.5, color='#4a4a6a', linewidth=1.5)

    for i in range(4):
        for j in range(4):
            char = board_array[i, j]
            if char != ' ':
                ax.text(j, i, label_map[char], ha='center', va='center',
                        color='white', fontsize=16, fontweight='bold',
                        linespacing=1.3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=15, color='white', pad=10)
    if subtitle:
        ax.set_xlabel(subtitle, color='#aaaacc', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150, facecolor='#1a1a2e')
    plt.close()
    print(f"Saved: {filename}")

def train_naive(mode, epochs=800):
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 150), torch.nn.ReLU(),
        torch.nn.Linear(150, 100), torch.nn.ReLU(),
        torch.nn.Linear(100, 4)
    )
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gamma, epsilon = 0.9, 1.0
    replay = deque(maxlen=1000)
    losses = []
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

    for i in range(epochs):
        game = Gridworld(size=4, mode=mode)
        state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
        status, mov = 1, 0
        while status == 1:
            mov += 1
            qval = model(state)
            action_ = np.random.randint(0, 4) if random.random() < epsilon else np.argmax(qval.data.numpy())
            game.makeMove(action_set[action_])
            next_state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
            reward = game.reward()
            done = reward != -1
            replay.append((state, action_, np.float32(reward), next_state, np.float32(done)))
            state = next_state
            if len(replay) > 200:
                mb = random.sample(replay, 200)
                s1b = torch.cat([s[0] for s in mb])
                ab = torch.Tensor([s[1] for s in mb])
                rb = torch.Tensor([s[2] for s in mb])
                s2b = torch.cat([s[3] for s in mb])
                db = torch.Tensor([s[4] for s in mb])
                Q1 = model(s1b)
                with torch.no_grad():
                    Q2 = model(s2b)
                Y = rb + gamma * ((1 - db) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(1, ab.long().unsqueeze(1)).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(loss.item())
            if reward != -1 or mov > 50:
                status = 0
        if epsilon > 0.1:
            epsilon -= 1 / epochs

    return model, losses

def train_dueling(mode, epochs=1200):
    model = DuelingDQN(64, 150, 100, 4)
    model2 = copy.deepcopy(model)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gamma, epsilon = 0.9, 1.0
    replay = deque(maxlen=1000)
    losses = []
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    j = 0

    for i in range(epochs):
        game = Gridworld(size=4, mode=mode)
        state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
        status, mov = 1, 0
        while status == 1:
            mov += 1; j += 1
            qval = model(state)
            action_ = np.random.randint(0, 4) if random.random() < epsilon else np.argmax(qval.data.numpy())
            game.makeMove(action_set[action_])
            next_state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
            reward = game.reward()
            done = reward != -1
            replay.append((state, action_, np.float32(reward), next_state, np.float32(done)))
            state = next_state
            if len(replay) > 200:
                mb = random.sample(replay, 200)
                s1b = torch.cat([s[0] for s in mb])
                ab = torch.Tensor([s[1] for s in mb])
                rb = torch.Tensor([s[2] for s in mb])
                s2b = torch.cat([s[3] for s in mb])
                db = torch.Tensor([s[4] for s in mb])
                Q1 = model(s1b)
                with torch.no_grad():
                    Q2 = model2(s2b)
                Y = rb + gamma * ((1 - db) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(1, ab.long().unsqueeze(1)).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                if j % 500 == 0:
                    model2.load_state_dict(model.state_dict())
                losses.append(loss.item())
            if reward != -1 or mov > 50:
                status = 0
        if epsilon > 0.1:
            epsilon -= 1 / epochs

    return model, losses

def get_game_sequence(model, mode, action_set={0: 'u', 1: 'd', 2: 'l', 3: 'r'}):
    game = Gridworld(size=4, mode=mode)
    frames = [game.display().copy()]
    status, moves = 1, 0
    won = False
    while status == 1 and moves < 20:
        state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
        action_ = np.argmax(model(state).data.numpy())
        game.makeMove(action_set[action_])
        frames.append(game.display().copy())
        reward = game.reward()
        if reward == 10:
            won = True
            status = 0
        elif reward == -10:
            status = 0
        moves += 1
    return frames, won

def save_loss_curve(losses, filename, title, color):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    smoothed = np.convolve(losses, np.ones(20)/20, mode='valid') if len(losses) > 20 else losses
    ax.plot(losses, alpha=0.25, color=color)
    ax.plot(range(len(smoothed)), smoothed, color=color, linewidth=2, label='Smoothed Loss')
    ax.set_title(title, color='white', fontsize=14)
    ax.set_xlabel('Training Steps', color='#aaaacc')
    ax.set_ylabel('Loss (MSE)', color='#aaaacc')
    ax.tick_params(colors='#aaaacc')
    for spine in ax.spines.values():
        spine.set_color('#4a4a6a')
    ax.legend(facecolor='#16213e', labelcolor='white')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor='#1a1a2e')
    plt.close()
    print(f"Saved: {filename}")

if __name__ == '__main__':
    print("=== [1/3] Training Naive DQN - Static Mode ===")
    model_static, losses_static = train_naive('static', epochs=800)
    save_loss_curve(losses_static, 'loss_hw3_1_static.png', 'HW3-1 | Naive DQN - Static Mode | Loss Curve', '#5599ff')
    frames, won = get_game_sequence(model_static, 'static')
    plot_grid(frames[0], 'grid_static_start.png', 'Static Mode — Start', 'P starts at (0,3) — fixed layout')
    plot_grid(frames[-1], 'grid_static_result.png', f'Static Mode — {"Win! 🎉" if won else "Result"}', 'AI navigated to Goal (+)')

    print("=== [2/3] Training Dueling DQN - Player Mode ===")
    model_player, losses_player = train_dueling('player', epochs=1200)
    save_loss_curve(losses_player, 'loss_hw3_2_player.png', 'HW3-2 | Dueling DQN - Player Mode | Loss Curve', '#55dd88')
    frames, won = get_game_sequence(model_player, 'player')
    plot_grid(frames[0], 'grid_player_start.png', 'Player Mode — Start', 'P starts at random position; others fixed')
    plot_grid(frames[-1], 'grid_player_result.png', f'Player Mode — {"Win! 🎉" if won else "Result"}', 'AI generalizes across starting positions')

    print("=== [3/3] Training Dueling DQN - Random Mode ===")
    model_random, losses_random = train_dueling('random', epochs=1500)
    save_loss_curve(losses_random, 'loss_hw3_3_random.png', 'HW3-3 | Dueling DQN - Random Mode | Loss Curve', '#ff9955')
    frames, won = get_game_sequence(model_random, 'random')
    plot_grid(frames[0], 'grid_random_start.png', 'Random Mode — Start', 'All pieces spawn at random positions')
    plot_grid(frames[-1], 'grid_random_result.png', f'Random Mode — {"Win! 🎉" if won else "Result"}', 'Hardest generalization challenge')

    print("\n✅ All images generated successfully!")
