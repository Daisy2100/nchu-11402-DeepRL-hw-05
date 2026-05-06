import numpy as np
import torch
from Gridworld import Gridworld
from hw3_2_dueling_dqn import DuelingDQN
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')

def train_and_get_trace(mode, model_type='naive'):
    if model_type == 'naive':
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 150), torch.nn.ReLU(),
            torch.nn.Linear(150, 100), torch.nn.ReLU(),
            torch.nn.Linear(100, 4)
        )
    else:
        model = DuelingDQN(64, 150, 100, 4)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gamma, epsilon = 0.9, 1.0
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    epochs = 1500
    replay = deque(maxlen=1000)
    
    for i in range(epochs):
        game = Gridworld(size=4, mode=mode)
        state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
        status = 1
        mov = 0
        while status == 1:
            mov += 1
            qval = model(state)
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval.data.numpy())
            game.makeMove(action_set[action_])
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
                
                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model(state2_batch)
                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if reward != -1 or mov > 50:
                status = 0
        if epsilon > 0.1:
            epsilon -= (1/epochs)

    # Now generate trace
    game = Gridworld(size=4, mode=mode)
    trace = [str(game.display())]
    status = 1
    moves = 0
    while status == 1 and moves < 15:
        state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
        qval = model(state)
        action_ = np.argmax(qval.data.numpy())
        game.makeMove(action_set[action_])
        reward = game.reward()
        trace.append(f"Action: {action_set[action_]}\n" + str(game.display()))
        if reward == 10:
            trace.append("Result: AI Won (+)")
            break
        elif reward == -10:
            trace.append("Result: AI Lost (-)")
            break
        moves += 1
    return trace

traces = []
for mode in ['static', 'player', 'random']:
    t = train_and_get_trace(mode, 'dueling' if mode == 'random' else 'naive')
    traces.append(f"### Mode: {mode.capitalize()}\n```text\n" + "\n\n".join(t) + "\n```\n")

with open("traces.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(traces))
