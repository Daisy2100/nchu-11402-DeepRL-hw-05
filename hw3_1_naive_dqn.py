import numpy as np
import torch
from Gridworld import Gridworld
import random
from collections import deque

def run_hw3_1():
    L1 = 64
    L2 = 150
    L3 = 100
    L4 = 4

    model = torch.nn.Sequential(
        torch.nn.Linear(L1, L2),
        torch.nn.ReLU(),
        torch.nn.Linear(L2, L3),
        torch.nn.ReLU(),
        torch.nn.Linear(L3, L4)
    )
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.9
    epsilon = 1.0

    action_set = {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r'
    }

    epochs = 1000 # Short epochs for static mode
    losses = []
    mem_size = 1000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    max_moves = 50

    for i in range(epochs):
        game = Gridworld(size=4, mode='static')
        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        mov = 0

        while status == 1:
            mov += 1
            qval = model(state1)
            qval_ = qval.data.numpy()
            if (random.random() < epsilon):
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)
            action = action_set[action_]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            done = True if reward != -1 else False
            exp = (state1, action_, reward, state2, done)
            replay.append(exp)
            state1 = state2

            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model(state2_batch)
                
                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if abs(reward) == 10 or mov > max_moves:
                status = 0
                mov = 0

        if epsilon > 0.1:
            epsilon -= (1 / epochs)

    print("HW3-1 (Naive DQN with Experience Replay on static mode) Finished Training!")

if __name__ == "__main__":
    run_hw3_1()
