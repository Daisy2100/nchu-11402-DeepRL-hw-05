import numpy as np
import torch
import time
import os
from Gridworld import Gridworld
from hw3_2_dueling_dqn import DuelingDQN
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')

def get_trained_model():
    L1, L2, L3, L4 = 64, 150, 100, 4
    model = DuelingDQN(L1, L2, L3, L4)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gamma, epsilon = 0.9, 1.0
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    epochs = 1500
    replay = deque(maxlen=1000)
    
    print("正在快速訓練一個 Dueling DQN 模型供 Demo 使用 (大約需要 5-10 秒)...")
    for i in range(epochs):
        game = Gridworld(size=4, mode='random')
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
    return model

def demo():
    model = get_trained_model()
    model.eval()
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    game = Gridworld(size=4, mode='random')
    
    print("\n--- Demo 開始！準備觀看 AI 玩遊戲 ---")
    time.sleep(2)
    status = 1
    moves = 0
    while status == 1 and moves < 20:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"移動步數: {moves}")
        print("符號說明: P=玩家, +=目標, -=陷阱, W=牆壁\n")
        print(game.display())
        time.sleep(1.0)
        
        state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0).float()
        qval = model(state)
        action_ = np.argmax(qval.data.numpy())
        print(f"AI 選擇的動作: {action_set[action_]}")
        time.sleep(0.5)
        
        game.makeMove(action_set[action_])
        reward = game.reward()
        if reward == 10:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(game.display())
            print("🎉 AI 獲勝！成功抵達目標 (+)")
            break
        elif reward == -10:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(game.display())
            print("💀 AI 失敗！掉入陷阱 (-)")
            break
            
        moves += 1
        
if __name__ == '__main__':
    demo()
