# Homework 3: DQN and its variants

這份作業包含深度強化學習中最經典的 DQN 演算法及其進階變體的實作，並使用 Gridworld 遊戲環境進行測試。

## 執行環境與前置準備

在執行程式碼之前，請確保安裝了以下套件：
```bash
pip install numpy torch pytorch-lightning matplotlib
```

所有任務的執行檔均已編寫完畢，你可以透過以下指令獨立運行每個任務：
```bash
python hw3_1_naive_dqn.py
python hw3_2_double_dqn.py
python hw3_2_dueling_dqn.py
python hw3_3_lightning_dqn.py
```

---

## HW3-1：基礎 DQN 理解報告 (Naive DQN & Experience Replay Buffer)

### 執行腳本：`hw3_1_naive_dqn.py`

*   **基礎 DQN 實作邏輯**：
    我們實作了一個包含多個隱藏層的神經網路 `torch.nn.Sequential`。這個神經網路負責接收目前 Gridworld 環境的狀態 (State，轉換成長度 64 的向量) 作為輸入，然後輸出 4 個動作 (`u, d, l, r`) 對應的 Q 值 (Q-value)。我們會使用 $\epsilon$-greedy 策略，讓 AI 有一定機率探索隨機動作，或是根據最大 Q 值選擇最佳動作。
*   **Experience Replay Buffer (經驗回放池) 的功用**：
    在 `hw3_1_naive_dqn.py` 中，我們用 `collections.deque` 建立了一個 `replay` 串列。每當 AI 走了一步，就會產生一筆包含 `(當前狀態, 動作, 回饋值, 新狀態, 遊戲是否結束)` 的經驗並存入回放池。
    **功能與優點**：
    1.  **打亂時間相關性**：強化學習在玩遊戲時，連續取得的資料高度相關（例如：走到這格的下一步一定在隔壁）。如果直接用連續的資料訓練，神經網路很容易發生「災難性遺忘」或無法收斂。透過將經驗存起來再「隨機抽取（Random Sample）」，可以打破資料間的時間相關性。
    2.  **提高資料利用率**：同一筆過去的經驗，可以被反覆抽樣並用來多次訓練模型，大大提升學習效率。

---

## HW3-2：進階變體實作與比較 (Double DQN & Dueling DQN)

我們在 `player mode` (玩家初始位置隨機) 的環境下實作了兩種進階變體。

### 1. Double DQN (腳本：`hw3_2_double_dqn.py`)
*   **如何改善 DQN**：傳統 DQN 在計算目標 Q 值時，會使用同一個網路來「選擇最大動作」與「評估動作價值」，這導致容易選擇到剛好被高估的 Q 值（Overestimation）。
*   **實作細節**：我們準備了兩個網路：`model` (Online Network) 與 `model2` (Target Network)。計算目標 Q 值時，先用 `model` 選出在 $S'$ 狀態下 Q 值最高的動作 $a^*$，再用 `model2` 來評估該動作的真正價值 $Q(S', a^*)$。這種「選擇與評估分離」的機制，有效減少了 Q 值的過度估計，讓訓練更平穩。

### 2. Dueling DQN (腳本：`hw3_2_dueling_dqn.py`)
*   **如何改善 DQN**：傳統 DQN 直接用一個全連接層輸出每個動作的 Q 值，但在某些狀態下，其實「不管選什麼動作都差不多（例如：死路）」，此時區分具體動作的好壞意義不大。
*   **實作細節**：我們在 `DuelingDQN` 類別中，將神經網路倒數第二層分拆成了兩條獨立的路徑（Streams）：
    1.  **Value Stream**：負責評估目前「狀態本身的價值」 $V(s)$。
    2.  **Advantage Stream**：負責評估「各個動作比平均動作好多少的優勢」 $A(s, a)$。
    最後透過公式 $Q(s, a) = V(s) + A(s, a) - \text{mean}(A(s, a))$ 將兩者結合。這樣的架構能讓神經網路更精確地評估狀態的價值，並且在面對眾多動作選項的環境中學習得更快。

---

## HW3-3：架構轉換與訓練技巧 (PyTorch Lightning)

### 執行腳本：`hw3_3_lightning_dqn.py`

在挑戰難度最高的 `random mode`（目標、牆壁、陷阱和玩家全隨機）中，我們將原本的 PyTorch 程式重構成了高模組化的 **PyTorch Lightning** 框架。

*   **框架轉換優勢**：
    我們實作了 `pl.LightningModule` 類別與一個用於處理經驗回放的自訂 `IterableDataset`。這讓訓練邏輯（`training_step`）、優化器設定（`configure_optimizers`）變得極度簡潔，免去了原先又長又複雜的 `while` 迴圈與手動梯度歸零的操作。
*   **進階訓練技巧 (Bonus)**：
    1.  **Gradient Clipping (梯度裁剪)**：在 `pl.Trainer` 中設定了 `gradient_clip_val=1.0`。這能防止神經網路反向傳播時發生「梯度爆炸」，使權重更新維持在安全穩定的範圍內。
    2.  **LR Scheduling (學習率調度)**：在 `configure_optimizers` 內加入了 `StepLR`，讓學習率在訓練過程（每 1000 step）中自動乘上 `0.9` 衰減，幫助神經網路在訓練末期時，步伐變小以更容易精準收斂。
