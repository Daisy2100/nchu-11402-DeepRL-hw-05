# 📘 Homework 3: DQN and its variants

**深度強化學習 HW3 — 學習報告 | Deep Reinforcement Learning HW3 — Understanding Report**

> 作業目標：在 Gridworld 環境中，從最基本的 Naive DQN 開始，逐步實作 Double DQN、Dueling DQN，並最終透過 PyTorch Lightning 框架加入訓練技巧，應對難度最高的隨機模式。

---

## 📂 1. 環境說明 (Setup & Reference)

實作基礎來自 [DRL in Action GitHub Repo](https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master) 的 `Gridworld` 環境。

**棋盤符號說明：**

| 符號 | 名稱 | 意義 |
|:---:|:---:|:---|
| `P` | Player（玩家） | 受訓練 AI 控制，可上下左右移動 |
| `+` | Goal（目標） | 碰到獲得 **+10** 獎勵，遊戲獲勝 |
| `-` | Pit（陷阱） | 碰到獲得 **-10** 懲罰，遊戲失敗 |
| `W` | Wall（牆壁） | 無法穿越 |
|  ` ` | Empty | 每走一步獲得 **-1** 懲罰，鼓勵 AI 走最短路徑 |

**三種模式難度說明：**

| 模式 | Player 位置 | 其他物件位置 | 說明 |
|:---:|:---:|:---:|:---|
| `static` | 固定 (0,3) | 固定（Goal→(0,0), Pit→(0,1), Wall→(1,1)）| 最簡單，HW3-1 測試邏輯正確性 |
| `player` | **隨機** | 固定 | 中等難度，測試策略的泛化能力 |
| `random` | 隨機 | **全部隨機** | 最難，HW3-3 挑戰通用策略學習 |

---

## 🧠 2. HW3-1: Naive DQN (Static Mode) [30%]

### 執行腳本：`hw3_1_naive_dqn.py`

#### 網路架構
使用三個線性層的全連接神經網路 (Fully Connected Neural Network)：
```
Input (64) → Linear(150) → ReLU → Linear(100) → ReLU → Output (4 Q-values)
```
輸入是將 4×4×4 的棋盤狀態展平成長度 64 的向量，輸出是 4 個方向動作 (↑↓←→) 各自的 Q-Value。

#### 核心概念理解：Experience Replay Buffer（經驗回放池）
這是 DQN 中最關鍵的設計之一，使用 `collections.deque` 實作。

**為什麼需要它？**
在強化學習中，AI 連續玩遊戲所產生的資料具有高度的時間相關性（例如：連續兩步的棋盤畫面幾乎一模一樣）。若直接使用這種連續資料訓練，神經網路會發生以下兩個問題：
1. **相關性問題**：網路只學到「目前狀態的附近局面怎麼走」，而非真正的策略。
2. **災難性遺忘 (Catastrophic Forgetting)**：新學的覆蓋舊的，模型無法累積長期知識。

**Replay Buffer 如何解決？**
每一步將經驗 `(s, a, r, s', done)` 存入緩衝區，訓練時**隨機取樣 (Random Sampling)** mini-batch。這樣做有兩大好處：
- 打破時間相關性，讓每批訓練資料更接近 i.i.d. 分佈。
- 同一筆好的經驗可以被反覆使用，大幅提升**樣本效率 (Sample Efficiency)**。

#### 訓練成果截圖 — Static Mode

<p align="center">
  <img src="loss_hw3_1_static.png" alt="HW3-1 Loss Curve" width="600"/>
</p>

<p align="center">
  <strong>開始狀態</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>結束狀態</strong>
</p>
<p align="center">
  <img src="grid_static_start.png" alt="Static Start" width="280"/>
  &nbsp;&nbsp;&nbsp;➡️&nbsp;&nbsp;&nbsp;
  <img src="grid_static_result.png" alt="Static Result" width="280"/>
</p>

---

## ⚖️ 3. HW3-2: Enhanced DQN Variants (Player Mode) [40%]

在 `player` 模式下，玩家初始位置隨機，其他物件固定，難度提升，需要策略具備一定的泛化能力。

### A. Double DQN (`hw3_2_double_dqn.py`)

**傳統 DQN 的問題 — 過度估計 (Overestimation Bias)：**
標準 DQN 在計算目標 Q 值時，用同一個網路既「選擇最佳動作」又「評估該動作的價值」：
```
Target = r + γ * max Q(s', a')   ← 同一個網路做兩件事
```
這容易造成剛好被高估的動作不斷被選中，讓學習方向出現偏差。

**Double DQN 的解決方案 — 選擇與評估分離：**
```
a* = argmax Q_online(s', a)         ← Online Network 負責選動作
Target = r + γ * Q_target(s', a*)   ← Target Network 負責評估價值
```
用**兩個不同的網路**（Online / Target）分工合作，有效降低估值偏差，讓 Q-Value 的學習更加穩健。

---

### B. Dueling DQN (`hw3_2_dueling_dqn.py`)

**傳統 DQN 的問題 — 對每個動作強制估值：**
在某些狀態下，不管選什麼動作後果都差不多（例如在空地移動）。此時每個動作的 Q 值差異極小，但傳統 DQN 仍強迫分辨，學習效率低下。

**Dueling DQN 的解決方案 — 架構分流：**
```
                  ┌─────────────────────┐
Input → Shared →  │ Value Stream:  V(s) │ ─┐
  Layer           │ Adv. Stream: A(s,a) │  │→ Q(s,a) = V(s) + A(s,a) - mean(A)
                  └─────────────────────┘ ─┘
```

- **Value Stream $V(s)$**：學習「這個棋盤局面本身值多少分」。
- **Advantage Stream $A(s,a)$**：學習「選這個動作比平均好多少」。
- **公式**：$Q(s,a) = V(s) + A(s,a) - \text{mean}(A(s,a))$

這讓 $V(s)$ 能夠更頻繁地更新，特別在「動作選擇影響不大」的狀態下，學習效率顯著優於標準 DQN。

#### 訓練成果截圖 — Player Mode (Dueling DQN)

<p align="center">
  <img src="loss_hw3_2_player.png" alt="HW3-2 Loss Curve" width="600"/>
</p>

<p align="center">
  <strong>開始狀態（玩家位置隨機）</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>結束狀態</strong>
</p>
<p align="center">
  <img src="grid_player_start.png" alt="Player Start" width="280"/>
  &nbsp;&nbsp;&nbsp;➡️&nbsp;&nbsp;&nbsp;
  <img src="grid_player_result.png" alt="Player Result" width="280"/>
</p>

---

## 🔁 4. HW3-3: Enhance DQN for Random Mode with Training Tips [30%]

`random` 模式是最高難度：Player、Goal、Pit、Wall 全部隨機生成，AI 必須學會通用策略。為了穩定訓練，我們將架構升級為 **PyTorch Lightning**，並加入進階訓練技巧。

### 執行腳本：`hw3_3_lightning_dqn.py`

#### 框架轉換：PyTorch → PyTorch Lightning
原始的 PyTorch 程式碼將環境互動、Loss 計算、梯度更新等全都混在一個巨大的 `while` 迴圈裡，難以維護。PyTorch Lightning 的好處：

| | 原本 PyTorch | PyTorch Lightning |
|:---:|:---:|:---:|
| 訓練邏輯 | 手動 `for/while` 迴圈 | 封裝在 `training_step` |
| 資料流 | 手動 Deque 取樣 | 自訂 `IterableDataset` |
| 優化器 | 手動 `optimizer.zero_grad() / step()` | 由 `configure_optimizers` 統一管理 |
| 進階技巧 | 需手動實作 | Trainer 參數一鍵啟用 |

#### 進階訓練技巧 (Bonus)

**① Gradient Clipping（梯度裁剪）**
```python
trainer = pl.Trainer(gradient_clip_val=1.0)
```
在隨機模式下，AI 可能遇到從未見過的極端棋局，產生極大的 Loss 值，導致「梯度爆炸」（權重在一次更新中跑飛）。設定 `gradient_clip_val=1.0` 後，每次反向傳播的梯度向量長度會被限制在 1 以內，保護訓練過程不崩潰。

**② LR Scheduling（學習率調度）**
```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
```
訓練初期需要較大的學習率快速探索，後期則需要較小的步長精準收斂。`StepLR` 讓學習率每 1000 步自動乘以 0.9 衰減，避免後期因步長太大而在最佳解附近來回震盪。

#### 訓練成果截圖 — Random Mode

<p align="center">
  <img src="loss_hw3_3_random.png" alt="HW3-3 Loss Curve" width="600"/>
</p>

<p align="center">
  <strong>開始狀態（全部隨機配置）</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>結束狀態</strong>
</p>
<p align="center">
  <img src="grid_random_start.png" alt="Random Start" width="280"/>
  &nbsp;&nbsp;&nbsp;➡️&nbsp;&nbsp;&nbsp;
  <img src="grid_random_result.png" alt="Random Result" width="280"/>
</p>

---

## 🎮 如何親自執行？

安裝相依套件：
```bash
pip install numpy torch pytorch-lightning matplotlib
```

執行各任務：
```bash
python hw3_1_naive_dqn.py    # HW3-1: Naive DQN (Static Mode)
python hw3_2_double_dqn.py   # HW3-2: Double DQN (Player Mode)
python hw3_2_dueling_dqn.py  # HW3-2: Dueling DQN (Player Mode)
python hw3_3_lightning_dqn.py # HW3-3: PyTorch Lightning (Random Mode)
python demo.py               # 動畫 Demo：看 AI 在終端機裡即時走迷宮！
```

---

## 📊 三種模式比較總結

| | HW3-1 Static | HW3-2 Player | HW3-3 Random |
|:---:|:---:|:---:|:---:|
| **難度** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **模型** | Naive DQN | Double + Dueling DQN | Dueling DQN (Lightning) |
| **訓練技巧** | Experience Replay | Replay + Target Network | + Gradient Clipping + LR Scheduling |
| **改善重點** | 奠定基礎 | 解決高估偏差 + 架構分流 | 框架重構 + 穩定訓練 |
