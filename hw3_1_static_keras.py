"""
HW3 Stage 1 — Static Mode
Mechanisms: S1 Replay Buffer + S2 Target Network
Framework: TensorFlow / tf.keras + GradientTape

Environment Analysis
--------------------
- Layout is fully deterministic: Player always starts at (0,3), Goal at (0,0),
  Pit at (0,1), Wall at (1,1).
- Because the board never changes, a simple MLP can memorize the optimal path.

Training Instability Symptom (without mechanisms)
---------------------------------------------------
- Without Replay Buffer: consecutive transitions are highly correlated, causing
  the network to "forget" older experiences (catastrophic forgetting).
- Without Target Network: the TD target changes every step because the same network
  is used for both prediction AND target, creating a non-stationary learning signal
  that causes divergence.

Mechanisms Selected
--------------------
  S1 Replay Buffer  — breaks temporal correlation, improves sample efficiency
  S2 Target Network — stabilises the TD target, prevents divergence

Mechanisms Skipped (and why)
------------------------------
  S3 Double DQN  — overestimation is negligible in a fixed, small grid (4x4)
  S4 Dueling DQN — only 4 actions, value/advantage decomposition gives no benefit
  S5 PER         — uniform sampling works fine; the replay buffer won't saturate
                   with uninformative transitions in static mode
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF logs

import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from Gridworld import Gridworld

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
STATE_DIM   = 64
ACTION_DIM  = 4
LR          = 1e-3
GAMMA       = 0.9
EPSILON_START = 1.0
EPSILON_MIN   = 0.1
EPOCHS      = 1000
MEM_SIZE    = 1000
BATCH_SIZE  = 200
MAX_MOVES   = 50
SYNC_EVERY  = 300        # S2: copy online → target every N steps
ACTION_MAP  = {0: "u", 1: "d", 2: "l", 3: "r"}

# ── S1: Replay Buffer ──────────────────────────────────────────────────────────
class ReplayBuffer:
    """Fixed-size circular buffer storing (s, a, r, s', done) tuples."""
    def __init__(self, maxlen=MEM_SIZE):
        self.buf = deque(maxlen=maxlen)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s.astype(np.float32),
                         int(a),
                         float(r),
                         s_next.astype(np.float32),
                         float(done)))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r),
                np.array(s2), np.array(d))

    def __len__(self):
        return len(self.buf)

# ── Network builder ────────────────────────────────────────────────────────────
def build_model(state_dim=STATE_DIM, action_dim=ACTION_DIM):
    """3-layer MLP: same architecture used throughout HW3."""
    inp = tf.keras.Input(shape=(state_dim,))
    x   = tf.keras.layers.Dense(150, activation="relu")(inp)
    x   = tf.keras.layers.Dense(100, activation="relu")(x)
    out = tf.keras.layers.Dense(action_dim)(x)
    return tf.keras.Model(inp, out)

# ── One gradient-tape update step ─────────────────────────────────────────────
@tf.function
def train_step(online_model, target_model, optimizer, loss_fn,
               states, actions, rewards, next_states, dones):
    """
    TD target (S2): uses the frozen target network for next-state Q-values.
    target = r + γ * max_a Q_target(s', a)  if not done
    target = r                               if done
    """
    next_q   = target_model(next_states, training=False)      # S2 target network
    max_next = tf.reduce_max(next_q, axis=1)
    targets  = rewards + GAMMA * (1.0 - dones) * max_next

    with tf.GradientTape() as tape:
        q_all   = online_model(states, training=True)
        indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
        q_pred  = tf.gather_nd(q_all, indices)
        loss    = loss_fn(targets, q_pred)

    grads = tape.gradient(loss, online_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, online_model.trainable_variables))
    return loss

# ── Main training loop ─────────────────────────────────────────────────────────
def train():
    online_model = build_model()
    target_model = build_model()
    target_model.set_weights(online_model.get_weights())   # S2 init

    optimizer = tf.keras.optimizers.Adam(LR)
    loss_fn   = tf.keras.losses.MeanSquaredError()
    replay    = ReplayBuffer()

    epsilon   = EPSILON_START
    losses    = []
    step      = 0

    print("=" * 60)
    print(" Stage 1 | Static Mode | S1 Replay + S2 Target Network")
    print("=" * 60)

    for epoch in range(EPOCHS):
        game   = Gridworld(size=4, mode="static")
        state  = game.board.render_np().reshape(64) + np.random.rand(64) / 100
        status = 1
        mov    = 0

        while status == 1:
            step += 1; mov += 1

            # ε-greedy action selection
            if random.random() < epsilon:
                action = np.random.randint(ACTION_DIM)
            else:
                q = online_model(state[None], training=False).numpy()[0]
                action = int(np.argmax(q))

            game.makeMove(ACTION_MAP[action])
            next_state = game.board.render_np().reshape(64) + np.random.rand(64) / 100
            reward     = game.reward()
            done       = reward != -1

            # S1: store experience
            replay.push(state, action, reward, next_state, done)
            state = next_state

            # S1 + S2: mini-batch training
            if len(replay) >= BATCH_SIZE:
                s, a, r, s2, d = replay.sample(BATCH_SIZE)
                loss = train_step(
                    online_model, target_model, optimizer, loss_fn,
                    tf.constant(s, dtype=tf.float32),
                    tf.cast(a, tf.int32),
                    tf.constant(r, dtype=tf.float32),
                    tf.constant(s2, dtype=tf.float32),
                    tf.constant(d, dtype=tf.float32)
                )
                losses.append(float(loss))

            # S2: periodic target sync
            if step % SYNC_EVERY == 0:
                target_model.set_weights(online_model.get_weights())

            if done or mov > MAX_MOVES:
                status = 0

        # Decay epsilon
        if epsilon > EPSILON_MIN:
            epsilon -= (EPSILON_START - EPSILON_MIN) / EPOCHS

        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1:4d}/{EPOCHS} | "
                  f"ε={epsilon:.3f} | "
                  f"loss={losses[-1] if losses else 0:.4f}")

    print("\n[Done] Stage 1 training complete.\n")
    return online_model, losses

# ── Visualisation helpers ──────────────────────────────────────────────────────
def plot_loss(losses, outfile="loss_s1_static.png"):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#16213e")
    ax.plot(losses, alpha=0.3, color="#5599ff")
    w = 50
    if len(losses) >= w:
        smoothed = np.convolve(losses, np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, len(losses)), smoothed, color="#5599ff",
                linewidth=2, label="Smoothed Loss")
    ax.set_title("Stage 1 | Naive DQN + S1 + S2 | Static Mode Loss", color="white")
    ax.set_xlabel("Training Steps", color="#aaaacc")
    ax.set_ylabel("MSE Loss", color="#aaaacc")
    ax.tick_params(colors="#aaaacc")
    for sp in ax.spines.values(): sp.set_color("#4a4a6a")
    ax.legend(facecolor="#16213e", labelcolor="white")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, facecolor="#1a1a2e")
    plt.close()
    print(f"Saved: {outfile}")

def render_grid(board_arr, outfile, title, subtitle=""):
    cmap = {" ": [0.15,0.15,0.25], "P": [0.22,0.45,0.99],
            "+": [0.18,0.78,0.45], "-": [0.90,0.25,0.28], "W": [0.35,0.35,0.42]}
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#1a1a2e")
    img = np.array([[cmap[board_arr[i, j]] for j in range(4)] for i in range(4)])
    ax.imshow(img)
    for x in range(5):
        ax.axhline(x-.5, color="#4a4a6a", lw=1.5)
        ax.axvline(x-.5, color="#4a4a6a", lw=1.5)
    labels = {" ": "", "P": "P\nPlayer", "+": "+\nGoal",
              "-": "-\nPit", "W": "W\nWall"}
    for i in range(4):
        for j in range(4):
            c = board_arr[i, j]
            if c != " ":
                ax.text(j, i, labels[c], ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold", linespacing=1.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color="white", fontsize=13, pad=8)
    if subtitle:
        ax.set_xlabel(subtitle, color="#aaaacc", fontsize=10)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"Saved: {outfile}")

def snapshot(model, mode, outfile_start, outfile_end):
    game = Gridworld(size=4, mode=mode)
    render_grid(game.display(), outfile_start, f"{mode.capitalize()} — Start")
    status, moves = 1, 0
    while status == 1 and moves < 20:
        s = game.board.render_np().reshape(64) + np.random.rand(64) / 100
        a = int(np.argmax(model(s[None], training=False).numpy()[0]))
        game.makeMove(ACTION_MAP[a])
        reward = game.reward()
        if reward == 10:
            render_grid(game.display(), outfile_end, f"{mode.capitalize()} — Win!")
            return True
        elif reward == -10:
            render_grid(game.display(), outfile_end, f"{mode.capitalize()} — Lost")
            return False
        moves += 1
    render_grid(game.display(), outfile_end, f"{mode.capitalize()} — Timeout")
    return False

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, losses = train()
    plot_loss(losses)
    snapshot(model, "static", "grid_s1_start.png", "grid_s1_end.png")
    print("Stage 1 complete. Run hw3_2_player_keras.py next.")
