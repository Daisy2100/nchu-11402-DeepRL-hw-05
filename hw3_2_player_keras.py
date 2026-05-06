"""
HW3 Stage 2 — Player Mode
Mechanisms: S1+S2 (inherited) + S3 Double DQN + S4 Dueling DQN
Framework: TensorFlow / tf.keras + GradientTape

Environment Analysis
--------------------
- Player spawns at a RANDOM position; Goal/Pit/Wall remain fixed.
- The agent must generalise across ~12 possible starting positions.

Training Instability Symptoms (with only S1+S2)
------------------------------------------------
- Symptom 1 — Overestimation Bias (why S3 is needed):
    In vanilla DQN, the same network SELECTS and EVALUATES the best next action.
    This causes Q-values to be optimistically biased; with multiple start positions
    the accumulated error causes Q-values to diverge.
- Symptom 2 — Slow state-value learning (why S4 is needed):
    When the player is far from any piece, all 4 actions have nearly equal value.
    The network wastes capacity discriminating identical actions instead of first
    learning "how good is this board state?". Dueling DQN fixes this via V(s)/A(s,a) split.

Mechanisms Skipped
-------------------
  S5 PER — uniform sampling is sufficient; player-mode transitions are diverse
            enough that there is no severe imbalance of informative experiences.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from Gridworld import Gridworld

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

STATE_DIM  = 64
ACTION_DIM = 4
LR         = 1e-3
GAMMA      = 0.9
EPSILON_START = 1.0
EPSILON_MIN   = 0.1
EPOCHS     = 2000
MEM_SIZE   = 2000
BATCH_SIZE = 200
MAX_MOVES  = 50
SYNC_EVERY = 500
ACTION_MAP = {0: "u", 1: "d", 2: "l", 3: "r"}


class ReplayBuffer:
    def __init__(self, maxlen=MEM_SIZE):
        self.buf = deque(maxlen=maxlen)

    def push(self, s, a, r, s2, done):
        self.buf.append((s.astype(np.float32), int(a), float(r),
                         s2.astype(np.float32), float(done)))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r), np.array(s2), np.array(d))

    def __len__(self):
        return len(self.buf)


def build_dueling_model():
    """S4 Dueling Network: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))"""
    inp    = tf.keras.Input(shape=(STATE_DIM,))
    shared = tf.keras.layers.Dense(150, activation="relu")(inp)
    val    = tf.keras.layers.Dense(100, activation="relu")(shared)
    val    = tf.keras.layers.Dense(1)(val)
    adv    = tf.keras.layers.Dense(100, activation="relu")(shared)
    adv    = tf.keras.layers.Dense(ACTION_DIM)(adv)
    q      = val + adv - tf.reduce_mean(adv, axis=1, keepdims=True)
    return tf.keras.Model(inp, q)


@tf.function
def train_step(online, target, optimizer, loss_fn,
               states, actions, rewards, next_states, dones):
    """S3 Double DQN: online selects action, target evaluates it."""
    nq_online    = online(next_states, training=False)
    best_actions = tf.cast(tf.argmax(nq_online, axis=1), tf.int32)
    nq_target    = target(next_states, training=False)
    idx_t  = tf.stack([tf.range(tf.shape(best_actions)[0]), best_actions], axis=1)
    q_next = tf.gather_nd(nq_target, idx_t)
    targets = rewards + GAMMA * (1.0 - dones) * q_next

    with tf.GradientTape() as tape:
        q_all  = online(states, training=True)
        idx    = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
        q_pred = tf.gather_nd(q_all, idx)
        loss   = loss_fn(targets, q_pred)

    grads = tape.gradient(loss, online.trainable_variables)
    optimizer.apply_gradients(zip(grads, online.trainable_variables))
    return loss


def train():
    online = build_dueling_model()
    target = build_dueling_model()
    target.set_weights(online.get_weights())

    optimizer = tf.keras.optimizers.Adam(LR)
    loss_fn   = tf.keras.losses.MeanSquaredError()
    replay    = ReplayBuffer()

    epsilon, losses, step = EPSILON_START, [], 0

    print("=" * 60)
    print(" Stage 2 | Player Mode | S1+S2+S3+S4")
    print("=" * 60)

    for epoch in range(EPOCHS):
        game   = Gridworld(size=4, mode="player")
        state  = game.board.render_np().reshape(64) + np.random.rand(64) / 100
        status, mov = 1, 0

        while status == 1:
            step += 1; mov += 1
            action = np.random.randint(ACTION_DIM) if random.random() < epsilon \
                     else int(np.argmax(online(state[None], training=False).numpy()[0]))
            game.makeMove(ACTION_MAP[action])
            next_state = game.board.render_np().reshape(64) + np.random.rand(64) / 100
            reward     = game.reward()
            done       = reward != -1
            replay.push(state, action, reward, next_state, done)
            state      = next_state

            if len(replay) >= BATCH_SIZE:
                s, a, r, s2, d = replay.sample(BATCH_SIZE)
                loss = train_step(online, target, optimizer, loss_fn,
                                  tf.constant(s, dtype=tf.float32),
                                  tf.cast(a, tf.int32),
                                  tf.constant(r, dtype=tf.float32),
                                  tf.constant(s2, dtype=tf.float32),
                                  tf.constant(d, dtype=tf.float32))
                losses.append(float(loss))

            if step % SYNC_EVERY == 0:
                target.set_weights(online.get_weights())

            if done or mov > MAX_MOVES:
                status = 0

        if epsilon > EPSILON_MIN:
            epsilon -= (EPSILON_START - EPSILON_MIN) / EPOCHS

        if (epoch + 1) % 400 == 0:
            print(f"  Epoch {epoch+1:4d}/{EPOCHS} | e={epsilon:.3f} | "
                  f"loss={losses[-1] if losses else 0:.4f}")

    print("\n[Done] Stage 2 complete.\n")
    return online, losses


def plot_loss(losses, outfile="loss_s2_player.png"):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#16213e")
    ax.plot(losses, alpha=0.3, color="#55dd88")
    w = 80
    if len(losses) >= w:
        sm = np.convolve(losses, np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, len(losses)), sm, color="#55dd88",
                linewidth=2, label="Smoothed Loss")
    ax.set_title("Stage 2 | S3 Double + S4 Dueling DQN | Player Mode", color="white")
    ax.set_xlabel("Training Steps", color="#aaaacc")
    ax.set_ylabel("MSE Loss", color="#aaaacc")
    ax.tick_params(colors="#aaaacc")
    for sp in ax.spines.values(): sp.set_color("#4a4a6a")
    ax.legend(facecolor="#16213e", labelcolor="white")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, facecolor="#1a1a2e"); plt.close()
    print(f"Saved: {outfile}")


def render_grid(board_arr, outfile, title):
    cmap   = {" ":[0.15,0.15,0.25],"P":[0.22,0.45,0.99],
              "+":[0.18,0.78,0.45],"-":[0.90,0.25,0.28],"W":[0.35,0.35,0.42]}
    labels = {" ":"","P":"P\nPlayer","+":"+\nGoal","-":"-\nPit","W":"W\nWall"}
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#1a1a2e")
    img = np.array([[cmap[board_arr[i,j]] for j in range(4)] for i in range(4)])
    ax.imshow(img)
    for x in range(5):
        ax.axhline(x-.5, color="#4a4a6a", lw=1.5)
        ax.axvline(x-.5, color="#4a4a6a", lw=1.5)
    for i in range(4):
        for j in range(4):
            c = board_arr[i, j]
            if c != " ":
                ax.text(j, i, labels[c], ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold", linespacing=1.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color="white", fontsize=13, pad=8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(); print(f"Saved: {outfile}")


def snapshot(model, mode, out_start, out_end):
    game = Gridworld(size=4, mode=mode)
    render_grid(game.display(), out_start, f"{mode.capitalize()} — Start")
    status, moves = 1, 0
    while status == 1 and moves < 20:
        s = game.board.render_np().reshape(64) + np.random.rand(64) / 100
        a = int(np.argmax(model(s[None], training=False).numpy()[0]))
        game.makeMove(ACTION_MAP[a])
        r = game.reward()
        label = f"{mode.capitalize()} — Win!" if r == 10 else \
                f"{mode.capitalize()} — Lost" if r == -10 else None
        if label:
            render_grid(game.display(), out_end, label)
            return r == 10
        moves += 1
    render_grid(game.display(), out_end, f"{mode.capitalize()} — Timeout")
    return False


if __name__ == "__main__":
    model, losses = train()
    plot_loss(losses)
    snapshot(model, "player", "grid_s2_start.png", "grid_s2_end.png")
    print("Stage 2 complete. Run hw3_3_random_keras.py next.")
