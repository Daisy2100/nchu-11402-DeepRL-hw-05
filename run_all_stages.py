"""
HW3 — All 3 Stages: Static → Player → Random
tf.keras (Keras 3 compatible) + GradientTape custom training loops
No model.fit(). All training is manual.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, ops
from collections import deque
from Gridworld import Gridworld

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

ACTION_MAP = {0: "u", 1: "d", 2: "l", 3: "r"}
GAMMA      = 0.9

# ────────────────────────────────────────────────────────────────
# Shared Utilities
# ────────────────────────────────────────────────────────────────
def build_mlp(input_dim=64, hidden=(150, 100), output_dim=4, name="dqn"):
    """Standard 3-layer MLP used in Stage 1 & shared backbone in Stage 2/3."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden[0], activation="relu"),
        layers.Dense(hidden[1], activation="relu"),
        layers.Dense(output_dim),
    ], name=name)
    return model

def build_dueling(input_dim=64, shared_units=150, stream_units=100, output_dim=4, name="dueling"):
    """
    S4 Dueling Network.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    Implemented as a custom Keras Model (not Sequential) so the
    advantage-centering lambda is applied inside call(), which is
    fully Keras-3 compatible.
    """
    class DuelingModel(keras.Model):
        def __init__(self):
            super().__init__(name=name)
            self.shared = layers.Dense(shared_units, activation="relu")
            self.val_h  = layers.Dense(stream_units, activation="relu")
            self.val_o  = layers.Dense(1)
            self.adv_h  = layers.Dense(stream_units, activation="relu")
            self.adv_o  = layers.Dense(output_dim)

        def call(self, x, training=False):
            x   = self.shared(x)
            val = self.val_o(self.val_h(x))          # (B, 1)
            adv = self.adv_o(self.adv_h(x))          # (B, A)
            q   = val + adv - tf.reduce_mean(adv, axis=1, keepdims=True)
            return q

    return DuelingModel()

class ReplayBuffer:
    """S1 — Standard uniform replay buffer."""
    def __init__(self, maxlen):
        self.buf = deque(maxlen=maxlen)

    def push(self, s, a, r, s2, done):
        self.buf.append((s.astype(np.float32), int(a), np.float32(r),
                         s2.astype(np.float32), np.float32(done)))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s, np.float32), np.array(a, np.int32),
                np.array(r, np.float32), np.array(s2, np.float32),
                np.array(d, np.float32))

    def __len__(self): return len(self.buf)

class PrioritizedReplayBuffer:
    """S5 — PER: replay transitions with probability ∝ |TD-error|^alpha."""
    def __init__(self, maxlen, alpha=0.6, eps=1e-5):
        self.maxlen = maxlen; self.alpha = alpha; self.eps = eps
        self.buf = []; self.prios = []; self.pos = 0

    def push(self, s, a, r, s2, done):
        exp   = (s.astype(np.float32), int(a), np.float32(r),
                 s2.astype(np.float32), np.float32(done))
        max_p = max(self.prios, default=1.0)
        if len(self.buf) < self.maxlen:
            self.buf.append(exp); self.prios.append(max_p)
        else:
            self.buf[self.pos]  = exp; self.prios[self.pos] = max_p
        self.pos = (self.pos + 1) % self.maxlen

    def sample(self, n, beta=0.4):
        P  = np.array(self.prios, np.float32) ** self.alpha
        P /= P.sum()
        idx = np.random.choice(len(self.buf), n, replace=False, p=P)
        batch = [self.buf[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        w  = (len(self.buf) * P[idx]) ** (-beta)
        w /= w.max()
        return (np.array(s,np.float32), np.array(a,np.int32),
                np.array(r,np.float32), np.array(s2,np.float32),
                np.array(d,np.float32), w.astype(np.float32), idx)

    def update_priorities(self, idx, td_errors):
        for i, e in zip(idx, td_errors):
            self.prios[i] = float(abs(e)) + self.eps

    def __len__(self): return len(self.buf)

def get_state(game):
    return game.board.render_np().reshape(64).astype(np.float32) + \
           np.random.rand(64).astype(np.float32) / 100.0

def epsilon_greedy(model, state, epsilon, n_actions=4):
    if random.random() < epsilon:
        return np.random.randint(n_actions)
    q = model(state[None], training=False).numpy()[0]
    return int(np.argmax(q))

# ────────────────────────────────────────────────────────────────
# Stage 1: Static Mode — S1 Replay + S2 Target Network
# ────────────────────────────────────────────────────────────────
def stage1_static(epochs=1000, batch=200, sync_every=300, mem=1000, lr=1e-3):
    print("\n" + "="*60)
    print("  STAGE 1 | Static Mode | S1 Replay Buffer + S2 Target Network")
    print("="*60)
    online = build_mlp(name="s1_online")
    target = build_mlp(name="s1_target")
    target.set_weights(online.get_weights())
    opt    = keras.optimizers.Adam(lr)
    replay = ReplayBuffer(mem)
    eps    = 1.0; losses = []; step = 0

    for ep in range(epochs):
        game   = Gridworld(size=4, mode="static")
        state  = get_state(game)
        done   = False; mov = 0

        while not done:
            step += 1; mov += 1
            action = epsilon_greedy(online, state, eps)
            game.makeMove(ACTION_MAP[action])
            next_state = get_state(game)
            reward     = game.reward()
            terminal   = reward != -1
            replay.push(state, action, reward, next_state, terminal)
            state      = next_state

            if len(replay) >= batch:
                s, a, r, s2, d = replay.sample(batch)
                # S2: compute TD target using frozen target network
                nq     = target(s2, training=False)
                max_nq = tf.reduce_max(nq, axis=1)
                tgt    = r + GAMMA * (1.0 - d) * max_nq.numpy()

                with tf.GradientTape() as tape:
                    q_all  = online(s, training=True)
                    idx    = tf.stack([tf.range(batch), a], axis=1)
                    q_pred = tf.gather_nd(q_all, idx)
                    loss   = tf.reduce_mean(tf.square(tgt - q_pred))

                grads = tape.gradient(loss, online.trainable_variables)
                opt.apply_gradients(zip(grads, online.trainable_variables))
                losses.append(float(loss))

            # S2: periodic sync
            if step % sync_every == 0:
                target.set_weights(online.get_weights())

            if terminal or mov > 50:
                done = True

        if eps > 0.1: eps -= 0.9 / epochs
        if (ep + 1) % 200 == 0:
            print(f"  Ep {ep+1:4d}/{epochs} | eps={eps:.3f} | loss={losses[-1] if losses else 0:.4f}")

    print("[Done] Stage 1\n")
    return online, losses

# ────────────────────────────────────────────────────────────────
# Stage 2: Player Mode — + S3 Double DQN + S4 Dueling
# ────────────────────────────────────────────────────────────────
def stage2_player(epochs=2000, batch=200, sync_every=500, mem=2000, lr=1e-3):
    print("\n" + "="*60)
    print("  STAGE 2 | Player Mode | +S3 Double DQN + S4 Dueling DQN")
    print("="*60)
    online = build_dueling(name="s2_online")
    target = build_dueling(name="s2_target")
    # Force build before set_weights
    _ = online(np.zeros((1, 64), np.float32))
    _ = target(np.zeros((1, 64), np.float32))
    target.set_weights(online.get_weights())
    opt    = keras.optimizers.Adam(lr)
    replay = ReplayBuffer(mem)
    eps    = 1.0; losses = []; step = 0

    for ep in range(epochs):
        game   = Gridworld(size=4, mode="player")
        state  = get_state(game)
        done   = False; mov = 0

        while not done:
            step += 1; mov += 1
            action = epsilon_greedy(online, state, eps)
            game.makeMove(ACTION_MAP[action])
            next_state = get_state(game)
            reward     = game.reward()
            terminal   = reward != -1
            replay.push(state, action, reward, next_state, terminal)
            state      = next_state

            if len(replay) >= batch:
                s, a, r, s2, d = replay.sample(batch)
                # S3: online selects, target evaluates
                nq_online    = online(s2, training=False).numpy()
                best_actions = np.argmax(nq_online, axis=1)
                nq_target    = target(s2, training=False).numpy()
                q_next       = nq_target[np.arange(batch), best_actions]
                tgt          = r + GAMMA * (1.0 - d) * q_next

                with tf.GradientTape() as tape:
                    q_all  = online(s, training=True)
                    idx    = tf.stack([tf.range(batch), a], axis=1)
                    q_pred = tf.gather_nd(q_all, idx)
                    loss   = tf.reduce_mean(tf.square(tgt - q_pred))

                grads = tape.gradient(loss, online.trainable_variables)
                opt.apply_gradients(zip(grads, online.trainable_variables))
                losses.append(float(loss))

            if step % sync_every == 0:
                target.set_weights(online.get_weights())

            if terminal or mov > 50: done = True

        if eps > 0.1: eps -= 0.9 / epochs
        if (ep + 1) % 400 == 0:
            print(f"  Ep {ep+1:4d}/{epochs} | eps={eps:.3f} | loss={losses[-1] if losses else 0:.4f}")

    print("[Done] Stage 2\n")
    return online, losses

# ────────────────────────────────────────────────────────────────
# Stage 3: Random Mode — + S5 PER + Grad Clip + LR Schedule
# ────────────────────────────────────────────────────────────────
def stage3_random(epochs=3000, batch=256, sync_every=500, mem=3000):
    print("\n" + "="*60)
    print("  STAGE 3 | Random Mode | +S5 PER + Grad Clipping + LR Decay")
    print("="*60)
    online = build_dueling(name="s3_online")
    target = build_dueling(name="s3_target")
    _ = online(np.zeros((1, 64), np.float32))
    _ = target(np.zeros((1, 64), np.float32))
    target.set_weights(online.get_weights())

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.995)
    opt    = keras.optimizers.Adam(lr_schedule)
    replay = PrioritizedReplayBuffer(mem)
    eps    = 1.0; losses = []; step = 0; beta = 0.4

    for ep in range(epochs):
        game   = Gridworld(size=4, mode="random")
        state  = get_state(game)
        done   = False; mov = 0

        while not done:
            step += 1; mov += 1
            action = epsilon_greedy(online, state, eps)
            game.makeMove(ACTION_MAP[action])
            next_state = get_state(game)
            reward     = game.reward()
            terminal   = reward != -1
            replay.push(state, action, reward, next_state, terminal)
            state      = next_state

            beta = min(1.0, 0.4 + 0.6 * step / (epochs * 50))

            if len(replay) >= batch:
                s, a, r, s2, d, w, idxs = replay.sample(batch, beta)
                # S3 Double DQN target
                nq_online    = online(s2, training=False).numpy()
                best_actions = np.argmax(nq_online, axis=1)
                nq_target    = target(s2, training=False).numpy()
                q_next       = nq_target[np.arange(batch), best_actions]
                tgt          = r + GAMMA * (1.0 - d) * q_next

                with tf.GradientTape() as tape:
                    q_all  = online(s, training=True)
                    idx    = tf.stack([tf.range(batch), a], axis=1)
                    q_pred = tf.gather_nd(q_all, idx)
                    td_err = tgt - q_pred.numpy()
                    # S5: IS-weighted loss
                    loss   = tf.reduce_mean(w * tf.square(tgt - q_pred))

                grads = tape.gradient(loss, online.trainable_variables)
                # Gradient Clipping
                grads, _ = tf.clip_by_global_norm(grads, 1.0)
                opt.apply_gradients(zip(grads, online.trainable_variables))
                replay.update_priorities(idxs, td_err)
                losses.append(float(loss))

            if step % sync_every == 0:
                target.set_weights(online.get_weights())

            if terminal or mov > 50: done = True

        if eps > 0.1: eps -= 0.9 / epochs
        if (ep + 1) % 500 == 0:
            print(f"  Ep {ep+1:4d}/{epochs} | eps={eps:.3f} | loss={losses[-1] if losses else 0:.4f}")

    print("[Done] Stage 3\n")
    return online, losses

# ────────────────────────────────────────────────────────────────
# Visualisation helpers
# ────────────────────────────────────────────────────────────────
DARK_BG = "#1a1a2e"
PANEL   = "#16213e"

def plot_loss(losses, title, outfile, color):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(PANEL)
    ax.plot(losses, alpha=0.2, color=color)
    w = max(1, len(losses) // 80)
    if len(losses) >= w:
        sm = np.convolve(losses, np.ones(w)/w, mode="valid")
        ax.plot(range(w-1, len(losses)), sm, color=color, lw=2, label="Smoothed")
    ax.set_title(title, color="white", fontsize=13)
    ax.set_xlabel("Training Steps", color="#aaaacc")
    ax.set_ylabel("Loss (MSE)", color="#aaaacc")
    ax.tick_params(colors="#aaaacc")
    for sp in ax.spines.values(): sp.set_color("#4a4a6a")
    ax.legend(facecolor=PANEL, labelcolor="white")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, facecolor=DARK_BG); plt.close()
    print(f"Saved: {outfile}")

def render_grid(board_arr, outfile, title):
    cmap   = {" ":[0.1,0.1,0.2],"P":[0.22,0.45,0.99],
              "+":[0.18,0.78,0.45],"-":[0.90,0.25,0.28],"W":[0.3,0.3,0.4]}
    labels = {" ":"","P":"P\nPlayer","+":"+\nGoal","-":"-\nPit","W":"W\nWall"}
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
    img = np.array([[cmap[board_arr[i,j]] for j in range(4)] for i in range(4)])
    ax.imshow(img, aspect="equal")
    for x in range(5):
        ax.axhline(x-.5, color="#4a4a6a", lw=1.5)
        ax.axvline(x-.5, color="#4a4a6a", lw=1.5)
    for i in range(4):
        for j in range(4):
            c = board_arr[i,j]
            if c != " ":
                ax.text(j, i, labels[c], ha="center", va="center",
                        color="white", fontsize=15, fontweight="bold", linespacing=1.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color="white", fontsize=13, pad=8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(); print(f"Saved: {outfile}")

def snapshot(model, mode, f_start, f_end):
    game = Gridworld(size=4, mode=mode)
    render_grid(game.display(), f_start, f"{mode.capitalize()} — Initial State")
    moves = 0
    while moves < 20:
        s = get_state(game)
        a = int(np.argmax(model(s[None], training=False).numpy()[0]))
        game.makeMove(ACTION_MAP[a])
        r = game.reward()
        if r == 10:
            render_grid(game.display(), f_end, f"{mode.capitalize()} — WIN!")
            return
        elif r == -10:
            render_grid(game.display(), f_end, f"{mode.capitalize()} — Lost")
            return
        moves += 1
    render_grid(game.display(), f_end, f"{mode.capitalize()} — Timeout")

# ────────────────────────────────────────────────────────────────
# Main: run all 3 stages
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m1, l1 = stage1_static()
    plot_loss(l1, "Stage 1 | Naive DQN + S1 Replay + S2 Target | Static Mode",
              "loss_s1_static.png", "#5599ff")
    snapshot(m1, "static", "grid_s1_start.png", "grid_s1_end.png")

    m2, l2 = stage2_player()
    plot_loss(l2, "Stage 2 | +S3 Double DQN + S4 Dueling | Player Mode",
              "loss_s2_player.png", "#55dd88")
    snapshot(m2, "player", "grid_s2_start.png", "grid_s2_end.png")

    m3, l3 = stage3_random()
    plot_loss(l3, "Stage 3 | +S5 PER + Grad Clipping + LR Decay | Random Mode",
              "loss_s3_random.png", "#ff9955")
    snapshot(m3, "random", "grid_s3_start.png", "grid_s3_end.png")

    print("\nAll stages done! All images saved.")
