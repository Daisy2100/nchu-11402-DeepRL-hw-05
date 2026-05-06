"""
HW3 Stage 3 — Random Mode
Mechanisms: S1+S2+S3+S4 (inherited) + S5 Prioritized Experience Replay
            + Gradient Clipping + LR Scheduling
Framework: TensorFlow / tf.keras + GradientTape

Environment Analysis
--------------------
- ALL pieces (Player, Goal, Pit, Wall) are placed randomly each episode.
- The agent cannot memorize any fixed layout; it must learn a truly general
  "avoid red, reach green" policy.

Training Instability Symptoms (with only S1-S4)
------------------------------------------------
- Symptom 1 — Sparse informative transitions (why S5 is needed):
    With random layouts, the agent rarely finds the goal early in training.
    The replay buffer fills mostly with boring (-1 reward) transitions.
    Uniform sampling means the rare +10/-10 experiences are diluted and
    contribute little to the gradient, causing extremely slow convergence.

- Symptom 2 — Gradient explosion on unseen states:
    Randomly arranged boards can produce large TD errors on the first encounter.
    Without clipping, a single catastrophic update can corrupt learned weights.

- Symptom 3 — Learning rate too coarse near convergence:
    In the final training phase, a fixed LR causes the policy to oscillate
    rather than settle. LR scheduling helps the agent fine-tune.

Mechanisms Selected
--------------------
  S5 PER              — prioritises high-TD-error transitions so rare goal/pit
                        encounters are replayed more frequently
  Gradient Clipping   — limits gradient norm to 1.0, preventing catastrophic
                        weight updates on unseen board configurations
  LR Scheduling       — exponential decay of LR ensures fine convergence

Mechanisms Already Inherited
------------------------------
  S1 Replay Buffer    — from Stage 1
  S2 Target Network   — from Stage 1
  S3 Double DQN       — from Stage 2
  S4 Dueling Network  — from Stage 2
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
LR_INIT    = 1e-3
LR_DECAY   = 0.995
GAMMA      = 0.9
EPSILON_START = 1.0
EPSILON_MIN   = 0.1
EPOCHS     = 3000
MEM_SIZE   = 3000
BATCH_SIZE = 256
MAX_MOVES  = 50
SYNC_EVERY = 500
CLIP_NORM  = 1.0          # gradient clipping
PER_ALPHA  = 0.6          # S5 PER priority exponent
PER_BETA   = 0.4          # S5 PER importance-sampling exponent
PER_EPS    = 1e-5         # S5 small constant to avoid zero priority
ACTION_MAP = {0: "u", 1: "d", 2: "l", 3: "r"}


# ── S5: Prioritized Experience Replay ─────────────────────────────────────────
class PrioritizedReplayBuffer:
    """
    Each transition is stored with a priority proportional to its TD error.
    High-error transitions are sampled more frequently, giving the network
    more exposure to rare, informative events (goal / pit encounters).
    """
    def __init__(self, maxlen=MEM_SIZE, alpha=PER_ALPHA):
        self.maxlen   = maxlen
        self.alpha    = alpha
        self.buf      = []
        self.priorities = []
        self.pos      = 0

    def push(self, s, a, r, s2, done):
        exp = (s.astype(np.float32), int(a), float(r),
               s2.astype(np.float32), float(done))
        max_p = max(self.priorities, default=1.0)
        if len(self.buf) < self.maxlen:
            self.buf.append(exp)
            self.priorities.append(max_p)
        else:
            self.buf[self.pos]        = exp
            self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.maxlen

    def sample(self, n, beta=PER_BETA):
        P   = np.array(self.priorities) ** self.alpha
        P  /= P.sum()
        idxs = np.random.choice(len(self.buf), n, replace=False, p=P)
        batch = [self.buf[i] for i in idxs]

        # Importance-sampling weights
        N = len(self.buf)
        weights = (N * P[idxs]) ** (-beta)
        weights /= weights.max()

        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r), np.array(s2),
                np.array(d), weights.astype(np.float32), idxs)

    def update_priorities(self, idxs, td_errors):
        for i, e in zip(idxs, td_errors):
            self.priorities[i] = float(abs(e)) + PER_EPS

    def __len__(self):
        return len(self.buf)


# ── S4: Dueling Network (same as Stage 2) ─────────────────────────────────────
def build_dueling_model():
    inp    = tf.keras.Input(shape=(STATE_DIM,))
    shared = tf.keras.layers.Dense(150, activation="relu")(inp)
    val    = tf.keras.layers.Dense(100, activation="relu")(shared)
    val    = tf.keras.layers.Dense(1)(val)
    adv    = tf.keras.layers.Dense(100, activation="relu")(shared)
    adv    = tf.keras.layers.Dense(ACTION_DIM)(adv)
    q      = val + adv - tf.reduce_mean(adv, axis=1, keepdims=True)
    return tf.keras.Model(inp, q)


# ── S2+S3+S5 training step with IS weights and gradient clipping ───────────────
def train_step_per(online, target, optimizer,
                   states, actions, rewards, next_states, dones, is_weights):
    """
    Combines:
      S3 Double DQN TD target
      S5 IS-weighted MSE loss
      Gradient Clipping (stabilisation)
    Returns (loss, td_errors) so PER priorities can be updated.
    """
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
        td_err = targets - q_pred               # per-sample TD error
        # S5: IS-weighted MSE
        loss   = tf.reduce_mean(is_weights * tf.square(td_err))

    grads = tape.gradient(loss, online.trainable_variables)
    # Gradient Clipping stabilisation
    grads, _ = tf.clip_by_global_norm(grads, CLIP_NORM)
    optimizer.apply_gradients(zip(grads, online.trainable_variables))
    return loss, td_err.numpy()


# ── Main training loop ─────────────────────────────────────────────────────────
def train():
    online = build_dueling_model()
    target = build_dueling_model()
    target.set_weights(online.get_weights())

    # LR Scheduling: exponential decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LR_INIT,
        decay_steps=1000,
        decay_rate=LR_DECAY,
        staircase=False
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    replay    = PrioritizedReplayBuffer()

    epsilon, losses, step = EPSILON_START, [], 0
    beta = PER_BETA

    print("=" * 60)
    print(" Stage 3 | Random Mode | S1-S5 + Clipping + LR Schedule")
    print("=" * 60)

    for epoch in range(EPOCHS):
        game   = Gridworld(size=4, mode="random")
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

            # Anneal IS beta towards 1 over training
            beta = min(1.0, PER_BETA + (1.0 - PER_BETA) * step / (EPOCHS * MAX_MOVES))

            if len(replay) >= BATCH_SIZE:
                s, a, r, s2, d, w, idxs = replay.sample(BATCH_SIZE, beta)
                loss, td_err = train_step_per(
                    online, target, optimizer,
                    tf.constant(s, dtype=tf.float32),
                    tf.cast(a, tf.int32),
                    tf.constant(r, dtype=tf.float32),
                    tf.constant(s2, dtype=tf.float32),
                    tf.constant(d, dtype=tf.float32),
                    tf.constant(w, dtype=tf.float32)
                )
                replay.update_priorities(idxs, td_err)
                losses.append(float(loss))

            if step % SYNC_EVERY == 0:
                target.set_weights(online.get_weights())

            if done or mov > MAX_MOVES:
                status = 0

        if epsilon > EPSILON_MIN:
            epsilon -= (EPSILON_START - EPSILON_MIN) / EPOCHS

        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:4d}/{EPOCHS} | e={epsilon:.3f} | "
                  f"beta={beta:.3f} | loss={losses[-1] if losses else 0:.4f}")

    print("\n[Done] Stage 3 complete.\n")
    return online, losses


# ── Visualisation ──────────────────────────────────────────────────────────────
def plot_loss(losses, outfile="loss_s3_random.png"):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#16213e")
    ax.plot(losses, alpha=0.25, color="#ff9955")
    w = 100
    if len(losses) >= w:
        sm = np.convolve(losses, np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, len(losses)), sm, color="#ff9955",
                linewidth=2, label="Smoothed Loss")
    ax.set_title("Stage 3 | S1-S5 + Grad Clip + LR Decay | Random Mode", color="white")
    ax.set_xlabel("Training Steps", color="#aaaacc")
    ax.set_ylabel("IS-Weighted MSE Loss", color="#aaaacc")
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
        if r == 10:
            render_grid(game.display(), out_end, f"{mode.capitalize()} — Win!")
            return True
        elif r == -10:
            render_grid(game.display(), out_end, f"{mode.capitalize()} — Lost")
            return False
        moves += 1
    render_grid(game.display(), out_end, f"{mode.capitalize()} — Timeout")
    return False


if __name__ == "__main__":
    model, losses = train()
    plot_loss(losses)
    snapshot(model, "random", "grid_s3_start.png", "grid_s3_end.png")
    print("All 3 stages complete!")
