"""
Reinforcement Learning Methods
================================
Reinforcement Learning (RL) trains an *agent* to take actions in an
*environment* to maximise cumulative *reward*.  No labeled dataset is given;
the agent discovers optimal behaviour through trial and error.

Wine Quality Environment
-------------------------
We frame wine quality estimation as an RL task: the agent is shown a randomly
selected wine's features and must guess whether it is "good" (quality ≥ 7).
- State  : scaled feature vector of the wine
- Actions: 0 = "not good", 1 = "good"
- Reward : +1 for correct classification, -1 for wrong

This is a simplified educational environment; in production RL is used for
sequential decision problems (robotics, game playing, recommendation systems).

When to prefer each algorithm
------------------------------
• Q-Learning       – tabular; exact; only practical for small discrete spaces.
• DQN              – approximates Q-values with a neural network; handles
                     continuous/large state spaces.
• Policy Gradient  – directly optimises the policy; works in continuous action
                     spaces; high variance but unbiased.
• Actor-Critic     – combines a policy (actor) with a value baseline (critic);
                     lower variance than pure policy gradient.
• PPO              – clips the surrogate objective to prevent destructive updates;
                     state-of-the-art for many continuous control tasks.
"""

import numpy as np
from collections import deque
import random

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from utils.data_utils import print_section, print_result, print_info


# ── Wine Classification Environment ──────────────────────────────────────────

class WineEnv:
    """Minimal Gym-like environment backed by the wine dataset."""

    def __init__(self, X, y_binary):
        self.X        = X
        self.y        = y_binary
        self.n        = len(X)
        self.state_dim = X.shape[1]
        self.n_actions = 2
        self._idx     = 0

    def reset(self):
        self._idx = np.random.randint(self.n)
        return self.X[self._idx].copy()

    def step(self, action: int):
        correct = int(action == self.y[self._idx])
        reward  = 1.0 if correct else -1.0
        next_state = self.reset()
        done = True          # episodic: one wine = one episode
        return next_state, reward, done, {}

    @property
    def accuracy_of_random(self):
        return max(self.y.mean(), 1 - self.y.mean())


# ── 1. Q-Learning (tabular, discretised) ────────────────────────────────────

def run_q_learning(X, y_binary, episodes: int = 5000):
    """
    Q-Learning (Tabular)
    ---------------------
    Maintains a table Q(s, a) – the expected cumulative reward for taking
    action a in state s.  Updates via the Bellman equation:
        Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]

    Since our state space is continuous we project each state to one of k
    discrete bins (simple quantile discretisation).

    Best when: the state and action spaces are small and discrete; you need a
    theoretically guaranteed convergent method; interpretability of the Q-table
    is useful.
    """
    print_section("1. Q-Learning  (tabular, state discretised to 10 bins)")
    print_info("Bellman updates on a finite Q-table – simplest RL algorithm.")

    env  = WineEnv(X, y_binary)
    bins = 10

    # Discretise each feature into equal-width bins
    edges = [np.linspace(X[:, i].min(), X[:, i].max(), bins - 1)
             for i in range(X.shape[1])]

    def discretise(state):
        return tuple(np.searchsorted(edges[i], state[i]) for i in range(len(state)))

    Q = {}  # defaultdict-style sparse Q-table

    def q(s, a):
        return Q.get((s, a), 0.0)

    alpha, gamma, epsilon = 0.1, 0.9, 1.0
    epsilon_decay = 0.999
    min_eps       = 0.05
    rewards_hist  = []

    for ep in range(episodes):
        state_cont = env.reset()
        state      = discretise(state_cont)
        _, reward, _, _ = env.step(
            0 if q(state, 0) >= q(state, 1) else 1   # greedy w.r.t Q
            if random.random() > epsilon
            else random.randint(0, 1)
        )
        # Bellman update (single-step episode)
        action = (0 if random.random() > epsilon else random.randint(0, 1))
        _, reward, _, _ = env.step(action)
        next_state = discretise(env.reset())
        best_next = max(q(next_state, 0), q(next_state, 1))
        Q[(state, action)] = q(state, action) + alpha * (
            reward + gamma * best_next - q(state, action))
        epsilon = max(min_eps, epsilon * epsilon_decay)
        rewards_hist.append(reward)

    avg_reward = np.mean(rewards_hist[-1000:])
    print_result("Episodes trained", episodes)
    print_result("Avg reward (last 1 000 eps)", f"{avg_reward:.3f}")
    print_result("Q-table entries", len(Q))
    print_info(f"Random-policy baseline reward: {2*env.accuracy_of_random - 1:.3f}")
    return avg_reward


# ── 2. Deep Q-Network (DQN) ──────────────────────────────────────────────────

def run_dqn(X, y_binary, episodes: int = 1000):
    """
    Deep Q-Network (DQN)
    ----------------------
    Replaces the Q-table with a neural network Qθ(s, a).  Two key innovations
    (Mnih et al., 2015):
    1. **Experience Replay**: stores transitions in a replay buffer and samples
       random mini-batches, breaking temporal correlations.
    2. **Target Network**: a periodically frozen copy of Qθ provides stable
       training targets.

    Best when: the state space is high-dimensional or continuous; tabular Q-
    learning is infeasible; you can afford GPU compute.
    """
    print_section("2. Deep Q-Network (DQN)")
    print_info("Neural Q-function + experience replay + target network.")

    if not TF_AVAILABLE:
        print_info("SKIPPED – tensorflow not installed  (pip install tensorflow)")
        return float("nan")

    tf.random.set_seed(42)
    env = WineEnv(X, y_binary)

    def build_q_net():
        return tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu",
                                  input_shape=(env.state_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(env.n_actions, activation="linear"),
        ])

    q_net    = build_q_net()
    tgt_net  = build_q_net()
    tgt_net.set_weights(q_net.get_weights())
    optimizer = tf.keras.optimizers.Adam(1e-3)
    loss_fn   = tf.keras.losses.Huber()

    replay      = deque(maxlen=5000)
    batch_size  = 64
    gamma       = 0.9
    epsilon     = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    rewards_hist  = []

    for ep in range(episodes):
        state = env.reset()
        if random.random() < epsilon:
            action = random.randint(0, env.n_actions - 1)
        else:
            qs     = q_net(state[np.newaxis], training=False).numpy()[0]
            action = int(np.argmax(qs))

        next_state, reward, done, _ = env.step(action)
        replay.append((state, action, reward, next_state, done))
        rewards_hist.append(reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if len(replay) >= batch_size:
            batch = random.sample(replay, batch_size)
            s  = np.array([t[0] for t in batch], dtype=np.float32)
            a  = np.array([t[1] for t in batch], dtype=np.int32)
            r  = np.array([t[2] for t in batch], dtype=np.float32)
            s2 = np.array([t[3] for t in batch], dtype=np.float32)
            d  = np.array([t[4] for t in batch], dtype=np.float32)

            q_next = tgt_net(s2, training=False).numpy()
            targets = r + gamma * q_next.max(axis=1) * (1 - d)

            with tf.GradientTape() as tape:
                q_vals = q_net(s, training=True)
                idx    = tf.stack([tf.range(batch_size), a], axis=1)
                pred   = tf.gather_nd(q_vals, idx)
                loss   = loss_fn(targets, pred)
            grads = tape.gradient(loss, q_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_net.trainable_variables))

        if ep % 100 == 0:
            tgt_net.set_weights(q_net.get_weights())

    avg_reward = np.mean(rewards_hist[-200:])
    print_result("Episodes trained", episodes)
    print_result("Avg reward (last 200 eps)", f"{avg_reward:.3f}")
    return avg_reward


# ── 3. Policy Gradient (REINFORCE) ───────────────────────────────────────────

def run_policy_gradient(X, y_binary, episodes: int = 2000):
    """
    Policy Gradient – REINFORCE (Williams, 1992)
    ---------------------------------------------
    Directly parameterises a stochastic policy πθ(a|s) and maximises the
    expected return by gradient ascent:
        ∇θ J ≈ Σ_t ∇θ log πθ(aₜ|sₜ) · Gₜ
    where Gₜ is the discounted future return from step t.

    Best when: the action space is continuous; you want to directly optimise
    the policy; stochasticity in the policy is desirable (exploration).
    High variance estimator; mitigated by baselines / actor-critic.
    """
    print_section("3. Policy Gradient  (REINFORCE)")
    print_info("Directly ascends the policy gradient – unbiased but high-variance.")

    if not TF_AVAILABLE:
        print_info("SKIPPED – tensorflow not installed  (pip install tensorflow)")
        return float("nan")

    tf.random.set_seed(42)
    env = WineEnv(X, y_binary)

    policy = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu",
                              input_shape=(env.state_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(env.n_actions, activation="softmax"),
    ])
    optimizer = tf.keras.optimizers.Adam(3e-4)

    rewards_hist = []
    for ep in range(episodes):
        state = env.reset()
        with tf.GradientTape() as tape:
            probs  = policy(state[np.newaxis], training=True)
            dist   = tf.squeeze(probs)
            action = int(tf.random.categorical(tf.math.log(probs), 1).numpy()[0, 0])
            _, reward, _, _ = env.step(action)
            log_prob = tf.math.log(dist[action] + 1e-8)
            loss     = -log_prob * reward           # single-step return
        grads = tape.gradient(loss, policy.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy.trainable_variables))
        rewards_hist.append(reward)

    avg_reward = np.mean(rewards_hist[-500:])
    print_result("Episodes trained", episodes)
    print_result("Avg reward (last 500 eps)", f"{avg_reward:.3f}")
    return avg_reward


# ── 4. Actor-Critic (A2C) ────────────────────────────────────────────────────

def run_actor_critic(X, y_binary, episodes: int = 2000):
    """
    Actor-Critic (Advantage Actor-Critic – A2C)
    --------------------------------------------
    The **actor** outputs a policy π(a|s); the **critic** estimates the state
    value V(s).  The actor is updated using the *advantage*:
        A(s,a) = r + γ V(s') − V(s)
    subtracting the baseline V(s) reduces the variance of the policy gradient
    without introducing bias.

    Best when: you want lower variance than REINFORCE with the same unbiasedness;
    episodic or continuous tasks; easily parallelised (A3C variant).
    """
    print_section("4. Actor-Critic  (A2C, shared backbone)")
    print_info("Actor (policy) + Critic (value baseline) – lower variance than REINFORCE.")

    if not TF_AVAILABLE:
        print_info("SKIPPED – tensorflow not installed  (pip install tensorflow)")
        return float("nan")

    tf.random.set_seed(42)
    env = WineEnv(X, y_binary)

    # Shared backbone → actor head + critic head
    inp     = tf.keras.Input(shape=(env.state_dim,))
    hidden  = tf.keras.layers.Dense(64, activation="relu")(inp)
    hidden  = tf.keras.layers.Dense(32, activation="relu")(hidden)
    actor   = tf.keras.layers.Dense(env.n_actions, activation="softmax")(hidden)
    critic  = tf.keras.layers.Dense(1, activation="linear")(hidden)

    model     = tf.keras.Model(inp, [actor, critic])
    optimizer = tf.keras.optimizers.Adam(3e-4)
    gamma     = 0.99
    rewards_hist = []

    for ep in range(episodes):
        state = env.reset()
        state_t = tf.constant(state[np.newaxis], dtype=tf.float32)

        with tf.GradientTape() as tape:
            probs_t, value_t = model(state_t, training=True)
            probs  = tf.squeeze(probs_t)
            value  = tf.squeeze(value_t)
            action = int(tf.random.categorical(
                tf.math.log(tf.expand_dims(probs, 0)), 1).numpy()[0, 0])
            next_state, reward, done, _ = env.step(action)
            next_state_t  = tf.constant(next_state[np.newaxis], dtype=tf.float32)
            _, next_value = model(next_state_t, training=False)
            next_value    = tf.squeeze(next_value)
            td_target  = reward + gamma * next_value * (1.0 - float(done))
            advantage  = td_target - value
            actor_loss = -tf.math.log(probs[action] + 1e-8) * advantage
            critic_loss = advantage ** 2
            total_loss  = actor_loss + 0.5 * critic_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        rewards_hist.append(reward)

    avg_reward = np.mean(rewards_hist[-500:])
    print_result("Episodes trained", episodes)
    print_result("Avg reward (last 500 eps)", f"{avg_reward:.3f}")
    return avg_reward


# ── 5. Proximal Policy Optimisation (PPO) ────────────────────────────────────

def run_ppo(X, y_binary, episodes: int = 2000, clip_eps: float = 0.2):
    """
    Proximal Policy Optimisation (PPO, Schulman et al., 2017)
    ----------------------------------------------------------
    PPO clips the policy update ratio r_t(θ) = πθ(a|s) / π_old(a|s) to stay
    within [1-ε, 1+ε].  This prevents catastrophically large policy updates
    that destabilise training – the key weakness of vanilla policy gradient
    and TRPO.

    Objective: L^{CLIP} = E [ min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t) ]

    Best when: you need stable, sample-efficient policy learning; continuous
    control tasks (robotics, simulation); on-policy batch updates.
    """
    print_section("5. Proximal Policy Optimisation  (PPO, clip ε=0.2)")
    print_info("Clipped surrogate objective prevents large destabilising updates.")

    if not TF_AVAILABLE:
        print_info("SKIPPED – tensorflow not installed  (pip install tensorflow)")
        return float("nan")

    tf.random.set_seed(42)
    env = WineEnv(X, y_binary)

    inp    = tf.keras.Input(shape=(env.state_dim,))
    h      = tf.keras.layers.Dense(64, activation="relu")(inp)
    h      = tf.keras.layers.Dense(32, activation="relu")(h)
    actor  = tf.keras.layers.Dense(env.n_actions, activation="softmax")(h)
    critic = tf.keras.layers.Dense(1, activation="linear")(h)

    model     = tf.keras.Model(inp, [actor, critic])
    optimizer = tf.keras.optimizers.Adam(3e-4)
    n_epochs  = 4

    rewards_hist  = []
    buffer_states, buffer_actions, buffer_rewards, buffer_old_probs = [], [], [], []

    for ep in range(episodes):
        state = env.reset()
        probs_t, _ = model(state[np.newaxis], training=False)
        probs       = tf.squeeze(probs_t).numpy()
        action      = np.random.choice(env.n_actions, p=probs)
        old_prob    = probs[action]

        _, reward, _, _ = env.step(action)

        buffer_states.append(state)
        buffer_actions.append(action)
        buffer_rewards.append(reward)
        buffer_old_probs.append(old_prob)
        rewards_hist.append(reward)

        # Mini-batch update every 64 steps
        if (ep + 1) % 64 == 0 and buffer_states:
            s_arr  = np.array(buffer_states,    dtype=np.float32)
            a_arr  = np.array(buffer_actions,   dtype=np.int32)
            r_arr  = np.array(buffer_rewards,   dtype=np.float32)
            op_arr = np.array(buffer_old_probs, dtype=np.float32)

            for _ in range(n_epochs):
                with tf.GradientTape() as tape:
                    new_probs_t, values_t = model(s_arr, training=True)
                    values   = tf.squeeze(values_t)
                    adv      = tf.constant(r_arr - values.numpy())
                    idx_2d   = tf.stack(
                        [tf.range(len(a_arr)), tf.constant(a_arr)], axis=1)
                    new_prob = tf.gather_nd(new_probs_t, idx_2d)
                    ratio    = new_prob / (tf.constant(op_arr) + 1e-8)
                    clip_r   = tf.clip_by_value(ratio, 1 - clip_eps, 1 + clip_eps)
                    actor_l  = -tf.reduce_mean(tf.minimum(ratio * adv, clip_r * adv))
                    critic_l = tf.reduce_mean((r_arr - values) ** 2)
                    loss     = actor_l + 0.5 * critic_l
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            buffer_states.clear(); buffer_actions.clear()
            buffer_rewards.clear(); buffer_old_probs.clear()

    avg_reward = np.mean(rewards_hist[-500:])
    print_result("Episodes trained", episodes)
    print_result("Avg reward (last 500 eps)", f"{avg_reward:.3f}")
    return avg_reward


# ── Entry point ───────────────────────────────────────────────────────────────

def run_all(X, y_binary):
    results = {}
    results["q_learning"]      = run_q_learning(X, y_binary, episodes=5000)
    results["dqn"]             = run_dqn(X, y_binary, episodes=1000)
    results["policy_gradient"] = run_policy_gradient(X, y_binary, episodes=2000)
    results["actor_critic"]    = run_actor_critic(X, y_binary, episodes=2000)
    results["ppo"]             = run_ppo(X, y_binary, episodes=2000)
    return results
