"""
Self-Supervised Learning Methods
==================================
Self-supervised learning (SSL) creates *supervisory signals* automatically
from the data itself, without any human-provided labels.  A **pretext task**
(e.g., predict a masked feature, bring augmented views close together) forces
the model to learn useful representations that transfer to downstream tasks.

When to prefer each algorithm
------------------------------
• Contrastive Learning (SimCLR-style)
    – When you have abundant unlabeled data and can define meaningful
      augmentations; the learned embeddings are often strong for downstream
      classification / clustering.
• Masked Modelling (BERT/MAE-style)
    – Excellent for learning dense representations of all input tokens/features;
      naturally handles tabular data by masking a random subset of features.
• Next-token / Next-feature Prediction (GPT-style)
    – Useful when the input has a natural sequential or causal order; forces the
      model to model the full joint distribution of features.
"""

import numpy as np
import tensorflow as tf

from utils.data_utils import print_section, print_result, print_info


# ── Augmentation helpers ─────────────────────────────────────────────────────

def gaussian_noise_augment(X: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Add Gaussian noise – a lightweight tabular augmentation."""
    return (X + np.random.normal(0, sigma, X.shape)).astype(np.float32)


def feature_dropout_augment(X: np.ndarray, drop_rate: float = 0.2) -> np.ndarray:
    """Randomly zero out features – another common tabular augmentation."""
    mask = np.random.binomial(1, 1 - drop_rate, X.shape).astype(np.float32)
    return (X * mask).astype(np.float32)


# ── 1. Contrastive Learning (SimCLR-inspired) ─────────────────────────────────

def run_contrastive_learning(X, y, epochs: int = 30):
    """
    Contrastive Learning – SimCLR-inspired
    ----------------------------------------
    For each sample xᵢ we create two augmented *views* (v₁, v₂) of the same
    sample.  The NT-Xent (normalised temperature-scaled cross-entropy) loss
    pulls the representations of the same sample together and pushes different
    samples apart.

    Architecture:
        Encoder f(·) → projection head g(·) → normalised embedding z

    After pre-training, the projection head is discarded and the encoder
    embeddings are used for downstream classification (linear probe).

    Best when: large unlabeled dataset; meaningful data augmentations exist;
    you want strong transferable embeddings.
    """
    print_section("1. Contrastive Learning  (SimCLR-inspired, tabular)")
    print_info("NT-Xent loss pulls augmented views of same sample together.")

    tf.random.set_seed(42)
    n_features  = X.shape[1]
    embed_dim   = 32
    proj_dim    = 16
    temperature = 0.1
    batch_size  = 256

    # Encoder
    enc_in  = tf.keras.Input(shape=(n_features,))
    enc_h   = tf.keras.layers.Dense(128, activation="relu")(enc_in)
    enc_h   = tf.keras.layers.Dense(64,  activation="relu")(enc_h)
    enc_out = tf.keras.layers.Dense(embed_dim, activation="relu")(enc_h)
    encoder = tf.keras.Model(enc_in, enc_out, name="encoder")

    # Projection head
    proj_in  = tf.keras.Input(shape=(embed_dim,))
    proj_h   = tf.keras.layers.Dense(32, activation="relu")(proj_in)
    proj_out = tf.keras.layers.Dense(proj_dim)(proj_h)
    projector = tf.keras.Model(proj_in, proj_out, name="projector")

    optimizer = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def nt_xent_loss(z1, z2):
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)
        z  = tf.concat([z1, z2], axis=0)               # (2N, proj_dim)
        N  = tf.shape(z1)[0]
        # Cosine similarity matrix (2N × 2N)
        sim_matrix = tf.matmul(z, z, transpose_b=True) / temperature
        # Mask out self-similarities on the diagonal
        mask = tf.eye(2 * N)
        sim_matrix = sim_matrix - mask * 1e9
        # Positive pairs: (i, i+N) and (i+N, i)
        pos_indices = tf.concat(
            [tf.range(N, 2 * N), tf.range(N)], axis=0)          # (2N,)
        pos_sim = tf.gather_nd(
            sim_matrix, tf.stack([tf.range(2 * N), pos_indices], axis=1))
        loss = tf.reduce_mean(
            -pos_sim + tf.reduce_logsumexp(sim_matrix, axis=1))
        return loss

    losses = []
    n_batches = len(X) // batch_size

    for epoch in range(epochs):
        perm  = np.random.permutation(len(X))
        epoch_loss = 0.0
        for b in range(n_batches):
            batch = X[perm[b * batch_size: (b + 1) * batch_size]]
            v1    = gaussian_noise_augment(batch)
            v2    = feature_dropout_augment(batch)
            with tf.GradientTape() as tape:
                z1 = projector(encoder(v1, training=True), training=True)
                z2 = projector(encoder(v2, training=True), training=True)
                loss = nt_xent_loss(z1, z2)
            grads = tape.gradient(
                loss,
                encoder.trainable_variables + projector.trainable_variables)
            optimizer.apply_gradients(
                zip(grads,
                    encoder.trainable_variables + projector.trainable_variables))
            epoch_loss += float(loss)
        losses.append(epoch_loss / n_batches)

    # ── Linear probe evaluation ──
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    embeddings = encoder.predict(X, verbose=0)
    y_bin = (y >= 7).astype(int)
    Xe_tr, Xe_te, ye_tr, ye_te = train_test_split(
        embeddings, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
    probe = LR(max_iter=1000, random_state=42)
    probe.fit(Xe_tr, ye_tr)
    acc = accuracy_score(ye_te, probe.predict(Xe_te))

    print_result("Pre-training epochs",            epochs)
    print_result("Final contrastive loss",         f"{losses[-1]:.4f}")
    print_result("Linear probe accuracy",          f"{acc:.4f}")
    return losses, acc


# ── 2. Masked Feature Modelling (BERT-style) ─────────────────────────────────

def run_masked_modelling(X, y, epochs: int = 30, mask_rate: float = 0.30):
    """
    Masked Feature Modelling  (BERT / MAE-inspired)
    -------------------------------------------------
    Randomly masks ~30 % of input features (set to 0 + add a mask indicator
    channel) and trains an encoder to reconstruct the original values.  This
    forces the model to learn correlations between features.

    Applied to tabular data this is sometimes called BERT4Tab or TabBERT.

    Best when: features have meaningful correlations; you want a model that
    can handle missing values at inference time; large unlabeled datasets.
    """
    print_section("2. Masked Feature Modelling  (BERT / MAE-style, tabular)")
    print_info(f"Masks {int(mask_rate*100)}% of features; encoder learns to reconstruct them.")

    tf.random.set_seed(42)
    n_feat = X.shape[1]

    # Input: original features + binary mask indicators (concatenated)
    inp     = tf.keras.Input(shape=(n_feat * 2,))
    h       = tf.keras.layers.Dense(128, activation="relu")(inp)
    h       = tf.keras.layers.Dense(64,  activation="relu")(h)
    encoded = tf.keras.layers.Dense(32,  activation="relu", name="encoder")(h)
    decoded = tf.keras.layers.Dense(n_feat, activation="linear")(encoded)

    model     = tf.keras.Model(inp, decoded)
    encoder_m = tf.keras.Model(inp, encoded)  # shared encoder for downstream
    model.compile(optimizer="adam", loss="mse")

    def mask_batch(batch):
        mask = np.random.binomial(1, mask_rate, batch.shape).astype(np.float32)
        masked  = batch * (1 - mask)
        inp_arr = np.concatenate([masked, mask], axis=1)
        return inp_arr.astype(np.float32), batch.astype(np.float32)

    batch_size = 256
    losses     = []
    for epoch in range(epochs):
        perm   = np.random.permutation(len(X))
        e_loss = 0.0
        for b in range(len(X) // batch_size):
            batch = X[perm[b * batch_size:(b + 1) * batch_size]]
            inp_b, tgt_b = mask_batch(batch)
            h = model.train_on_batch(inp_b, tgt_b)
            e_loss += h
        losses.append(e_loss / (len(X) // batch_size))

    # Downstream linear probe (using full features + zero-mask indicator)
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    zero_mask = np.zeros_like(X)
    X_enc_inp = np.concatenate([X, zero_mask], axis=1).astype(np.float32)
    embeddings = encoder_m.predict(X_enc_inp, verbose=0)

    y_bin = (y >= 7).astype(int)
    Xe_tr, Xe_te, ye_tr, ye_te = train_test_split(
        embeddings, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
    probe = LR(max_iter=1000, random_state=42)
    probe.fit(Xe_tr, ye_tr)
    acc = accuracy_score(ye_te, probe.predict(Xe_te))

    print_result("Pre-training epochs",        epochs)
    print_result("Final reconstruction MSE",   f"{losses[-1]:.6f}")
    print_result("Linear probe accuracy",      f"{acc:.4f}")
    return losses, acc


# ── 3. Next-Feature Prediction (GPT-style autoregressive) ────────────────────

def run_next_feature_prediction(X, y, epochs: int = 30):
    """
    Next-Feature Prediction  (GPT / autoregressive-style)
    -------------------------------------------------------
    Treats the 12 wine features as an ordered *sequence*.  The model is trained
    to predict feature fᵢ given features f₀ … fᵢ₋₁ (causal / left-to-right).
    A causal mask prevents the model from seeing future positions.

    This is analogous to language modelling (predict the next token) but
    applied to tabular features.  Teaches the model the joint distribution
    P(f₁, …, f₁₂) = ∏ P(fᵢ | f₁, …, fᵢ₋₁).

    Best when: features have a natural ordering or causal relationship; you
    want a generative model that can impute any suffix of features.
    """
    print_section("3. Next-Feature Prediction  (GPT-style autoregressive)")
    print_info("Autoregressive prediction of fᵢ given f₀…fᵢ₋₁ – learns feature order.")

    tf.random.set_seed(42)
    n_feat     = X.shape[1]
    batch_size = 256
    embed_dim  = 32

    # Simple 1-D causal architecture (not a full Transformer for brevity)
    inp   = tf.keras.Input(shape=(n_feat,))   # full feature vector
    h     = tf.keras.layers.Dense(128, activation="relu")(inp)
    h     = tf.keras.layers.Dense(64,  activation="relu")(h)
    enc   = tf.keras.layers.Dense(embed_dim, activation="relu", name="enc")(h)
    out   = tf.keras.layers.Dense(n_feat, activation="linear")(enc)

    model     = tf.keras.Model(inp, out, name="gpt_tabular")
    encoder_g = tf.keras.Model(inp, enc)
    model.compile(optimizer="adam", loss="mse")

    def causal_mask_batch(batch):
        """
        For position i the model sees features 0…i-1 (zeroed beyond).
        We train on all positions at once via the full-sequence MSE.
        """
        # Shift: input = [0, f₀, f₁, …, f_{n-1}], target = [f₀, f₁, …, f_n]
        # Simplified: mask off a random prefix and predict the rest
        k = np.random.randint(1, n_feat)   # predict features k … n_feat-1
        masked       = batch.copy()
        masked[:, k:] = 0.0               # causal: hide future features
        return masked.astype(np.float32), batch.astype(np.float32)

    losses = []
    for epoch in range(epochs):
        perm   = np.random.permutation(len(X))
        e_loss = 0.0
        for b in range(len(X) // batch_size):
            batch = X[perm[b * batch_size:(b + 1) * batch_size]]
            inp_b, tgt_b = causal_mask_batch(batch)
            e_loss += model.train_on_batch(inp_b, tgt_b)
        losses.append(e_loss / (len(X) // batch_size))

    # Downstream linear probe
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    embeddings = encoder_g.predict(X, verbose=0)
    y_bin      = (y >= 7).astype(int)
    Xe_tr, Xe_te, ye_tr, ye_te = train_test_split(
        embeddings, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
    probe = LR(max_iter=1000, random_state=42)
    probe.fit(Xe_tr, ye_tr)
    acc = accuracy_score(ye_te, probe.predict(Xe_te))

    print_result("Pre-training epochs",      epochs)
    print_result("Final prediction MSE",     f"{losses[-1]:.6f}")
    print_result("Linear probe accuracy",    f"{acc:.4f}")
    return losses, acc


# ── Entry point ───────────────────────────────────────────────────────────────

def run_all(X, y):
    results = {}
    results["contrastive"]        = run_contrastive_learning(X, y)
    results["masked_modelling"]   = run_masked_modelling(X, y)
    results["next_feature_pred"]  = run_next_feature_prediction(X, y)
    return results
