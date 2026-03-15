"""
Transfer Learning Methods
==========================
Transfer learning reuses knowledge gained while solving one problem and applies
it to a different but related problem.  This is especially valuable when the
target domain has limited data.

Wine Dataset Framing
---------------------
We treat **red wine** samples as the *source domain* and **white wine** samples
as the *target domain*.  A model pre-trained on the abundant red-wine data is
then adapted (fine-tuned, feature-extracted, or domain-adapted) to predict
white-wine quality with fewer samples.

When to prefer each algorithm
------------------------------
• Fine-tuning        – all pre-trained weights are updated on the target domain.
                       Best when target data is sufficient and domains are
                       moderately different.
• Feature Extraction – freeze pre-trained layers; train only a new head.
                       Best when target data is scarce; avoids overfitting the
                       backbone.
• Domain Adaptation  – explicitly minimises the distribution shift between
                       source and target feature distributions (e.g., via MMD
                       or adversarial alignment).  Best when labelled target
                       data is very scarce or absent.
"""

import numpy as np
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics        import accuracy_score

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from utils.data_utils import print_section, print_result, print_info


# ── Shared backbone factory ───────────────────────────────────────────────────

def build_backbone(input_dim: int, embed_dim: int = 32):
    inp = tf.keras.Input(shape=(input_dim,))
    h   = tf.keras.layers.Dense(128, activation="relu")(inp)
    h   = tf.keras.layers.Dense(64,  activation="relu")(h)
    emb = tf.keras.layers.Dense(embed_dim, activation="relu", name="embedding")(h)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="head")(emb)
    return tf.keras.Model(inp, out, name="backbone")


# ── Source / target data split ────────────────────────────────────────────────

def split_by_color(df, X, y_binary):
    """Return (X_red, y_red), (X_white, y_white) using colour indicator."""
    red_mask   = df["color_enc"].values == 0
    white_mask = df["color_enc"].values == 1
    return ((X[red_mask],   y_binary[red_mask]),
            (X[white_mask], y_binary[white_mask]))


# ── 1. Fine-tuning ────────────────────────────────────────────────────────────

def run_fine_tuning(df, X, y_binary, source_epochs: int = 30,
                    target_epochs: int = 20):
    """
    Fine-tuning a Pre-trained Model
    ---------------------------------
    1. Pre-train the backbone on the *source* domain (red wine).
    2. Continue training all layers on the *target* domain (white wine),
       typically with a smaller learning rate to preserve learned features.

    Best when: source and target domains share substantial structure; the
    target dataset is large enough to update all parameters without severe
    overfitting.
    """
    print_section("1. Fine-tuning  (pre-trained on red → fine-tuned on white)")
    print_info("All backbone layers are updated on the target domain.")

    if not TF_AVAILABLE:
        print_info("SKIPPED – tensorflow not installed  (pip install tensorflow)")
        return float("nan")

    tf.random.set_seed(42)
    (X_red, y_red), (X_white, y_white) = split_by_color(df, X, y_binary)

    # Train / test split for white wine (target)
    n_test  = int(len(X_white) * 0.2)
    X_wtr, X_wte = X_white[:-n_test], X_white[-n_test:]
    y_wtr, y_wte = y_white[:-n_test], y_white[-n_test:]

    # Phase 1 – pre-train on source (red wine)
    model = build_backbone(X.shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_red, y_red, epochs=source_epochs, batch_size=64,
              verbose=0, validation_split=0.1)
    print_result("Source pre-training accuracy (val)",
                 f"{model.history.history['val_accuracy'][-1]:.4f}")

    # Phase 2 – fine-tune on target (white wine) with a smaller lr
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_wtr, y_wtr, epochs=target_epochs, batch_size=64,
              verbose=0, validation_split=0.1)

    y_pred = (model.predict(X_wte, verbose=0) > 0.5).astype(int).flatten()
    acc    = accuracy_score(y_wte, y_pred)
    print_result("Fine-tuned accuracy on white wine", f"{acc:.4f}")
    return acc


# ── 2. Feature Extraction ────────────────────────────────────────────────────

def run_feature_extraction(df, X, y_binary, source_epochs: int = 30):
    """
    Feature Extraction (Frozen Backbone)
    ---------------------------------------
    1. Pre-train the backbone on the source domain (red wine).
    2. Freeze all backbone layers.
    3. Extract embeddings for the target domain (white wine).
    4. Train a lightweight linear classifier on those embeddings only.

    Best when: target dataset is small; you cannot risk overfitting the backbone;
    source and target domains share low-level features.
    """
    print_section("2. Feature Extraction  (frozen backbone + new linear head)")
    print_info("Backbone frozen; only a logistic regression head is trained on white wine.")

    if not TF_AVAILABLE:
        print_info("SKIPPED – tensorflow not installed  (pip install tensorflow)")
        return float("nan")

    tf.random.set_seed(42)
    (X_red, y_red), (X_white, y_white) = split_by_color(df, X, y_binary)

    n_test  = int(len(X_white) * 0.2)
    X_wtr, X_wte = X_white[:-n_test], X_white[-n_test:]
    y_wtr, y_wte = y_white[:-n_test], y_white[-n_test:]

    # Pre-train on red wine
    model = build_backbone(X.shape[1])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(X_red, y_red, epochs=source_epochs, batch_size=64,
              verbose=0, validation_split=0.1)

    # Freeze all layers
    for layer in model.layers:
        layer.trainable = False

    # Extract embeddings from the "embedding" layer
    extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("embedding").output,
    )
    emb_train = extractor.predict(X_wtr, verbose=0)
    emb_test  = extractor.predict(X_wte, verbose=0)

    # Linear classifier on frozen embeddings
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(emb_train, y_wtr)
    acc = accuracy_score(y_wte, clf.predict(emb_test))
    print_result("Feature-extraction accuracy on white wine", f"{acc:.4f}")
    return acc


# ── 3. Domain Adaptation (MMD-based) ─────────────────────────────────────────

def run_domain_adaptation(df, X, y_binary, epochs: int = 40):
    """
    Domain Adaptation  (Maximum Mean Discrepancy – MMD)
    ------------------------------------------------
    Trains an encoder to simultaneously (a) classify source-domain samples
    correctly and (b) minimise the **Maximum Mean Discrepancy** (MMD) between
    the source and target feature distributions in the embedding space.

    MMD is a kernel-based distance between distributions:
        MMD²(P, Q) = E[k(x,x')] − 2E[k(x,y)] + E[k(y,y')]
    Minimising it encourages the encoder to produce domain-invariant features.

    Best when: target labels are scarce or absent; a soft domain-alignment
    regulariser is preferred over adversarial training; you need a principled
    measure of distribution shift.
    """
    print_section("3. Domain Adaptation  (MMD regularisation, red→white)")
    print_info("Encoder minimises source classification loss + MMD(source, target).")

    if not TF_AVAILABLE:
        print_info("SKIPPED – tensorflow not installed  (pip install tensorflow)")
        return float("nan")

    tf.random.set_seed(42)
    (X_red, y_red), (X_white, y_white) = split_by_color(df, X, y_binary)

    n_test  = int(len(X_white) * 0.2)
    X_wtr, X_wte = X_white[:-n_test], X_white[-n_test:]
    y_wtr, y_wte = y_white[:-n_test], y_white[-n_test:]

    input_dim = X.shape[1]
    embed_dim = 32

    inp  = tf.keras.Input(shape=(input_dim,))
    h    = tf.keras.layers.Dense(128, activation="relu")(inp)
    h    = tf.keras.layers.Dense(64,  activation="relu")(h)
    emb  = tf.keras.layers.Dense(embed_dim, activation="relu", name="emb")(h)
    out  = tf.keras.layers.Dense(1, activation="sigmoid")(emb)

    model    = tf.keras.Model(inp, [emb, out])
    encoder  = tf.keras.Model(inp, emb)
    optimizer = tf.keras.optimizers.Adam(1e-3)
    bce       = tf.keras.losses.BinaryCrossentropy()

    def rbf_kernel(X1, X2, sigma=1.0):
        sq_dist = (tf.reduce_sum(X1 ** 2, axis=1, keepdims=True)
                   + tf.reduce_sum(X2 ** 2, axis=1)
                   - 2 * tf.matmul(X1, X2, transpose_b=True))
        return tf.exp(-sq_dist / (2 * sigma ** 2))

    def mmd_loss(Zs, Zt):
        """Empirical MMD² with RBF kernel."""
        k_ss = rbf_kernel(Zs, Zs)
        k_tt = rbf_kernel(Zt, Zt)
        k_st = rbf_kernel(Zs, Zt)
        return tf.reduce_mean(k_ss) - 2 * tf.reduce_mean(k_st) + tf.reduce_mean(k_tt)

    batch = 128
    lam   = 0.5   # MMD regularisation weight

    # Match batch sizes
    n_red   = len(X_red)
    n_white = len(X_wtr)

    for epoch in range(epochs):
        perm_r = np.random.permutation(n_red)
        perm_w = np.random.permutation(n_white)
        n_batches = min(n_red, n_white) // batch

        for b in range(n_batches):
            Xs = X_red[perm_r[b * batch:(b + 1) * batch]]
            ys = y_red[perm_r[b * batch:(b + 1) * batch]]
            Xt = X_wtr[perm_w[b * batch:(b + 1) * batch]]

            with tf.GradientTape() as tape:
                Zs, pred_s = model(Xs, training=True)
                Zt, _      = model(Xt, training=True)
                cls_loss   = bce(ys.reshape(-1, 1), pred_s)
                mmd        = mmd_loss(Zs, Zt)
                loss       = cls_loss + lam * mmd

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Evaluate on white wine test set using extracted features + LR head
    emb_tr = encoder.predict(X_wtr, verbose=0)
    emb_te = encoder.predict(X_wte, verbose=0)
    clf    = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(emb_tr, y_wtr)
    acc = accuracy_score(y_wte, clf.predict(emb_te))
    print_result("Domain-adapted accuracy on white wine", f"{acc:.4f}")
    return acc


# ── Entry point ───────────────────────────────────────────────────────────────

def run_all(df, X, y_binary):
    results = {}
    results["fine_tuning"]         = run_fine_tuning(df, X, y_binary)
    results["feature_extraction"]  = run_feature_extraction(df, X, y_binary)
    results["domain_adaptation"]   = run_domain_adaptation(df, X, y_binary)
    return results
