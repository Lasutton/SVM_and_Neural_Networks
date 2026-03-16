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

from utils.data_utils import (
    print_section, print_result, print_info, print_explain, print_score_bar,
)


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
    print_section("1. Fine-tuning  (learn from red wine, then adapt to white wine)")
    print_info("What is Transfer Learning?")
    print_explain(
        "Transfer Learning is like a chef who was trained in French cuisine "
        "(red wine pairing) and then applies that cooking knowledge to learn "
        "Italian cuisine (white wine pairing) much faster than starting from scratch.  "
        "The model first trains on RED wine data (1,599 samples).  "
        "Then it uses that knowledge as a starting point to learn WHITE wine data "
        "(4,898 samples).  Because red and white wine share chemistry fundamentals, "
        "the head start makes the white-wine model better and faster!")
    print_info("What is Fine-tuning?")
    print_explain(
        "Fine-tuning means we take the pre-trained model (trained on red wine) "
        "and continue training ALL its layers on white wine data — but with a "
        "much smaller learning rate (10× smaller) so we don't overwrite the "
        "valuable red-wine knowledge too aggressively.  "
        "Like a chef adjusting their recipe book rather than rewriting it entirely.")
    print_info(f"Phase 1: train on red wine for {source_epochs} epochs.  "
               f"Phase 2: fine-tune on white wine for {target_epochs} epochs.")

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
    source_val_acc = model.history.history['val_accuracy'][-1]
    print_score_bar("Red wine pre-training val accuracy", source_val_acc)
    print_result("Red wine source pre-training accuracy (validation)",
                 f"{source_val_acc:.4f}  ({source_val_acc*100:.1f}%)")
    print_explain(
        f"The model correctly identified {source_val_acc*100:.1f}% of red wines as good/not-good.  "
        "This knowledge is now 'baked into' the network weights.")

    # Phase 2 – fine-tune on target (white wine) with a smaller lr
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_wtr, y_wtr, epochs=target_epochs, batch_size=64,
              verbose=0, validation_split=0.1)

    y_pred = (model.predict(X_wte, verbose=0) > 0.5).astype(int).flatten()
    acc    = accuracy_score(y_wte, y_pred)
    print_score_bar("White wine fine-tuned accuracy", acc)
    print_result("Fine-tuned accuracy on white wine (test set)", f"{acc:.4f}  ({acc*100:.1f}%)")
    print_explain(
        f"After fine-tuning on white wine, accuracy = {acc*100:.1f}%.  "
        "Compare this to Training From Scratch on white wine only — transfer learning "
        "typically gives a better starting point and sometimes better final accuracy!")
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
    print_section("2. Feature Extraction  (use red-wine brain as a frozen detector)")
    print_info("What is Feature Extraction?")
    print_explain(
        "Feature Extraction is the most conservative form of transfer learning.  "
        "We train on red wine, then FREEZE (lock) ALL the neural network weights.  "
        "The frozen network becomes a feature detector: feed in a white wine, "
        "get out a 32-number 'fingerprint' that captures the wine's key patterns.  "
        "Then we ONLY train a tiny logistic regression classifier on those fingerprints.  "
        "Think of it as: the red-wine chef's entire knowledge is frozen in a book, "
        "and a new apprentice reads the book and uses it to evaluate white wines — "
        "the book never gets rewritten.  "
        "Great when you have very few white wine examples!")
    print_info("Backbone frozen after red-wine pre-training — only the new head trains.")

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
    print_score_bar("White wine feature-extraction accuracy", acc)
    print_result("Feature-extraction accuracy on white wine", f"{acc:.4f}  ({acc*100:.1f}%)")
    print_explain(
        f"Using the frozen red-wine encoder as a feature extractor + a simple "
        f"linear classifier on white wine achieves {acc*100:.1f}%.  "
        "If this is close to fine-tuning accuracy, it means the red-wine features "
        "transfer well and you don't need to update the backbone at all!")
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
    print_section("3. Domain Adaptation  (aligning red and white wine in the same 'space')")
    print_info("What is Domain Adaptation?")
    print_explain(
        "Domain Adaptation tackles a tricky problem: red wine and white wine "
        "have different chemical profiles (different 'distributions').  "
        "A model trained on red wine might perform poorly on white wine simply "
        "because the two look different even when they have similar quality patterns.  "
        "Domain Adaptation trains the model to produce features that look the "
        "SAME for both red and white wine — so the quality-predicting parts of "
        "the model don't get confused by the red-vs-white difference.  "
        "MMD (Maximum Mean Discrepancy) measures how different the two distributions "
        "are and penalises the model for keeping them apart.  "
        "Think of it as teaching the model to ignore whether wine is red or white "
        "and focus only on quality-relevant signals!")
    print_info("Minimising: classification loss + 0.5 × MMD(red embeddings, white embeddings).")

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
    print_score_bar("Domain-adapted accuracy on white wine", acc)
    print_result("Domain-adapted accuracy on white wine", f"{acc:.4f}  ({acc*100:.1f}%)")
    print_explain(
        f"After alignment training, accuracy = {acc*100:.1f}% on white wine.  "
        "Domain adaptation is especially useful when you have very few white wine "
        "labels — the alignment allows the model to use red wine knowledge more "
        "effectively for the white wine task.")
    return acc


# ── Entry point ───────────────────────────────────────────────────────────────

def run_all(df, X, y_binary):
    results = {}
    results["fine_tuning"]         = run_fine_tuning(df, X, y_binary)
    results["feature_extraction"]  = run_feature_extraction(df, X, y_binary)
    results["domain_adaptation"]   = run_domain_adaptation(df, X, y_binary)
    return results
