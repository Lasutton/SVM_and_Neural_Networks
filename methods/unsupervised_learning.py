"""
Unsupervised Learning Methods
==============================
Unsupervised learning discovers hidden structure in data **without** labels.
Applied to the wine dataset we can find natural groupings, compress the feature
space, visualise high-dimensional data, and generate synthetic samples.

When to prefer each algorithm
------------------------------
• K-Means          – fast, scalable; best when clusters are roughly spherical
                     and you know k in advance.
• Hierarchical     – builds a dendrogram; no need to pre-specify k; expensive O(n²).
• DBSCAN           – discovers arbitrary-shape clusters; robust to outliers;
                     sensitive to epsilon/min_samples.
• PCA              – linear dimensionality reduction; fast; captures directions
                     of maximum variance; useful pre-processing step.
• Autoencoder      – non-linear compression; learns compact latent representations;
                     needs more data than PCA.
• GAN              – generates realistic synthetic samples; hard to train stably;
                     best for large datasets.
• t-SNE / UMAP     – 2-D/3-D visualisation; t-SNE preserves local structure;
                     UMAP also preserves global structure and is faster.
"""

import numpy as np
from sklearn.cluster          import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition    import PCA
from sklearn.metrics          import silhouette_score
from sklearn.manifold         import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from utils.data_utils import (
    print_section, print_result, print_info, print_explain, print_score_bar,
)


# ── 1. K-Means Clustering ────────────────────────────────────────────────────

def run_kmeans(X):
    """
    K-Means Clustering
    -------------------
    Iteratively assigns each point to the nearest centroid and recomputes
    centroids until convergence.  We use k=3 (low/medium/high quality proxy).

    Best when: clusters are convex and similar in size; k is known; fast
    iteration is needed on large datasets.
    """
    print_section("1. K-Means Clustering  (sorting wines into 3 natural groups)")
    print_info("What is K-Means Clustering?")
    print_explain(
        "Imagine you tip a bag of differently coloured marbles onto a table and "
        "want to sort them into 3 piles WITHOUT being told what the piles should "
        "mean.  K-Means does exactly that with wine data!  "
        "It picks 3 'centre points' (centroids), assigns every wine to its nearest "
        "centre, moves the centres to the middle of their group, then repeats until "
        "nobody needs to move.  We don't tell it what 'good' means — it finds "
        "its own natural groupings.")
    print_info("No labels used — the algorithm discovers structure on its own.")

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    sil = silhouette_score(X, labels, sample_size=2000, random_state=42)
    print_score_bar("Silhouette score (0→1)", sil)
    print_result("Silhouette score", f"{sil:.4f}")
    print_explain(
        "The Silhouette score tells us how well-separated the 3 groups are.  "
        "Score close to +1.0 = wines within each group are very similar to each "
        "other AND very different from other groups (perfect separation!).  "
        "Score near 0 = groups overlap a lot.  "
        f"Our score of {sil:.2f} means the groups are "
        f"{'fairly well separated' if sil > 0.3 else 'somewhat overlapping — wine quality is a spectrum!'}.")
    print_result("Cluster sizes", dict(zip(*np.unique(labels, return_counts=True))))
    print_explain("Each cluster number is just a label the algorithm invented — "
                  "0, 1, 2 don't mean low/medium/high automatically.")
    return labels, sil


# ── 2. Hierarchical Clustering ───────────────────────────────────────────────

def run_hierarchical(X):
    """
    Hierarchical (Agglomerative) Clustering
    ----------------------------------------
    Merges the two closest clusters bottom-up (Ward linkage minimises the
    total within-cluster variance at each step).  Produces a dendrogram that
    lets you choose k after the fact.

    Best when: you do not know k in advance; dataset is small-to-medium; a
    dendrogram revealing cluster hierarchy is useful.
    """
    print_section("2. Hierarchical Clustering  (building a family tree of wines)")
    print_info("What is Hierarchical Clustering?")
    print_explain(
        "Think of it like building a family tree, but for wines.  Start with every "
        "wine as its own tiny group (leaf).  Then keep merging the two most similar "
        "groups together, step by step, until everything is one big group (root).  "
        "The result is a tree diagram called a 'dendrogram'.  "
        "You can cut the tree at any height to get any number of clusters — "
        "so you don't need to decide in advance how many groups you want!  "
        "We use a 2,000-wine sample here because the full dataset would be slow.")
    print_info("Ward linkage: merges groups in a way that keeps each group tight/compact.")

    # Subsample for speed (full dataset is 6497 rows)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=2000, replace=False)
    X_sub = X[idx]

    model = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels = model.fit_predict(X_sub)

    sil = silhouette_score(X_sub, labels)
    print_score_bar("Silhouette score (0→1)", sil)
    print_result("Silhouette score (2,000-sample subset)", f"{sil:.4f}")
    print_explain(
        f"Silhouette = {sil:.2f} on the 2,000-wine sample.  "
        "Same interpretation as K-Means: closer to 1 = better-defined groups.")
    print_result("Cluster sizes", dict(zip(*np.unique(labels, return_counts=True))))
    return labels, sil


# ── 3. DBSCAN ────────────────────────────────────────────────────────────────

def run_dbscan(X):
    """
    DBSCAN – Density-Based Spatial Clustering of Applications with Noise
    ----------------------------------------------------------------------
    Groups points that are densely packed, marking sparse regions as noise
    (label -1).  No need to specify k; finds arbitrary cluster shapes.

    Best when: clusters are non-convex; noise/outlier detection matters; the
    density of clusters is roughly uniform.
    """
    print_section("3. DBSCAN  (finding dense neighbourhoods, ignoring outliers)")
    print_info("What is DBSCAN?")
    print_explain(
        "Imagine you're at a party.  Groups of people chatting closely together "
        "are 'clusters'.  People standing alone in a corner are 'noise' (outliers).  "
        "DBSCAN works the same way with data: it looks for dense regions of wines "
        "that are similar to each other, and marks isolated wines as noise (label -1).  "
        "Unlike K-Means, you don't tell it how many groups to find — it discovers "
        "them based on density.  Great for finding unusual/outlier wines!")
    print_info("eps=0.8 = how close two wines must be to be 'neighbours'; "
               "min_samples=10 = minimum group size.")

    model = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print_result("Clusters found",  n_clusters)
    print_explain(
        f"DBSCAN found {n_clusters} dense group(s) naturally — "
        "we never told it how many to look for!")
    print_result("Noise points (outliers)",    n_noise)
    print_explain(
        f"{n_noise} wines were too isolated to belong to any group.  "
        "These could be unusual wines — maybe very high or very low quality outliers!")
    if n_clusters > 1:
        mask = labels != -1
        sil  = silhouette_score(X[mask], labels[mask], sample_size=2000,
                                random_state=42)
        print_score_bar("Silhouette score (0→1)", sil)
        print_result("Silhouette score (non-noise points)", f"{sil:.4f}")
    return labels, n_clusters


# ── 4. PCA ───────────────────────────────────────────────────────────────────

def run_pca(X):
    """
    Principal Component Analysis (PCA)
    ------------------------------------
    Rotates the data to the directions of maximum variance (eigenvectors of the
    covariance matrix).  Used to reduce dimensionality, visualise data, and
    remove correlated features before downstream modelling.

    Best when: features are correlated; you need fast linear compression;
    interpretability of principal components matters.
    """
    print_section("4. PCA  (squishing 12 features down to fewer dimensions)")
    print_info("What is PCA?")
    print_explain(
        "Our wines each have 12 measurements (features).  PCA is like taking "
        "a 3-D object and finding its 'shadow' — projecting it onto fewer dimensions "
        "while keeping as much interesting information as possible.  "
        "It finds the directions where the data 'spreads out' the most and keeps those.  "
        "For example, if alcohol and density are highly correlated, PCA can combine "
        "them into one 'super-feature' without much information loss.  "
        "Useful for visualisation (can't plot 12 dimensions!) and speeding up other models.")

    pca_full = PCA(random_state=42)
    pca_full.fit(X)

    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n95 = int(np.searchsorted(cum_var, 0.95)) + 1
    print_result("Components needed for 95% variance", n95)
    print_explain(
        f"Out of 12 original features, just {n95} 'super-features' (principal components) "
        f"can capture 95% of all the variation in the data.  "
        f"We can throw away {12 - n95} features with only 5% information loss!")
    top3_var = pca_full.explained_variance_ratio_[:3]
    for i, v in enumerate(top3_var):
        print_score_bar(f"PC{i+1} explains", v)
    print_result("Variance explained (first 3 PCs)",
                 [f"{v*100:.1f}%" for v in top3_var])
    print_explain(
        "PC1 is the single most informative 'summary' of all the wine measurements.  "
        "PC2 is the second most informative direction that PC1 missed, and so on.")

    pca2 = PCA(n_components=2, random_state=42)
    X_2d = pca2.fit_transform(X)
    print_result("2-D PCA shape (for visualisation)", X_2d.shape)
    print_explain("With just 2 components we can plot all 6,497 wines on a flat 2-D chart!")
    return X_2d, pca_full.explained_variance_ratio_


# ── 5. Autoencoder ───────────────────────────────────────────────────────────

def run_autoencoder(X):
    """
    Autoencoder (Neural Network)
    -----------------------------
    Encoder compresses the input to a low-dimensional *latent code*; decoder
    reconstructs the original input from that code.  Training minimises
    reconstruction error.  Learns non-linear manifolds that PCA cannot capture.

    Best when: data lies on a non-linear manifold; you need compact, learnable
    embeddings; anomaly detection (high reconstruction error = anomaly).
    """
    print_section("5. Autoencoder  (neural network that learns to compress and rebuild)")
    print_info("What is an Autoencoder?")
    print_explain(
        "An autoencoder is like a zip file for wine data, but smarter!  "
        "It has two halves: the Encoder squashes 12 wine measurements down to "
        "just 4 numbers (the 'latent code'), and the Decoder tries to rebuild "
        "the original 12 measurements from those 4 numbers.  "
        "It learns by checking: 'How different is the rebuilt version from the original?'  "
        "If the rebuilt version is almost identical, the 4-number code must contain "
        "everything important!  Unlike PCA, an autoencoder can capture curved "
        "(non-linear) patterns in the data because it uses neural network layers.")
    print_info("Compressing 12 features → 4 latent values → back to 12 features.")

    if not TF_AVAILABLE:
        print_info("SKIPPED – tensorflow not installed  (pip install tensorflow)")
        return None, float("nan")

    tf.random.set_seed(42)
    input_dim  = X.shape[1]
    latent_dim = 4

    # Encoder
    inp     = tf.keras.Input(shape=(input_dim,))
    enc     = tf.keras.layers.Dense(32, activation="relu")(inp)
    enc     = tf.keras.layers.Dense(latent_dim, activation="relu")(enc)
    # Decoder
    dec     = tf.keras.layers.Dense(32, activation="relu")(enc)
    out     = tf.keras.layers.Dense(input_dim, activation="linear")(dec)

    autoencoder = tf.keras.Model(inp, out)
    encoder     = tf.keras.Model(inp, enc)

    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X, X, epochs=30, batch_size=128,
                    validation_split=0.1, verbose=0)

    X_encoded = encoder.predict(X, verbose=0)
    recon     = autoencoder.predict(X, verbose=0)
    recon_err = np.mean((X - recon) ** 2)
    print_result("Mean reconstruction MSE (error)", f"{recon_err:.6f}")
    print_explain(
        f"Reconstruction error = {recon_err:.6f}  —  "
        "this measures how different the rebuilt wine data is from the original.  "
        f"A very small number like {recon_err:.4f} means the 4-number code is "
        "capturing most of the important wine information faithfully!")
    print_result("Latent representation shape (compressed)", X_encoded.shape)
    print_explain(
        f"Each of the {X_encoded.shape[0]:,} wines is now described by just "
        f"{X_encoded.shape[1]} numbers instead of {input_dim}.")
    return X_encoded, recon_err


# ── 6. GAN (Tabular) ─────────────────────────────────────────────────────────

def run_gan(X, epochs: int = 500):
    """
    Generative Adversarial Network – Tabular (TGAN)
    -------------------------------------------------
    A Generator G maps random noise z ~ N(0,I) to synthetic wine samples.
    A Discriminator D learns to distinguish real from fake.  Both are trained
    in a minimax game: G minimises – log D(G(z)), D maximises log D(x) + log(1–D(G(z))).

    Best when: you need realistic synthetic data (e.g., data augmentation or
    privacy-preserving datasets); large real datasets available for stable training.
    Notoriously difficult to train; mode collapse is a common failure mode.
    """
    print_section("6. GAN  (generating brand-new fake wines from scratch!)")
    print_info("What is a GAN?")
    print_explain(
        "GAN stands for Generative Adversarial Network.  It works like an "
        "art forger vs. an art detective game!  "
        "The Generator is the forger — it starts with random noise and tries to "
        "produce fake wine data that looks real.  "
        "The Discriminator is the detective — it tries to spot which wines are real "
        "and which are fake.  "
        "They compete: the forger gets better at fooling the detective, "
        "and the detective gets better at catching fakes.  "
        "After enough rounds, the Generator becomes SO good that it can create "
        "completely synthetic wines with realistic measurements!")
    print_info(f"Training for {epochs} rounds of forger-vs-detective competition.")

    if not TF_AVAILABLE:
        print_info("SKIPPED – tensorflow not installed  (pip install tensorflow)")
        return None, [], []

    tf.random.set_seed(42)
    n_features = X.shape[1]
    latent_dim = 16

    def make_generator():
        return tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="relu",
                                  input_shape=(latent_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(n_features, activation="linear"),
        ])

    def make_discriminator():
        return tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu",
                                  input_shape=(n_features,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])

    generator     = make_generator()
    discriminator = make_discriminator()

    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    g_opt = tf.keras.optimizers.Adam(1e-3)
    d_opt = tf.keras.optimizers.Adam(1e-3)

    X_tensor = tf.constant(X, dtype=tf.float32)
    batch_sz  = 128
    d_losses, g_losses = [], []

    for epoch in range(epochs):
        idx    = np.random.permutation(len(X))[:batch_sz]
        real   = tf.gather(X_tensor, idx)
        noise  = tf.random.normal([batch_sz, latent_dim])
        fake   = generator(noise, training=False)

        with tf.GradientTape() as tape:
            d_real  = discriminator(real,  training=True)
            d_fake  = discriminator(fake,  training=True)
            d_loss  = (cross_entropy(tf.ones_like(d_real),  d_real) +
                       cross_entropy(tf.zeros_like(d_fake), d_fake))
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_opt.apply_gradients(zip(grads, discriminator.trainable_variables))

        noise = tf.random.normal([batch_sz, latent_dim])
        with tf.GradientTape() as tape:
            fake   = generator(noise, training=True)
            d_fake = discriminator(fake, training=False)
            g_loss = cross_entropy(tf.ones_like(d_fake), d_fake)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_opt.apply_gradients(zip(grads, generator.trainable_variables))

        d_losses.append(float(d_loss))
        g_losses.append(float(g_loss))

    # Generate some synthetic samples
    synth = generator(tf.random.normal([10, latent_dim]), training=False).numpy()
    print_result("GAN training rounds completed", epochs)
    print_result("Final detective (discriminator) loss", f"{d_losses[-1]:.4f}")
    print_explain(
        "When discriminator loss is near 0.69 (= log 2), neither player is winning — "
        "the detective can't tell real from fake!  That's the ideal GAN balance.")
    print_result("Final forger (generator) loss", f"{g_losses[-1]:.4f}")
    print_explain(
        "Lower generator loss = the forger is producing more convincing fakes.  "
        "If both losses hover around 0.69, the GAN has reached a good equilibrium.")
    print_result("Synthetic wine samples generated", synth.shape)
    print_explain(
        f"The GAN just invented {synth.shape[0]} completely new wines from random noise!  "
        "These don't exist in the real world — they are computer-generated wine profiles.")
    return synth, d_losses, g_losses


# ── 7. t-SNE & UMAP ─────────────────────────────────────────────────────────

def run_tsne_umap(X, y):
    """
    t-SNE and UMAP (Dimensionality Reduction / Visualisation)
    ----------------------------------------------------------
    Both embed high-dimensional data in 2-D for visualisation by preserving
    pairwise similarities.

    • t-SNE  – excellent local structure preservation; stochastic; slow O(n²);
               distances between clusters are not meaningful.
    • UMAP   – faster; also preserves global structure; can be used as a general
               preprocessing step (unlike t-SNE).

    Best when: you need to visually inspect cluster separation; no specific
    downstream modelling is required (non-invertible).
    """
    print_section("7a. t-SNE  (making a 2-D map of wine-world)")
    print_info("What is t-SNE?")
    print_explain(
        "t-SNE is like a magical map maker for high-dimensional data.  "
        "Our wines each have 12 measurements — that's 12 dimensions, "
        "impossible to draw!  t-SNE squishes all that down to just 2 dimensions "
        "(a flat map) while trying to keep 'similar wines close together'.  "
        "The result is a scatter plot where wines that taste similar cluster "
        "near each other.  We use 2,000 wines for speed.")
    print_info("Using 2,000-wine random sample for speed (t-SNE is slow on large sets).")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=2000, replace=False)
    X_sub, y_sub = X[idx], y[idx]

    tsne   = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_sub)
    print_result("t-SNE embedding shape", X_tsne.shape)
    print_explain(
        f"Each of the {X_tsne.shape[0]} wines now has just 2 coordinates (x, y) "
        "so you could plot them on a graph.  Wines that cluster together on the map "
        "have similar chemical profiles.  NOTE: the distances between clusters on "
        "a t-SNE map are NOT meaningful — only closeness within clusters matters.")

    print_section("7b. UMAP  (faster map maker that also preserves big-picture structure)")
    print_info("What is UMAP?")
    print_explain(
        "UMAP (Uniform Manifold Approximation and Projection) does a similar job "
        "to t-SNE but is much faster AND preserves both local AND global structure.  "
        "This means not only are similar wines close together, but the distances "
        "BETWEEN clusters are more meaningful.  "
        "UMAP is also more suitable as a pre-processing step before other models.")

    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap  = reducer.fit_transform(X_sub)
        print_result("UMAP embedding shape", X_umap.shape)
        print_explain(
            f"Same {X_umap.shape[0]} wines now plotted in 2-D via UMAP.  "
            "Compare with the t-SNE map: UMAP tends to produce more 'spread-out' "
            "clusters that reflect the true global layout of the data better.")
    else:
        print_info("SKIPPED – umap-learn not installed  (pip install umap-learn)")
        X_umap = None

    return X_tsne, X_umap, y_sub


# ── Entry point ───────────────────────────────────────────────────────────────

def run_all(X, y):
    results = {}
    results["kmeans"]       = run_kmeans(X)
    results["hierarchical"] = run_hierarchical(X)
    results["dbscan"]       = run_dbscan(X)
    results["pca"]          = run_pca(X)
    results["autoencoder"]  = run_autoencoder(X)
    results["gan"]          = run_gan(X, epochs=500)
    results["tsne_umap"]    = run_tsne_umap(X, y)
    return results
