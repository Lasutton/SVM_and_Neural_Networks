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

from utils.data_utils import print_section, print_result, print_info


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
    print_section("1. K-Means Clustering  (k=3)")
    print_info("Partitions wines into 3 groups by minimising within-cluster variance.")

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    sil = silhouette_score(X, labels, sample_size=2000, random_state=42)
    print_result("Silhouette score", f"{sil:.4f}")
    print_result("Cluster sizes", dict(zip(*np.unique(labels, return_counts=True))))
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
    print_section("2. Hierarchical Clustering  (Ward linkage, k=3)")
    print_info("Bottom-up cluster merging – reveals nested structure without fixing k first.")

    # Subsample for speed (full dataset is 6497 rows)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=2000, replace=False)
    X_sub = X[idx]

    model = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels = model.fit_predict(X_sub)

    sil = silhouette_score(X_sub, labels)
    print_result("Silhouette score (2 000-sample subset)", f"{sil:.4f}")
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
    print_section("3. DBSCAN  (eps=0.8, min_samples=10)")
    print_info("Density-based clustering – labels noise as -1, no fixed k.")

    model = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print_result("Clusters found",  n_clusters)
    print_result("Noise points",    n_noise)
    if n_clusters > 1:
        mask = labels != -1
        sil  = silhouette_score(X[mask], labels[mask], sample_size=2000,
                                random_state=42)
        print_result("Silhouette score (non-noise)", f"{sil:.4f}")
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
    print_section("4. PCA  (Principal Component Analysis)")
    print_info("Linear projection onto directions of max variance.")

    pca_full = PCA(random_state=42)
    pca_full.fit(X)

    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n95 = int(np.searchsorted(cum_var, 0.95)) + 1
    print_result("Components for 95 % variance", n95)
    print_result("Explained variance (first 3 PCs)",
                 [f"{v:.3f}" for v in pca_full.explained_variance_ratio_[:3]])

    pca2 = PCA(n_components=2, random_state=42)
    X_2d = pca2.fit_transform(X)
    print_result("2-D PCA shape", X_2d.shape)
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
    print_section("5. Autoencoder  (encoding dim=4)")
    print_info("Non-linear encoder/decoder pair – learns a compressed latent space.")

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
    print_result("Mean reconstruction MSE", f"{recon_err:.6f}")
    print_result("Latent representation shape", X_encoded.shape)
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
    print_section("6. Generative Adversarial Network  (tabular, mini demo)")
    print_info("Generator + Discriminator play a minimax game – learns data distribution.")

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
    print_result("GAN training epochs", epochs)
    print_result("Final discriminator loss", f"{d_losses[-1]:.4f}")
    print_result("Final generator loss",     f"{g_losses[-1]:.4f}")
    print_result("Synthetic samples shape",  synth.shape)
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
    print_section("7. t-SNE  (2-D embedding, subset 2 000)")
    print_info("Non-linear 2-D projection preserving local neighbourhoods.")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=2000, replace=False)
    X_sub, y_sub = X[idx], y[idx]

    tsne   = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_sub)
    print_result("t-SNE embedding shape", X_tsne.shape)

    print_section("7b. UMAP  (2-D embedding, subset 2 000)")
    print_info("Uniform Manifold Approximation – faster and preserves global structure.")

    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap  = reducer.fit_transform(X_sub)
        print_result("UMAP embedding shape", X_umap.shape)
    else:
        print_info("SKIPPED – umap-learn not installed  (pip install umap-learn)")
        X_umap = None

    return X_tsne, X_umap, y_sub


# ── Entry point ───────────────────────────────────────────────────────────────

def run_all(X, y):
    results = {}
    results["kmeans"]      = run_kmeans(X)
    results["hierarchical"] = run_hierarchical(X)
    results["dbscan"]      = run_dbscan(X)
    results["pca"]         = run_pca(X)
    results["autoencoder"] = run_autoencoder(X)
    results["gan"]         = run_gan(X, epochs=500)
    results["tsne_umap"]   = run_tsne_umap(X, y)
    return results
