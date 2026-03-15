"""
Supervised Learning Methods
============================
Supervised learning trains a model on *labeled* examples – pairs (X, y) – so
that it can generalise to unseen inputs.  The wine dataset has quality scores
(3-9) which can be treated as a **regression** target or collapsed into a
**binary classification** target (good ≥ 7 vs not-good < 7).

When to prefer each algorithm
------------------------------
• Linear / Logistic Regression  – baseline; highly interpretable; fast to train;
  works well when the decision boundary is roughly linear.
• Decision Tree  – completely interpretable; handles mixed feature types; prone
  to overfitting when deep.
• Random Forest  – corrects DT overfitting via bagging; robust; little tuning
  needed; slightly less interpretable.
• SVM  – excellent in high-dimensional spaces; effective with a clear margin;
  kernel trick handles non-linearity; slow for very large datasets.
• Neural Network  – learns arbitrary non-linear functions; shines on large data;
  requires more tuning and data.
• KNN  – zero training cost; naturally multi-class; degrades with high
  dimensionality (curse of dimensionality).
• Gradient Boosting (XGBoost/LightGBM)  – usually the strongest tabular learner;
  handles missing values; requires careful regularisation to avoid overfitting.
• Naive Bayes  – very fast; works with small data; strong independence assumption
  often violated in practice but still competitive for simple tasks.
"""

import numpy as np
from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from utils.data_utils import (
    print_section, print_info,
    classification_report_short, regression_report_short,
)


# ── 1. Linear Regression ─────────────────────────────────────────────────────

def run_linear_regression(X_train, X_test, y_train, y_test):
    """
    Linear Regression
    -----------------
    Models the target as a weighted sum of features: ŷ = Xw + b.
    Minimises mean-squared error via the ordinary-least-squares closed-form
    solution.  Used here to *predict wine quality* as a continuous score.

    Best when: the relationship between features and target is approximately
    linear; you need a fast, interpretable baseline.
    """
    print_section("1. Linear Regression  (regression on quality score)")
    print_info("Fits ŷ = Xw + b by minimising MSE – interpretable baseline.")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = regression_report_short(y_test, y_pred, "Linear Regression")
    top3 = np.argsort(np.abs(model.coef_))[-3:][::-1]
    print_info(f"Top-3 feature weights by magnitude: indices {list(top3)}")
    return metrics


# ── 2. Logistic Regression ───────────────────────────────────────────────────

def run_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Logistic Regression
    --------------------
    Applies the logistic (sigmoid) function to a linear combination of features
    to estimate class probabilities.  Trained by maximising the log-likelihood.
    Used here for **binary classification**: good (≥7) vs not-good.

    Best when: you need class probabilities; the boundary is roughly linear;
    regularisation (L1/L2) handles multicollinearity.
    """
    print_section("2. Logistic Regression  (binary: good vs not-good)")
    print_info("Sigmoid of linear combination → probability; fast, calibrated.")

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return classification_report_short(y_test, y_pred, "Logistic Regression")


# ── 3. Decision Tree ─────────────────────────────────────────────────────────

def run_decision_tree(X_train, X_test, y_train, y_test):
    """
    Decision Tree Classifier
    -------------------------
    Recursively partitions the feature space by choosing the split that
    maximises information gain (Gini impurity).  The result is a human-readable
    tree of if-else rules.

    Best when: explainability is critical; you need a quick single-model
    baseline; mixing feature types.  Limit depth to prevent overfitting.
    """
    print_section("3. Decision Tree  (binary classification)")
    print_info("Recursive binary splits on features – fully interpretable.")

    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = classification_report_short(y_test, y_pred, "Decision Tree")
    print_info(f"Tree depth used: {model.get_depth()}  |  "
               f"Leaves: {model.get_n_leaves()}")
    return metrics


# ── 4. Random Forest ─────────────────────────────────────────────────────────

def run_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """
    Random Forest Classifier
    -------------------------
    Builds many decorrelated decision trees (via bootstrap sampling + random
    feature subsets) and averages their votes.  Reduces variance compared to a
    single deep tree while retaining low bias.

    Best when: you want strong out-of-the-box performance on tabular data with
    minimal tuning; feature importance is useful; robustness to outliers matters.
    """
    print_section("4. Random Forest  (binary classification)")
    print_info("Ensemble of decorrelated trees → lower variance, higher accuracy.")

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = classification_report_short(y_test, y_pred, "Random Forest")

    importances = model.feature_importances_
    top3_idx  = np.argsort(importances)[-3:][::-1]
    top3_feats = [(feature_names[i], f"{importances[i]:.3f}") for i in top3_idx]
    print_info(f"Top-3 features: {top3_feats}")
    return metrics


# ── 5. Support Vector Machine ─────────────────────────────────────────────────

def run_svm(X_train, X_test, y_train, y_test):
    """
    Support Vector Machine (SVM)
    -----------------------------
    Finds the maximum-margin hyperplane that separates classes.  The RBF kernel
    maps data to a higher-dimensional space implicitly, allowing non-linear
    boundaries.

    Best when: the feature space is high-dimensional; you have a clear margin of
    separation; dataset size is moderate (scales as O(n²–n³)).
    """
    print_section("5. Support Vector Machine  (binary classification, RBF kernel)")
    print_info("Maximum-margin hyperplane with RBF kernel for non-linear boundary.")

    model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = classification_report_short(y_test, y_pred, "SVM (RBF)")
    print_info(f"Support vectors: {model.n_support_}")
    return metrics


# ── 6. Neural Network (Keras / TensorFlow) ────────────────────────────────────

def run_neural_network(X_train, X_test, y_train, y_test):
    """
    Neural Network / Deep Learning (MLP)
    --------------------------------------
    A multi-layer perceptron with ReLU activations and dropout regularisation.
    Learns hierarchical feature representations through back-propagation.

    Best when: the dataset is large and the decision boundary is highly non-linear;
    you can invest time in architecture search and hyper-parameter tuning;
    GPU compute is available.
    """
    print_section("6. Neural Network  (MLP binary classification)")
    print_info("Deep MLP with ReLU + Dropout – learns arbitrary non-linear boundaries.")

    if not TF_AVAILABLE:
        print_info("SKIPPED – tensorflow not installed  (pip install tensorflow)")
        return {"accuracy": float("nan"), "f1": float("nan")}

    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu",
                              input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=30, batch_size=64,
              validation_split=0.1, verbose=0)

    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    return classification_report_short(y_test, y_pred, "Neural Network (MLP)")


# ── 7. K-Nearest Neighbours ──────────────────────────────────────────────────

def run_knn(X_train, X_test, y_train, y_test):
    """
    K-Nearest Neighbours (KNN)
    ---------------------------
    Classifies each sample by majority vote of its k closest training points
    (Euclidean distance after scaling).  Non-parametric – no training step.

    Best when: local structure dominates; dataset is small-to-medium; you want
    a simple, instance-based baseline.  Poor with high dimensions or large n.
    """
    print_section("7. K-Nearest Neighbours  (binary classification, k=7)")
    print_info("Majority vote of 7 nearest neighbours – zero training, lazy learner.")

    model = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return classification_report_short(y_test, y_pred, "KNN (k=7)")


# ── 8. Gradient Boosting ─────────────────────────────────────────────────────

def run_gradient_boosting(X_train, X_test, y_train, y_test):
    """
    Gradient Boosting – XGBoost & LightGBM
    ----------------------------------------
    Both build an ensemble of weak trees in a stage-wise fashion, fitting each
    new tree to the negative gradient of the loss.

    • XGBoost  – regularised gradient boosting; level-wise tree growth; great
      general performance; native handling of missing values.
    • LightGBM – leaf-wise growth; histogram-based splits; faster on large
      datasets; competitive or superior accuracy.

    Best when: highest accuracy on tabular data matters; features may have
    missing values; you can tune regularisation hyperparameters.
    """
    print_section("8a. XGBoost  (binary classification)")
    print_info("Stage-wise boosting of shallow trees – typically best tabular learner.")

    if XGB_AVAILABLE:
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        m1 = classification_report_short(y_test, y_pred_xgb, "XGBoost")
    else:
        print_info("SKIPPED – xgboost not installed  (pip install xgboost)")
        m1 = {"accuracy": float("nan"), "f1": float("nan")}

    print_section("8b. LightGBM  (binary classification)")
    print_info("Leaf-wise boosting with histograms – faster than XGBoost on large data.")

    if LGB_AVAILABLE:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        lgb_model.fit(X_train, y_train)
        y_pred_lgb = lgb_model.predict(X_test)
        m2 = classification_report_short(y_test, y_pred_lgb, "LightGBM")
    else:
        print_info("SKIPPED – lightgbm not installed  (pip install lightgbm)")
        m2 = {"accuracy": float("nan"), "f1": float("nan")}

    return m1, m2


# ── 9. Naive Bayes ────────────────────────────────────────────────────────────

def run_naive_bayes(X_train, X_test, y_train, y_test):
    """
    Gaussian Naive Bayes
    ---------------------
    Applies Bayes' theorem assuming features are **conditionally independent**
    given the class and Gaussian-distributed within each class.  Despite the
    strong independence assumption it often performs surprisingly well.

    Best when: dataset is small; features are reasonably independent; speed and
    simplicity are paramount; text classification (with Multinomial variant).
    """
    print_section("9. Naive Bayes  (binary classification, Gaussian)")
    print_info("Bayes' theorem + Gaussian likelihoods – blazing fast, good baseline.")

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return classification_report_short(y_test, y_pred, "Naive Bayes")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_all(X_train, X_test, y_train_reg, y_test_reg,
            y_train_bin, y_test_bin, feature_names):
    """Run every supervised learning method and return a summary dict."""
    results = {}
    results["linear_regression"]    = run_linear_regression(
        X_train, X_test, y_train_reg, y_test_reg)
    results["logistic_regression"]  = run_logistic_regression(
        X_train, X_test, y_train_bin, y_test_bin)
    results["decision_tree"]        = run_decision_tree(
        X_train, X_test, y_train_bin, y_test_bin)
    results["random_forest"]        = run_random_forest(
        X_train, X_test, y_train_bin, y_test_bin, feature_names)
    results["svm"]                  = run_svm(
        X_train, X_test, y_train_bin, y_test_bin)
    results["neural_network"]       = run_neural_network(
        X_train, X_test, y_train_bin, y_test_bin)
    results["knn"]                  = run_knn(
        X_train, X_test, y_train_bin, y_test_bin)
    xgb_m, lgb_m                   = run_gradient_boosting(
        X_train, X_test, y_train_bin, y_test_bin)
    results["xgboost"]              = xgb_m
    results["lightgbm"]             = lgb_m
    results["naive_bayes"]          = run_naive_bayes(
        X_train, X_test, y_train_bin, y_test_bin)
    return results
