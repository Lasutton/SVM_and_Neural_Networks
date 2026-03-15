"""
Semi-Supervised Learning Methods
==================================
Semi-supervised learning exploits a **small labeled set** together with a
**large pool of unlabeled data**.  In practice, labeling data is expensive and
time-consuming, while raw data is cheap.  These methods bridge the gap.

When to prefer each algorithm
------------------------------
• Self-training      – simple wrapper around any supervised classifier; iteratively
                       adds confident pseudo-labels to the training set.  Best when
                       you have a reasonably accurate base model to start with.
• Label Propagation  – graph-based; propagates labels through a similarity graph;
                       works well when similar samples tend to share the same label.
• Co-training        – trains two classifiers on different *views* (feature subsets);
                       each model labels examples for the other.  Best when two
                       conditionally independent views exist (e.g., text + image).
"""

import numpy as np
from sklearn.semi_supervised import LabelPropagation, SelfTrainingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import accuracy_score, f1_score

from utils.data_utils import (
    print_section, print_result, print_info,
    split_labeled_unlabeled, classification_report_short,
)


# ── 1. Self-training ─────────────────────────────────────────────────────────

def run_self_training(X_train, X_test, y_train, y_test,
                      labeled_fraction: float = 0.10):
    """
    Self-training
    -------------
    Wraps a base classifier (here: Logistic Regression with probability
    estimates).  In each iteration the model predicts labels for all unlabeled
    samples and permanently adds the ones whose confidence exceeds a threshold.
    Stops when no new samples are added or the maximum iteration count is reached.

    Best when: you have a decent base learner; the labeled set is small but
    representative; unlabeled data is abundant.
    """
    print_section("1. Self-training  (base: Logistic Regression, 10 % labeled)")
    print_info(f"Iteratively pseudo-labels high-confidence unlabeled samples "
               f"({int(labeled_fraction*100)}% labeled seed).")

    X_l, y_l, X_ul, _ = split_labeled_unlabeled(
        X_train, y_train, labeled_fraction=labeled_fraction)

    # sklearn SelfTrainingClassifier uses -1 to mark unlabeled entries
    X_all = np.vstack([X_l, X_ul])
    y_all = np.concatenate([y_l, -np.ones(len(X_ul), dtype=int)])

    base = LogisticRegression(max_iter=1000, random_state=42)
    model = SelfTrainingClassifier(base, threshold=0.85, max_iter=20, verbose=False)
    model.fit(X_all, y_all)

    y_pred = model.predict(X_test)
    metrics = classification_report_short(y_test, y_pred, "Self-training")

    # Compare against a purely supervised model trained on labeled only
    base_sup = LogisticRegression(max_iter=1000, random_state=42)
    base_sup.fit(X_l, y_l)
    y_pred_sup = base_sup.predict(X_test)
    acc_sup = accuracy_score(y_test, y_pred_sup)
    print_info(f"Supervised-only (labeled only) accuracy: {acc_sup:.4f}  "
               f"→ self-training gain: "
               f"{metrics['accuracy'] - acc_sup:+.4f}")
    return metrics


# ── 2. Label Propagation ─────────────────────────────────────────────────────

def run_label_propagation(X_train, X_test, y_train, y_test,
                          labeled_fraction: float = 0.10):
    """
    Label Propagation
    ------------------
    Constructs a similarity graph between all training samples.  Labels are
    iteratively spread across edges proportional to sample similarity until
    convergence.  The unlabeled samples' labels are inferred from their
    labeled neighbours.

    Best when: data has a smooth manifold structure; the labeled examples are
    distributed across clusters; graph construction cost is acceptable (O(n²)).
    """
    print_section("2. Label Propagation  (RBF kernel, 10 % labeled)")
    print_info("Spreads labels across a similarity graph – assumes manifold smoothness.")

    X_l, y_l, X_ul, _ = split_labeled_unlabeled(
        X_train, y_train, labeled_fraction=labeled_fraction)

    # Subsample to avoid O(n²) memory blow-up
    rng = np.random.default_rng(42)
    max_ul = 2000
    if len(X_ul) > max_ul:
        idx  = rng.choice(len(X_ul), size=max_ul, replace=False)
        X_ul = X_ul[idx]

    X_all = np.vstack([X_l, X_ul])
    y_all = np.concatenate([y_l, -np.ones(len(X_ul), dtype=int)])

    model = LabelPropagation(kernel="rbf", gamma=0.25, max_iter=1000)
    model.fit(X_all, y_all)

    y_pred = model.predict(X_test)
    return classification_report_short(y_test, y_pred, "Label Propagation")


# ── 3. Co-training ───────────────────────────────────────────────────────────

def run_co_training(X_train, X_test, y_train, y_test,
                    labeled_fraction: float = 0.10, iterations: int = 10,
                    k_best: int = 10):
    """
    Co-training (manual implementation)
    -------------------------------------
    Splits features into two *views* (physicochemical and compositional) and
    trains a separate classifier on each.  Classifiers take turns labeling the
    k most confident unlabeled examples for the *other* classifier.

    Assumption: the two views are *conditionally independent* given the label,
    which ensures that each model's predictions add genuine information.

    Best when: two naturally separate but complementary feature sets exist
    (multi-view data); each view is sufficient on its own for some accuracy.
    """
    print_section("3. Co-training  (2 views, 10 % labeled, 10 iterations)")
    print_info("Two classifiers on complementary feature views label each other's data.")

    X_l, y_l, X_ul, _ = split_labeled_unlabeled(
        X_train, y_train, labeled_fraction=labeled_fraction)

    # View 1: acidity / sulfur features (indices 0-6)
    # View 2: density / alcohol / colour features (indices 7-11)
    v1 = list(range(7))
    v2 = list(range(7, X_train.shape[1]))

    X_l1, X_l2   = X_l[:, v1], X_l[:, v2]
    X_ul1, X_ul2 = X_ul[:, v1], X_ul[:, v2]

    clf1 = LogisticRegression(max_iter=1000, random_state=1)
    clf2 = LogisticRegression(max_iter=1000, random_state=2)

    # Pool of unlabeled indices
    pool_idx = list(range(len(X_ul1)))

    for it in range(iterations):
        clf1.fit(X_l1, y_l)
        clf2.fit(X_l2, y_l)

        if not pool_idx:
            break

        Xu1 = X_ul1[pool_idx]
        Xu2 = X_ul2[pool_idx]

        # Each classifier picks its k most confident predictions
        prob1 = clf1.predict_proba(Xu1)
        prob2 = clf2.predict_proba(Xu2)
        conf1 = prob1.max(axis=1)
        conf2 = prob2.max(axis=1)

        top_k = min(k_best, len(pool_idx))
        new1  = np.argsort(conf1)[-top_k:]   # clf1 labels for clf2
        new2  = np.argsort(conf2)[-top_k:]   # clf2 labels for clf1

        def add_samples(which_new, proba, ul1_arr, ul2_arr):
            nonlocal X_l1, X_l2, y_l, pool_idx
            selected = [pool_idx[i] for i in which_new]
            pseudo_y = proba[which_new].argmax(axis=1)
            # Map pseudo_y back to real class labels
            pseudo_y = clf1.classes_[pseudo_y]
            X_l1  = np.vstack([X_l1, ul1_arr[which_new]])
            X_l2  = np.vstack([X_l2, ul2_arr[which_new]])
            y_l   = np.concatenate([y_l, pseudo_y])
            pool_idx = [p for p in pool_idx if p not in selected]

        add_samples(new1, prob1, Xu1, Xu2)
        add_samples(new2, prob2, Xu1, Xu2)

    # Final prediction: average probabilities from both classifiers
    Xt1 = X_test[:, v1]
    Xt2 = X_test[:, v2]
    p1  = clf1.predict_proba(Xt1)
    p2  = clf2.predict_proba(Xt2)
    # Align class arrays
    classes = np.union1d(clf1.classes_, clf2.classes_)
    avg_prob = np.zeros((len(X_test), len(classes)))
    for i, c in enumerate(classes):
        if c in clf1.classes_:
            avg_prob[:, i] += p1[:, list(clf1.classes_).index(c)]
        if c in clf2.classes_:
            avg_prob[:, i] += p2[:, list(clf2.classes_).index(c)]
    y_pred = classes[avg_prob.argmax(axis=1)]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print_result("Co-training Accuracy",     f"{acc:.4f}")
    print_result("Co-training F1 (weighted)", f"{f1:.4f}")
    return {"accuracy": acc, "f1": f1}


# ── Entry point ───────────────────────────────────────────────────────────────

def run_all(X_train, X_test, y_train, y_test):
    results = {}
    results["self_training"]      = run_self_training(X_train, X_test, y_train, y_test)
    results["label_propagation"]  = run_label_propagation(X_train, X_test, y_train, y_test)
    results["co_training"]        = run_co_training(X_train, X_test, y_train, y_test)
    return results
