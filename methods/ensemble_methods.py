"""
Ensemble Methods
=================
Ensemble methods combine multiple *base learners* to produce a prediction that
is more accurate and robust than any individual model.  The key intuition is
that errors made by different models on different samples cancel out when their
predictions are aggregated.

When to prefer each algorithm
------------------------------
• Bagging     – reduces variance; best when the base learner overfits (e.g.
                deep decision trees); naturally parallelisable.
• Boosting    – reduces bias; sequentially corrects residual errors; best when
                base learners are weak (shallow trees); may overfit on noisy data.
• Stacking    – meta-level learning; combines heterogeneous models; most flexible;
                highest computational cost; often the most accurate.
"""

from sklearn.ensemble         import (BaggingClassifier, AdaBoostClassifier,
                                       GradientBoostingClassifier,
                                       StackingClassifier, RandomForestClassifier)
from sklearn.tree             import DecisionTreeClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.naive_bayes      import GaussianNB
from sklearn.model_selection  import cross_val_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from utils.data_utils import (
    print_section, print_info,
    classification_report_short,
)


# ── 1. Bagging ────────────────────────────────────────────────────────────────

def run_bagging(X_train, X_test, y_train, y_test):
    """
    Bagging – Bootstrap Aggregating
    ---------------------------------
    Trains B independent copies of the same base learner on B bootstrap samples
    (drawn with replacement from the training set) and averages (or majority-
    votes) their predictions.

    Key mechanisms:
    • Bootstrap sampling decorrelates the base models.
    • Each model sees ~63 % of unique samples (the rest form the "out-of-bag"
      set, which gives a free validation estimate).

    Variants:
    • Random Forest is a special case of bagging with additional random feature
      subsampling at each split.
    • Pasting: bagging without replacement – useful when training set is small.

    Best when: the base learner has high variance (e.g. deep trees); you want
    to reduce overfitting without losing too much bias.
    """
    print_section("1. Bagging  (50 Decision Trees, bootstrap samples)")
    print_info("Bootstrap-sampled trees voted → reduces variance, not bias.")

    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=8),
        n_estimators=50,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = classification_report_short(y_test, y_pred, "Bagging (DT×50)")
    print_info(f"Out-of-bag score: {model.oob_score_:.4f}")
    return metrics


# ── 2a. AdaBoost ──────────────────────────────────────────────────────────────

def run_adaboost(X_train, X_test, y_train, y_test):
    """
    AdaBoost – Adaptive Boosting (Freund & Schapire, 1997)
    -------------------------------------------------------
    Fits a sequence of *weak learners* (stumps by default).  After each round
    misclassified samples receive higher weight so the next learner focuses on
    them.  The final prediction is a weighted majority vote.

    Best when: you have many weakly-correlated errors; base learner is a
    shallow tree; the dataset is noise-free (AdaBoost is sensitive to outliers).
    """
    print_section("2a. AdaBoost  (200 stumps, SAMME.R)")
    print_info("Sequential weighted combination of stumps – focuses on hard examples.")

    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),   # stumps
        n_estimators=200,
        learning_rate=0.5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return classification_report_short(y_test, y_pred, "AdaBoost")


# ── 2b. Gradient Boosting (sklearn) ──────────────────────────────────────────

def run_gradient_boosting_sklearn(X_train, X_test, y_train, y_test):
    """
    Gradient Boosting (sklearn, Friedman 2001)
    -------------------------------------------
    Fits trees stage-wise to the *negative gradient* of the loss function.
    Each new tree corrects the residuals of the current ensemble.  For binary
    cross-entropy the gradient is the residual error in predicted probabilities.

    Difference from AdaBoost:
    • AdaBoost re-weights samples; GBM fits residuals explicitly.
    • GBM is a more general framework that accommodates any differentiable loss.

    Best when: you need the best accuracy on tabular data; you can tune the
    learning rate, max_depth, and n_estimators carefully.
    """
    print_section("2b. Gradient Boosting  (sklearn GBM, 200 trees)")
    print_info("Stage-wise tree fitting on pseudo-residuals – highest accuracy baseline.")

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return classification_report_short(y_test, y_pred, "GradientBoosting (sklearn)")


# ── 2c. XGBoost Boosting (for completeness in ensemble section) ──────────────

def run_xgboost_ensemble(X_train, X_test, y_train, y_test):
    """
    XGBoost – Extreme Gradient Boosting
    -------------------------------------
    An optimised, regularised gradient boosting implementation with:
    • L1 + L2 regularisation on leaf weights.
    • Approximate greedy split finding via histogram buckets.
    • Column (feature) subsampling for diversity.
    • Parallel tree construction.

    Best when: you want top tabular performance with built-in regularisation;
    large datasets where speed matters.
    """
    print_section("2c. XGBoost  (ensemble context, 300 trees)")
    print_info("Regularised GBM with column subsampling – fast and state-of-the-art.")

    if not XGB_AVAILABLE:
        print_info("SKIPPED – xgboost not installed  (pip install xgboost)")
        return {"accuracy": float("nan"), "f1": float("nan")}

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return classification_report_short(y_test, y_pred, "XGBoost (Ensemble)")


# ── 3. Stacking ───────────────────────────────────────────────────────────────

def run_stacking(X_train, X_test, y_train, y_test):
    """
    Stacking – Stacked Generalisation (Wolpert, 1992)
    ---------------------------------------------------
    A two-level architecture:
    • Level-0 (base learners): a diverse set of models trained on the full
      training set.  Their *out-of-fold* predictions form a new feature matrix Z.
    • Level-1 (meta-learner): a model trained on Z to learn how to best combine
      the base learner predictions.

    Why diversity matters: if all base learners make the same mistakes the meta-
    learner cannot fix them.  Using heterogeneous models (tree, linear, kNN)
    maximises diversity.

    Best when: you want to squeeze every last drop of accuracy; competition-
    style settings where the added complexity pays off; you have many diverse
    models already trained.
    """
    print_section("3. Stacking  (RF + LR + KNN → Logistic meta-learner)")
    print_info("Diverse base learners → out-of-fold predictions → meta-learner.")

    base_estimators = [
        ("rf",  RandomForestClassifier(n_estimators=100, random_state=42,
                                        n_jobs=-1)),
        ("lr",  LogisticRegression(max_iter=1000, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
        ("nb",  GaussianNB()),
    ]
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)

    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False,       # meta-learner sees only base predictions
    )
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_test)

    metrics = classification_report_short(y_test, y_pred, "Stacking")

    # Cross-validation score to assess generalisation
    cv_scores = cross_val_score(stacking, X_train, y_train, cv=3,
                                scoring="accuracy", n_jobs=-1)
    print_info(f"3-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def run_all(X_train, X_test, y_train, y_test):
    results = {}
    results["bagging"]               = run_bagging(X_train, X_test, y_train, y_test)
    results["adaboost"]              = run_adaboost(X_train, X_test, y_train, y_test)
    results["gradient_boosting"]     = run_gradient_boosting_sklearn(
        X_train, X_test, y_train, y_test)
    results["xgboost_ensemble"]      = run_xgboost_ensemble(
        X_train, X_test, y_train, y_test)
    results["stacking"]              = run_stacking(X_train, X_test, y_train, y_test)
    return results
