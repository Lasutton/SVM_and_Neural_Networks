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
    print_section, print_info, print_explain,
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
    print_section("1. Bagging  (50 trees, each trained on a different random sample)")
    print_info("What are Ensemble Methods?")
    print_explain(
        "Ensemble methods combine MANY models instead of relying on just one.  "
        "The idea: even if each individual model is a bit wrong, they tend to make "
        "DIFFERENT mistakes.  When you average or vote across many models, "
        "the individual mistakes cancel out and the right answer wins!  "
        "It's like asking 50 different people to guess how many jellybeans are in "
        "a jar — the average of all guesses is usually closer than any single guess.")
    print_info("What is Bagging?")
    print_explain(
        "Bagging (Bootstrap AGGregatING) trains 50 decision trees, but each "
        "tree is trained on a DIFFERENT random sample of the wines.  "
        "'Bootstrap' means sampling WITH replacement — like picking cards from "
        "a deck, putting each card back, and shuffling again.  "
        "Some wines appear in multiple trees' training sets; some don't appear at all.  "
        "The wines left out of each tree's training (the 'out-of-bag' samples) "
        "provide a free accuracy estimate — shown as the OOB score below!")
    print_info("50 decision trees (max depth 8), each trained on a different bootstrap sample.")

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
    print_info(f"Out-of-bag (OOB) accuracy: {model.oob_score_:.4f}  ({model.oob_score_*100:.1f}%)")
    print_explain(
        f"OOB score = {model.oob_score_:.4f}  —  this is a FREE accuracy estimate!  "
        "For each tree, the wines it never saw in training are used to test it.  "
        "If the OOB score is close to the test accuracy, the model generalises well.  "
        "This means you don't even need a separate validation set — the training data "
        "itself tells you how well the model will do on unseen wines!")
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
    print_section("2a. AdaBoost  (200 stumps, each focusing on previous mistakes)")
    print_info("What is Boosting?")
    print_explain(
        "Boosting is like a relay race for weak learners.  "
        "Each model passes its mistakes to the next model, which focuses extra "
        "attention on the examples that were gotten wrong.  "
        "Unlike Bagging (where models are independent), Boosting models are "
        "SEQUENTIAL — each one must know what the previous one got wrong.")
    print_info("What is AdaBoost?")
    print_explain(
        "AdaBoost (Adaptive Boosting) uses 200 'decision stumps' — that's a "
        "Decision Tree with just ONE question (depth=1).  Incredibly simple!  "
        "After each stump votes, wrongly classified wines get their 'importance weight' "
        "increased so the NEXT stump pays more attention to those hard cases.  "
        "At the end, each stump's vote is weighted by how accurate it was — "
        "better stumps get louder voices.  "
        "AdaBoost is sensitive to noisy/mislabeled data because it keeps "
        "boosting the weight of any point it can't classify.")
    print_info("200 stumps (depth=1 trees), learning rate=0.5.")

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
    print_section("2b. Gradient Boosting  (200 trees, each fixing the previous residuals)")
    print_info("What is Gradient Boosting?")
    print_explain(
        "Gradient Boosting is like a team of sculptors.  The first sculptor "
        "makes a rough statue (first tree).  The second sculptor looks only at "
        "what the first got wrong and fixes those parts (second tree).  "
        "The third fixes what the first two missed, and so on.  "
        "Mathematically, each tree fits the 'gradient' of the error — pointing "
        "in the direction that will reduce mistakes fastest.  "
        "Gradient Boosting tends to achieve the best accuracy on tables of numbers "
        "like our wine data.  The trade-off: it's slow to train and can overfit.")
    print_info("200 trees, depth=4, learning rate=0.05 (small steps for stability).")

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
    print_section("2c. XGBoost  (the champion of tabular data competitions)")
    print_info("What makes XGBoost special?")
    print_explain(
        "XGBoost (eXtreme Gradient Boosting) is the most famous boosting library — "
        "it won hundreds of Kaggle data science competitions!  "
        "It's faster than sklearn's Gradient Boosting because it:  "
        "(1) Builds trees in parallel rather than one-at-a-time.  "
        "(2) Uses clever histogram approximations to find splits faster.  "
        "(3) Has built-in L1 and L2 regularisation to prevent overfitting.  "
        "(4) Handles missing values natively (no need to fill them in first).  "
        "On our wine dataset it's the single strongest individual model!")
    print_info("300 trees, depth=5, learning rate=0.05, column subsampling=80%.")

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
    print_section("3. Stacking  (a panel of experts, judged by a meta-expert)")
    print_info("What is Stacking?")
    print_explain(
        "Stacking is the most sophisticated ensemble method.  "
        "Instead of just voting, it uses a TWO-LEVEL system:  "
        "Level 1 — Four diverse experts each give their prediction:  "
        "  • Random Forest (tree-based, handles non-linearity)  "
        "  • Logistic Regression (linear, calibrated probabilities)  "
        "  • K-Nearest Neighbours (distance-based, local patterns)  "
        "  • Naive Bayes (probability-based, very fast)  "
        "Level 2 — A 'meta-learner' (another Logistic Regression) learns "
        "WHICH expert to trust for WHICH types of wine.  "
        "Maybe the KNN expert is great at identifying obvious 'great' wines, "
        "but the Random Forest is better at borderline cases.  "
        "The meta-learner figures this out automatically!  "
        "This is often the highest-accuracy approach.")
    print_info("Base: RandomForest + LogisticRegression + KNN + NaiveBayes.  "
               "Meta: LogisticRegression.  5-fold cross-validation stacking.")

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
    print_info(f"3-fold cross-validation accuracy: "
               f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print_explain(
        f"Cross-validation mean = {cv_scores.mean():.4f}, std = {cv_scores.std():.4f}.  "
        "This tests the model across 3 different data splits to make sure the result "
        "isn't just lucky on one particular test set.  "
        f"Small std ({cv_scores.std():.4f}) = consistent, reliable performance!")
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
