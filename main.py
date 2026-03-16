#!/usr/bin/env python3
"""
Wine Quality – Complete Machine Learning Methods Demo
======================================================
Demonstrates every major ML paradigm using 6,497 wine samples from the UCI
Wine Quality dataset (11 physicochemical features + wine colour, quality 3-9).

Paradigms covered
-----------------
1. Supervised Learning       – learns from labeled (feature, label) pairs
2. Unsupervised Learning      – discovers hidden structure without labels
3. Semi-Supervised Learning   – exploits a small labeled set + large unlabeled pool
4. Reinforcement Learning     – agent maximises reward via trial-and-error
5. Self-Supervised Learning   – model generates its own training signal
6. Transfer Learning          – reuses knowledge from a related source domain
7. Ensemble Methods           – combines multiple learners for greater accuracy

Usage
-----
    python main.py [--skip <category>] [--only <category>]

    Categories: supervised, unsupervised, semi, rl, ssl, transfer, ensemble

    Examples:
        python main.py                         # run everything
        python main.py --only supervised       # only supervised learning
        python main.py --skip rl               # skip reinforcement learning
"""

import sys
import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── Suppress TF/XLA noise ─────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ── Ensure project root is on PYTHONPATH ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_utils import (
    load_wine_data, get_train_test,
    print_header, print_section, print_result, print_info, print_explain,
    print_score_bar, Colors,
)

# ── Lazy module imports (so --only flag skips heavy TF/torch imports) ─────────
def _import(module_path):
    import importlib
    return importlib.import_module(module_path)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


def print_summary_table(all_results: dict) -> None:
    """Print a compact cross-category accuracy summary with ASCII histograms."""
    print_header("ACCURACY HISTOGRAM — ALL MODELS AT A GLANCE", "=")
    print_explain(
        "Every bar below represents one model's accuracy (0% → 100%).  "
        "Longer green bar = better at identifying good vs not-good wine.  "
        "Models with RL reward scores use a converted scale so they fit the same chart.")

    rows = []
    for category, res in all_results.items():
        if isinstance(res, dict):
            for name, metrics in res.items():
                if isinstance(metrics, dict) and "accuracy" in metrics:
                    rows.append((category, name, metrics["accuracy"]))
                elif isinstance(metrics, (int, float)):
                    # RL avg rewards: convert from [-1,+1] to [0,1]
                    import math
                    if not math.isnan(metrics):
                        rows.append((category, name, (metrics + 1) / 2.0))
                elif isinstance(metrics, tuple) and len(metrics) == 2:
                    # (losses_list, accuracy) from SSL
                    if isinstance(metrics[1], float):
                        import math
                        if not math.isnan(metrics[1]):
                            rows.append((category, name, metrics[1]))

    if not rows:
        print("  (no numeric metrics collected)")
        return

    rows.sort(key=lambda r: -r[2])
    print(f"\n  {'Category':<22} {'Model':<32}")
    print(f"  {'-'*22} {'-'*32}")
    for cat, name, score in rows:
        label = f"{cat}/{name}"[:45]
        print_score_bar(label, score)
        val = f"{score*100:.1f}%"
        print(f"            {Colors.YELLOW}({val}){Colors.END}")


def print_final_comparison(all_results: dict) -> None:
    """
    Print a comprehensive plain-English comparison and contrast of every ML
    paradigm and individual model, with use-case guidance.
    """
    import math

    # ── Collect all classification accuracies for comparison ──────────────────
    supervised_acc = {}
    if "supervised" in all_results:
        for name, m in all_results["supervised"].items():
            if isinstance(m, dict) and "accuracy" in m:
                acc = m["accuracy"]
                if not math.isnan(acc):
                    supervised_acc[name] = acc
            elif isinstance(m, tuple):
                for sub in m:
                    if isinstance(sub, dict) and "accuracy" in sub:
                        acc = sub["accuracy"]
                        if not math.isnan(acc):
                            supervised_acc[name] = acc

    ensemble_acc = {}
    if "ensemble" in all_results:
        for name, m in all_results["ensemble"].items():
            if isinstance(m, dict) and "accuracy" in m:
                acc = m["accuracy"]
                if not math.isnan(acc):
                    ensemble_acc[name] = acc

    # ─────────────────────────────────────────────────────────────────────────
    print_header("FINAL MODEL COMPARISON & USE-CASE GUIDE", "█")
    print_explain(
        "This section compares EVERY model we just ran, explains what each one "
        "does in plain English, and tells you WHEN you should (or shouldn't) use it.")

    # ── PART 1: The 7 Learning Paradigms ──────────────────────────────────────
    print_section("PART 1 — The 7 Ways Machines Learn")
    print_explain("Think of these as 7 completely different study strategies:")
    print()

    paradigms = [
        ("1. Supervised Learning",
         "You show the model thousands of examples WITH the right answers labelled.  "
         "Like a student studying a textbook with an answer key.",
         "WHEN TO USE: You have labeled data and a clear prediction target.  "
         "Works for: spam filters, medical diagnosis, price prediction.  "
         "NOT good for: when labeling is expensive or impossible."),

        ("2. Unsupervised Learning",
         "No labels at all!  The model explores the data and finds patterns on its own.  "
         "Like sorting a pile of random objects into groups without being told the categories.",
         "WHEN TO USE: You want to discover hidden structure, compress data, or "
         "detect anomalies.  Works for: customer segmentation, fraud detection, "
         "image compression.  NOT good for: when you need specific predictions."),

        ("3. Semi-Supervised Learning",
         "You have a tiny bit of labeled data and a large pool of unlabeled data.  "
         "The model uses the labeled examples as a guide and learns from the unlabeled rest.  "
         "Like a teacher marking 10 homework papers and the student guessing the rest.",
         "WHEN TO USE: Labeling is expensive but raw data is abundant.  "
         "Works for: medical imaging (few annotated scans), web content classification.  "
         "NOT good for: when even unlabeled data is scarce."),

        ("4. Reinforcement Learning",
         "No dataset at all!  The agent learns by trial and error in an environment, "
         "getting rewarded for good actions and penalised for bad ones.  "
         "Like training a dog with treats — no textbook, just feedback.",
         "WHEN TO USE: Sequential decision-making problems with a clear reward signal.  "
         "Works for: game playing (chess, Go), robotics, recommendation systems, "
         "autonomous driving.  NOT good for: simple classification/regression tasks — "
         "overkill and slow compared to supervised learning."),

        ("5. Self-Supervised Learning",
         "The model creates its own labels from the raw data by solving puzzles "
         "(predict the masked feature, recognise augmented versions).  "
         "Like a student who makes their own flashcard quizzes from a textbook — "
         "no teacher needed!",
         "WHEN TO USE: Massive unlabeled datasets where you want to pre-train a "
         "powerful model cheaply.  Works for: language models (BERT, GPT), "
         "image models (MAE, DINO), tabular pre-training.  "
         "NOT good for: small datasets — the pretext task needs lots of data."),

        ("6. Transfer Learning",
         "Train on one task/domain (red wine), then reuse that knowledge for a "
         "related task/domain (white wine).  Like a physicist who switches to "
         "engineering — the maths skills transfer!",
         "WHEN TO USE: Your target domain has limited data but a related source "
         "domain has plenty.  Works for: medical AI (few sick patients, many healthy), "
         "cross-language NLP, domain-specific image recognition.  "
         "NOT good for: when source and target domains are completely unrelated."),

        ("7. Ensemble Methods",
         "Combine many models and let them vote.  Wisdom of the crowd applied to AI!  "
         "Even if each model is imperfect, the combined vote is often near-perfect.",
         "WHEN TO USE: When you need maximum accuracy on tabular/structured data "
         "and training time is not a constraint.  Works for: Kaggle competitions, "
         "financial prediction, clinical risk scores.  "
         "NOT good for: real-time inference (slow), when interpretability is critical."),
    ]

    for name, what, when in paradigms:
        print(f"\n  {Colors.CYAN}{Colors.BOLD}{name}{Colors.END}")
        print_explain(f"What it is: {what}")
        print_info(f"Use-case guide: {when}")

    # ── PART 2: Individual Model Shootout ─────────────────────────────────────
    print_section("PART 2 — Individual Model Comparison (Supervised + Ensemble)")
    print_explain(
        "Now let's compare every individual model we tested.  "
        "All models below were trained on the SAME wine data and evaluated on "
        "the SAME test set — a fair fight!")

    if supervised_acc or ensemble_acc:
        all_acc = {**supervised_acc, **ensemble_acc}
        best_model = max(all_acc, key=all_acc.get)
        best_score = all_acc[best_model]
        print_explain(
            f"Best individual model: '{best_model}' with {best_score*100:.1f}% accuracy.  "
            "But accuracy isn't the only thing that matters — see the full guide below.")

    models_guide = [
        ("Linear Regression",
         "Predicts a wine quality NUMBER (e.g. 6.2).  Simple, transparent, fast.",
         supervised_acc.get("linear_regression"),
         "USE WHEN: You need to predict a continuous value and want an interpretable model.  "
         "You can read the weights and say 'each extra 1% alcohol raises score by X'.  "
         "DON'T USE WHEN: The relationship is highly non-linear."),

        ("Logistic Regression",
         "Good-vs-bad classification.  Gives calibrated probabilities.  Very interpretable.",
         supervised_acc.get("logistic_regression"),
         "USE WHEN: You need probability estimates ('80% confident this is good wine').  "
         "Fast, reliable baseline for any binary classification.  "
         "DON'T USE WHEN: You need to capture complex non-linear patterns."),

        ("Decision Tree",
         "A flowchart of yes/no rules.  Fully readable by humans.",
         supervised_acc.get("decision_tree"),
         "USE WHEN: You need a model you can literally print out and show to a client.  "
         "Doctors and lawyers love interpretable rules.  "
         "DON'T USE WHEN: You need the highest accuracy — one tree overfits easily."),

        ("Random Forest",
         "200 trees voting together.  Robust, nearly zero tuning required.",
         supervised_acc.get("random_forest"),
         "USE WHEN: You want solid accuracy without spending hours tuning.  "
         "Gives FREE feature importance scores showing which wine measurements matter most.  "
         "DON'T USE WHEN: You need a fully interpretable single set of rules."),

        ("SVM (RBF)",
         "Finds the widest gap between good and bad wines in feature space.",
         supervised_acc.get("svm"),
         "USE WHEN: Dataset is small-to-medium and you have many features.  "
         "Highly effective in high-dimensional spaces (e.g., text classification).  "
         "DON'T USE WHEN: Dataset is large (>100K rows) — becomes very slow."),

        ("Neural Network (MLP)",
         "A brain of connected layers that learns complex non-linear patterns.",
         supervised_acc.get("neural_network"),
         "USE WHEN: Your dataset is large and the patterns are deeply non-linear.  "
         "Scales beautifully with more data and compute (GPUs).  "
         "DON'T USE WHEN: Your dataset is small (<1K rows) — will memorise and overfit.  "
         "Also needs hyperparameter tuning to shine."),

        ("KNN (k=7)",
         "Asks the 7 most similar wines in the training set and takes a vote.",
         supervised_acc.get("knn"),
         "USE WHEN: Dataset is small, local structure matters, and training time is critical "
         "(KNN has zero training time!).  "
         "DON'T USE WHEN: Dataset is large (slow prediction) or has many features "
         "(distances become meaningless in high dimensions)."),

        ("XGBoost",
         "300 trees boosted sequentially.  The king of tabular data competitions.",
         supervised_acc.get("xgboost"),
         "USE WHEN: You need the highest accuracy on structured/tabular data.  "
         "Handles missing values natively, built-in regularisation, fast.  "
         "DON'T USE WHEN: You need a simple, interpretable model.  "
         "Hyperparameter tuning is needed to get the most out of it."),

        ("LightGBM",
         "Like XGBoost but faster.  Preferred on very large datasets.",
         supervised_acc.get("lightgbm"),
         "USE WHEN: Dataset has millions of rows (XGBoost gets slow).  "
         "Otherwise nearly identical use-case to XGBoost.  "
         "DON'T USE WHEN: Dataset is small — the histogram approximation loses value."),

        ("Naive Bayes",
         "Probability-based.  Blazing fast.  Assumes features are independent.",
         supervised_acc.get("naive_bayes"),
         "USE WHEN: Speed is critical (email spam filters process millions/second).  "
         "Works surprisingly well for text classification despite the 'naive' assumption.  "
         "DON'T USE WHEN: Feature interactions are important (wine chemistry features "
         "DO interact — alcohol + acidity together predict quality better than either alone)."),

        ("Bagging (50 trees)",
         "50 trees, each trained on a different random sample, then majority vote.",
         ensemble_acc.get("bagging"),
         "USE WHEN: Your single model overfits and you want to reduce variance.  "
         "Free OOB (out-of-bag) score means you don't need a separate validation set.  "
         "DON'T USE WHEN: Training time is a concern (50× slower than one tree)."),

        ("AdaBoost",
         "200 stumps, each focusing on previous mistakes.",
         ensemble_acc.get("adaboost"),
         "USE WHEN: Data is clean (no outliers) and you want a simple boosting baseline.  "
         "DON'T USE WHEN: Data has noisy/mislabeled points — AdaBoost amplifies noise."),

        ("Gradient Boosting",
         "Builds trees stage-wise to correct residual errors.",
         ensemble_acc.get("gradient_boosting"),
         "USE WHEN: Maximum accuracy on tabular data, you can tune parameters.  "
         "Slower to train than XGBoost/LightGBM but part of the standard sklearn toolkit.  "
         "DON'T USE WHEN: Training time is limited or you need a simple baseline."),

        ("XGBoost (Ensemble)",
         "Same as XGBoost above — included for direct ensemble comparison.",
         ensemble_acc.get("xgboost_ensemble"),
         "USE WHEN: Competition settings or production systems requiring top accuracy.  "
         "Almost always outperforms traditional Gradient Boosting on tabular data."),

        ("Stacking",
         "RF + LR + KNN + NaiveBayes → meta-learner decides who to trust.",
         ensemble_acc.get("stacking"),
         "USE WHEN: You want to squeeze every last percentage point of accuracy and "
         "training time doesn't matter.  Best for Kaggle competitions.  "
         "DON'T USE WHEN: You need a fast, interpretable, or production-friendly model.  "
         "Stacking is complex to deploy and maintain."),
    ]

    print()
    for model_name, what, acc, guidance in models_guide:
        acc_str = f"{acc*100:.1f}%" if acc is not None else "not run"
        print(f"\n  {Colors.CYAN}{Colors.BOLD}● {model_name}{Colors.END}  "
              f"{Colors.YELLOW}[accuracy: {acc_str}]{Colors.END}")
        if acc is not None:
            print_score_bar(f"  {model_name}", acc)
        print_explain(f"What it does: {what}")
        print_info(guidance)

    # ── PART 3: Quick Reference Decision Guide ────────────────────────────────
    print_section("PART 3 — Which Model Should YOU Choose?")
    print_explain(
        "Not sure which model to pick for your own project?  "
        "Use this quick reference guide:")
    print()

    scenarios = [
        ("I need maximum accuracy, I don't care about interpretability",
         "→ Try XGBoost / LightGBM first, then Stacking."),
        ("I need to explain the model to a non-technical person",
         "→ Use Decision Tree (for a small flowchart) or Logistic Regression "
         "(for a list of weighted features)."),
        ("I have very little data (< 500 examples)",
         "→ Try Naive Bayes, Logistic Regression, or KNN.  "
         "Avoid deep neural networks — they'll overfit."),
        ("I have a huge dataset (millions of rows)",
         "→ LightGBM is the fastest strong model.  "
         "Neural Networks are great if you have a GPU."),
        ("I have mostly unlabeled data with very few labels",
         "→ Use Semi-Supervised Learning (Self-Training or Label Propagation)."),
        ("I have NO labels at all",
         "→ Use Unsupervised Learning (K-Means for grouping, PCA for compression, "
         "Autoencoder for anomaly detection)."),
        ("I want to pre-train a model cheaply with no labels",
         "→ Use Self-Supervised Learning (Masked Modelling or Contrastive Learning)."),
        ("I have lots of data from a similar domain but few target labels",
         "→ Use Transfer Learning (fine-tuning or feature extraction)."),
        ("I'm building a game-playing AI or robot controller",
         "→ Use Reinforcement Learning (PPO for stability, DQN for discrete actions)."),
        ("I want a good out-of-the-box model with no tuning",
         "→ Random Forest.  It's robust, fast enough, and needs almost no configuration."),
        ("I'm in a Kaggle competition and want to win",
         "→ XGBoost/LightGBM + Stacking + Pseudo-labelling (semi-supervised trick)."),
        ("I want to detect outliers / anomalies",
         "→ DBSCAN or Autoencoder (high reconstruction error = anomaly)."),
        ("I need to generate synthetic data",
         "→ GAN (Generative Adversarial Network)."),
        ("I want to visualise my data in 2D",
         "→ t-SNE (for local cluster exploration) or UMAP (for overall structure)."),
    ]

    for question, answer in scenarios:
        print(f"  {Colors.BLUE}?{Colors.END}  {Colors.BOLD}{question}{Colors.END}")
        print(f"     {Colors.GREEN}{answer}{Colors.END}")
        print()

    # ── PART 4: The Big Picture ───────────────────────────────────────────────
    print_section("PART 4 — The Big Picture: Key Takeaways")
    print()
    takeaways = [
        "No single model wins everywhere — always try multiple and compare.",
        "More data almost always helps more than a fancier model.",
        "Always start with a SIMPLE baseline (Logistic Regression) before "
        "moving to complex models.  If it works well, stop there!",
        "Ensemble methods (Bagging, Boosting, Stacking) are almost always "
        "more accurate than any single model — but take longer to train.",
        "Interpretability vs accuracy is a real trade-off.  "
        "Decision Tree = most interpretable.  Stacking = most accurate.  "
        "Random Forest is a good middle ground.",
        "Overfitting = memorising the training data instead of learning general patterns.  "
        "Signs: training accuracy >> test accuracy.  "
        "Fix: more data, simpler model, regularisation, or Dropout.",
        "RL is powerful but complex — for most real-world tasks, supervised learning "
        "will get you further faster.  Use RL only when sequential decision-making "
        "and a reward signal are genuinely present.",
        "Transfer Learning and Self-Supervised Learning are the future of AI — "
        "they dramatically reduce the need for expensive labeled data.",
    ]
    for i, t in enumerate(takeaways, 1):
        print(f"  {Colors.YELLOW}{Colors.BOLD}{i}.{Colors.END}  {t}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Category runners
# ─────────────────────────────────────────────────────────────────────────────

def run_supervised(X_train, X_test, y_train, y_test,
                   y_train_bin, y_test_bin, feature_names):
    print_header("SUPERVISED LEARNING", "█")
    print_info(
        "The most common type of ML.  Every model below sees LABELED wine examples "
        "during training — each wine comes with its quality score already known.")
    print_explain(
        "Imagine teaching a child to recognise dogs by showing them thousands of "
        "pictures labelled 'dog' or 'not dog'.  That's supervised learning!  "
        "Here, the 'pictures' are wine chemistry measurements and the label is "
        "whether the wine scored ≥7 (good) or <7 (not good).")
    mod = _import("methods.supervised_learning")
    t   = time.time()
    res = mod.run_all(X_train, X_test, y_train, y_test,
                      y_train_bin, y_test_bin, feature_names)
    print_info(f"All supervised models completed in {elapsed(t)}")
    return res


def run_unsupervised(X, y):
    print_header("UNSUPERVISED LEARNING", "█")
    print_info(
        "No labels used here at all!  The algorithms explore the wine data "
        "purely based on patterns in the measurements.")
    print_explain(
        "Like a chef who has never seen a recipe book but groups ingredients by "
        "smell and texture alone.  Unsupervised learning finds structure without "
        "being told what to look for.")
    mod = _import("methods.unsupervised_learning")
    t   = time.time()
    res = mod.run_all(X, y)
    print_info(f"All unsupervised methods completed in {elapsed(t)}")
    return res


def run_semi_supervised(X_train, X_test, y_train_bin, y_test_bin):
    print_header("SEMI-SUPERVISED LEARNING", "█")
    print_info("Only 10% of wines have labels — the model must learn from the rest on its own.")
    print_explain(
        "Real-world scenario: you have 6,000 wines but can only afford to pay "
        "experts to rate 600 of them.  Semi-supervised learning uses those 600 "
        "rated wines as a guide to understand all 6,000.")
    mod = _import("methods.semi_supervised_learning")
    t   = time.time()
    res = mod.run_all(X_train, X_test, y_train_bin, y_test_bin)
    print_info(f"All semi-supervised methods completed in {elapsed(t)}")
    return res


def run_reinforcement(X, y_binary):
    print_header("REINFORCEMENT LEARNING", "█")
    print_info(
        "An agent interacts with a 'Wine Tasting Environment':  "
        "sees a wine's measurements → guesses 'good' or 'not good'  "
        "→ gets +1 for correct, -1 for wrong → learns over thousands of rounds.")
    print_explain(
        "No dataset with right answers!  The agent must discover the right strategy "
        "purely from reward signals — like learning to ride a bike by falling off "
        "and adjusting until you stay balanced.  "
        "In production, RL powers game-playing AI (AlphaGo), robotics, and ChatGPT RLHF.")
    mod = _import("methods.reinforcement_learning")
    t   = time.time()
    res = mod.run_all(X, y_binary)
    print_info(f"All RL methods completed in {elapsed(t)}")
    return res


def run_self_supervised(X, y):
    print_header("SELF-SUPERVISED LEARNING", "█")
    print_info(
        "No human labels used during pre-training.  "
        "The model creates its own puzzles from the raw wine measurements and "
        "learns by solving them.  Quality evaluated via a linear probe afterward.")
    print_explain(
        "Inspired by how humans learn language as babies — no one gives us labeled "
        "sentences, we just absorb patterns.  Self-supervised models learn to "
        "understand data by completing it, masking it, and comparing versions of it.")
    mod = _import("methods.self_supervised_learning")
    t   = time.time()
    res = mod.run_all(X, y)
    print_info(f"All self-supervised methods completed in {elapsed(t)}")
    return res


def run_transfer(df, X, y_binary):
    print_header("TRANSFER LEARNING", "█")
    print_info(
        "Source domain: RED wine (1,599 samples).  "
        "Target domain: WHITE wine (4,898 samples).  "
        "Knowledge learned from red wine is reused/adapted for white wine.")
    print_explain(
        "Like learning to drive in one country and then quickly adapting to "
        "driving in a different country with different road rules — most of the "
        "skill transfers, you just need to learn the local differences.")
    mod = _import("methods.transfer_learning")
    t   = time.time()
    res = mod.run_all(df, X, y_binary)
    print_info(f"All transfer learning methods completed in {elapsed(t)}")
    return res


def run_ensemble(X_train, X_test, y_train_bin, y_test_bin):
    print_header("ENSEMBLE METHODS", "█")
    print_info(
        "Multiple models combined for superior accuracy.  "
        "Three strategies: Bagging (↓ variance), Boosting (↓ bias), "
        "and Stacking (heterogeneous meta-learning).")
    print_explain(
        "The wisdom of crowds: a group of diverse models almost always "
        "outperforms the best single model.  "
        "Ensemble methods are the go-to approach whenever accuracy is the top priority.")
    mod = _import("methods.ensemble_methods")
    t   = time.time()
    res = mod.run_all(X_train, X_test, y_train_bin, y_test_bin)
    print_info(f"All ensemble methods completed in {elapsed(t)}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_MAP = {
    "supervised":   "supervised",
    "unsupervised": "unsupervised",
    "semi":         "semi_supervised",
    "rl":           "reinforcement",
    "ssl":          "self_supervised",
    "transfer":     "transfer",
    "ensemble":     "ensemble",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wine Quality – ML Methods Demo")
    parser.add_argument(
        "--skip", nargs="+", choices=list(CATEGORY_MAP.keys()),
        default=[], metavar="CAT",
        help="Categories to skip (space-separated)")
    parser.add_argument(
        "--only", nargs="+", choices=list(CATEGORY_MAP.keys()),
        default=[], metavar="CAT",
        help="Run ONLY these categories")
    return parser.parse_args()


def main():
    args   = parse_args()
    t_main = time.time()

    # ── Banner ────────────────────────────────────────────────────────────────
    print_header(
        "Wine Quality – Complete Machine Learning Methods Demo\n"
        "  Dataset : Wine_Quality_Data.csv  (6,497 samples, 12 features)\n"
        "  Target  : quality score 3-9  →  binary label (good ≥ 7 / not-good < 7)\n"
        "  Goal    : demonstrate all 7 major ML paradigms with plain-English output",
        "█",
    )
    print_explain(
        "Welcome!  This program shows you 7 completely different ways a computer "
        "can learn from wine data.  Each model will explain what it is in simple "
        "terms before running.  At the end, you'll get a guide comparing all of them.  "
        "Look for the ★ (pink/magenta) lines for plain-English explanations, "
        "the ℹ (blue) lines for technical details, and "
        "the ✓ (green) lines for results.  "
        "The coloured bars show accuracy at a glance — longer green bar = better!")

    # ── Load data ─────────────────────────────────────────────────────────────
    print_section("Loading & Preprocessing the Wine Dataset")
    print_explain(
        "Before any model can learn, the data must be prepared.  "
        "We load 6,497 wine samples (red + white), scale the measurements so they "
        "are all on the same numerical scale (important for many models), and split "
        "80% for training and 20% for testing (the test set is never used during training).")

    DATA_PATH = os.path.join(PROJECT_ROOT, "Wine_Quality_Data.csv")
    df, X, y, y_binary, feature_names, scaler = load_wine_data(DATA_PATH)

    print_result("Total wine samples",      f"{len(df):,}")
    print_result("Red wine samples",        f"{(df['color'] == 'red').sum():,}")
    print_result("White wine samples",      f"{(df['color'] == 'white').sum():,}")
    print_result("Features per wine",       len(feature_names))
    print_result("Quality score range",     f"{y.min()} – {y.max()}")
    print_result("'Good' wines (score ≥ 7)",
                 f"{y_binary.sum():,}  ({y_binary.mean()*100:.1f}% of all wines)")
    print_explain(
        f"Only {y_binary.mean()*100:.1f}% of wines score ≥7.  "
        "This is an 'imbalanced' classification problem — the model must learn to "
        "identify the rare 'good' wines without falsely labelling most wines as good.  "
        "That's why we use F1 score (not just accuracy) to measure performance.")

    # Splits
    X_train,   X_test,   y_train,   y_test   = get_train_test(X, y)
    X_train_b, X_test_b, y_train_b, y_test_b = get_train_test(X, y_binary)

    print_result("Training set size", f"{len(X_train):,} wines  (80%)")
    print_result("Test set size",     f"{len(X_test):,}  wines  (20%)")
    print_explain(
        "The test set is locked away and NEVER shown to any model during training.  "
        "This ensures we measure how well models work on wines they've NEVER seen — "
        "just like a real-world deployment!")

    # ── Determine which categories to run ─────────────────────────────────────
    all_cats = list(CATEGORY_MAP.keys())
    if args.only:
        cats = [c for c in all_cats if c in args.only]
    else:
        cats = [c for c in all_cats if c not in args.skip]

    print_info(f"Running categories: {cats}")

    all_results = {}

    # ── 1. Supervised ─────────────────────────────────────────────────────────
    if "supervised" in cats:
        all_results["supervised"] = run_supervised(
            X_train, X_test, y_train, y_test,
            y_train_b, y_test_b, feature_names)

    # ── 2. Unsupervised ───────────────────────────────────────────────────────
    if "unsupervised" in cats:
        all_results["unsupervised"] = run_unsupervised(X, y)

    # ── 3. Semi-supervised ────────────────────────────────────────────────────
    if "semi" in cats:
        all_results["semi_supervised"] = run_semi_supervised(
            X_train_b, X_test_b, y_train_b, y_test_b)

    # ── 4. Reinforcement Learning ─────────────────────────────────────────────
    if "rl" in cats:
        all_results["reinforcement"] = run_reinforcement(X, y_binary)

    # ── 5. Self-supervised ────────────────────────────────────────────────────
    if "ssl" in cats:
        all_results["self_supervised"] = run_self_supervised(X, y)

    # ── 6. Transfer Learning ─────────────────────────────────────────────────
    if "transfer" in cats:
        all_results["transfer"] = run_transfer(df, X, y_binary)

    # ── 7. Ensemble ───────────────────────────────────────────────────────────
    if "ensemble" in cats:
        all_results["ensemble"] = run_ensemble(
            X_train_b, X_test_b, y_train_b, y_test_b)

    # ── Summary histogram ─────────────────────────────────────────────────────
    print_summary_table(all_results)

    # ── Final comparison guide ────────────────────────────────────────────────
    print_final_comparison(all_results)

    print(f"\n{Colors.GREEN}{Colors.BOLD}"
          f"All selected categories completed in {elapsed(t_main)}.{Colors.END}\n")


if __name__ == "__main__":
    main()
