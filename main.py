#!/usr/bin/env python3
"""
Wine Quality ML Demo – Main Orchestrator
==========================================
Demonstrates every major machine-learning paradigm using the UCI Wine Quality
dataset (6 497 red + white wine samples, 11 physicochemical features, quality
scores 3-9).

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
    print_header, print_section, print_result, print_info,
    Colors,
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
    """Print a compact cross-category accuracy summary."""
    print_header("FINAL SUMMARY", "=")

    rows = []
    for category, res in all_results.items():
        if isinstance(res, dict):
            for name, metrics in res.items():
                if isinstance(metrics, dict) and "accuracy" in metrics:
                    rows.append((category, name, metrics["accuracy"]))
                elif isinstance(metrics, (int, float)):
                    # RL avg rewards stored directly
                    rows.append((category, name, metrics))
                elif isinstance(metrics, tuple) and len(metrics) == 2:
                    # (losses_list, accuracy) from SSL
                    if isinstance(metrics[1], float):
                        rows.append((category, name, metrics[1]))

    if not rows:
        print("  (no numeric metrics collected)")
        return

    rows.sort(key=lambda r: -r[2])
    print(f"\n  {'Category':<22} {'Model':<32} {'Score':>8}")
    print(f"  {'-'*22} {'-'*32} {'-'*8}")
    for cat, name, score in rows:
        bar_len = int(score * 20) if 0 <= score <= 1 else 0
        bar     = "█" * bar_len + "░" * (20 - bar_len) if 0 <= score <= 1 else ""
        val     = f"{score:.4f}" if abs(score) <= 1 else f"{score:+.3f}"
        print(f"  {cat:<22} {name:<32} {val:>8}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Category runners
# ─────────────────────────────────────────────────────────────────────────────

def run_supervised(X_train, X_test, y_train, y_test,
                   y_train_bin, y_test_bin, feature_names):
    print_header("SUPERVISED LEARNING", "█")
    print_info(
        "Learns a mapping f: X → y from labeled training examples.\n"
        "  The wine quality score provides rich supervision for both\n"
        "  regression (predict score) and classification (good vs not-good).")
    mod = _import("methods.supervised_learning")
    t   = time.time()
    res = mod.run_all(X_train, X_test, y_train, y_test,
                      y_train_bin, y_test_bin, feature_names)
    print_info(f"Completed in {elapsed(t)}")
    return res


def run_unsupervised(X, y):
    print_header("UNSUPERVISED LEARNING", "█")
    print_info(
        "Discovers hidden structure (clusters, manifolds, distributions)\n"
        "  without any labels – pure data-driven exploration.")
    mod = _import("methods.unsupervised_learning")
    t   = time.time()
    res = mod.run_all(X, y)
    print_info(f"Completed in {elapsed(t)}")
    return res


def run_semi_supervised(X_train, X_test, y_train_bin, y_test_bin):
    print_header("SEMI-SUPERVISED LEARNING", "█")
    print_info(
        "Exploits abundant unlabeled data alongside a small labeled seed.\n"
        "  Simulated here by using only 10 % of training labels.")
    mod = _import("methods.semi_supervised_learning")
    t   = time.time()
    res = mod.run_all(X_train, X_test, y_train_bin, y_test_bin)
    print_info(f"Completed in {elapsed(t)}")
    return res


def run_reinforcement(X, y_binary):
    print_header("REINFORCEMENT LEARNING", "█")
    print_info(
        "An agent interacts with a 'Wine Classification' environment:\n"
        "  observes a wine's features, guesses 'good' or 'not-good',\n"
        "  receives +1 reward for correct and -1 for wrong.\n"
        "  No labeled training set – reward is the only signal.")
    mod = _import("methods.reinforcement_learning")
    t   = time.time()
    res = mod.run_all(X, y_binary)
    print_info(f"Completed in {elapsed(t)}")
    return res


def run_self_supervised(X, y):
    print_header("SELF-SUPERVISED LEARNING", "█")
    print_info(
        "No human labels used during pre-training.\n"
        "  The model generates its own training signal from the raw features.\n"
        "  Downstream quality evaluated via a linear probe after pre-training.")
    mod = _import("methods.self_supervised_learning")
    t   = time.time()
    res = mod.run_all(X, y)
    print_info(f"Completed in {elapsed(t)}")
    return res


def run_transfer(df, X, y_binary):
    print_header("TRANSFER LEARNING", "█")
    print_info(
        "Source domain: red wine (~1 599 samples).\n"
        "  Target domain: white wine (~4 898 samples).\n"
        "  Pre-trained knowledge from red wine is adapted to white wine.")
    mod = _import("methods.transfer_learning")
    t   = time.time()
    res = mod.run_all(df, X, y_binary)
    print_info(f"Completed in {elapsed(t)}")
    return res


def run_ensemble(X_train, X_test, y_train_bin, y_test_bin):
    print_header("ENSEMBLE METHODS", "█")
    print_info(
        "Multiple models combined for superior accuracy and robustness.\n"
        "  Three strategies: Bagging (↓ variance), Boosting (↓ bias),\n"
        "  and Stacking (heterogeneous meta-learning).")
    mod = _import("methods.ensemble_methods")
    t   = time.time()
    res = mod.run_all(X_train, X_test, y_train_bin, y_test_bin)
    print_info(f"Completed in {elapsed(t)}")
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
        "  Dataset : Wine_Quality_Data.csv  (6 497 samples, 12 features)\n"
        "  Target  : quality score 3-9 → binary (good ≥ 7 / not-good)\n"
        "  Author  : claude/ml-wine-quality-demo",
        "█",
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    print_section("Loading & Preprocessing Data")
    DATA_PATH = os.path.join(PROJECT_ROOT, "Wine_Quality_Data.csv")
    df, X, y, y_binary, feature_names, scaler = load_wine_data(DATA_PATH)

    print_result("Total samples",      len(df))
    print_result("Red wine",           (df["color"] == "red").sum())
    print_result("White wine",         (df["color"] == "white").sum())
    print_result("Features",           len(feature_names))
    print_result("Quality range",      f"{y.min()} – {y.max()}")
    print_result("Good wine (≥7)",     f"{y_binary.sum()} "
                                        f"({y_binary.mean()*100:.1f}%)")

    # Splits
    X_train,   X_test,   y_train,   y_test   = get_train_test(X, y)
    X_train_b, X_test_b, y_train_b, y_test_b = get_train_test(X, y_binary)

    print_result("Train size", len(X_train))
    print_result("Test size",  len(X_test))

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

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary_table(all_results)
    print(f"\n{Colors.GREEN}{Colors.BOLD}"
          f"All selected categories completed in {elapsed(t_main)}.{Colors.END}\n")


if __name__ == "__main__":
    main()
