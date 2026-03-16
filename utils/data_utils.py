"""
Data Utilities for Wine Quality ML Demo
========================================
Shared data loading, preprocessing, and reporting helpers used by all method modules.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Colour codes for terminal output ─────────────────────────────────────────
class Colors:
    HEADER    = "\033[95m"
    BLUE      = "\033[94m"
    CYAN      = "\033[96m"
    GREEN     = "\033[92m"
    YELLOW    = "\033[93m"
    RED       = "\033[91m"
    MAGENTA   = "\033[35m"
    BOLD      = "\033[1m"
    UNDERLINE = "\033[4m"
    END       = "\033[0m"


def print_header(title: str, char: str = "=") -> None:
    width = 70
    print(f"\n{Colors.HEADER}{Colors.BOLD}{char * width}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {title}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{char * width}{Colors.END}")


def print_section(title: str) -> None:
    print(f"\n{Colors.CYAN}{Colors.BOLD}── {title} {'─' * (60 - len(title))}{Colors.END}")


def print_result(label: str, value) -> None:
    print(f"  {Colors.GREEN}✓{Colors.END} {label}: {Colors.YELLOW}{value}{Colors.END}")


def print_info(msg: str) -> None:
    print(f"  {Colors.BLUE}ℹ{Colors.END}  {msg}")


def print_explain(msg: str) -> None:
    """Print a plain-English kid-friendly explanation in a distinct style."""
    print(f"  {Colors.MAGENTA}★{Colors.END}  {Colors.MAGENTA}{msg}{Colors.END}")


def print_score_bar(label: str, value: float, max_val: float = 1.0,
                    bar_width: int = 40) -> None:
    """
    Print a coloured ASCII bar chart for a single score value.

    Example output:
      Accuracy  [████████████████████░░░░░░░░░░░░░░░░░░░░]  80.3%
    """
    if max_val <= 0 or not np.isfinite(value):
        return
    ratio   = max(0.0, min(1.0, value / max_val))
    filled  = int(round(ratio * bar_width))
    empty   = bar_width - filled

    # Colour the bar green/yellow/red based on ratio
    if ratio >= 0.85:
        bar_color = Colors.GREEN
    elif ratio >= 0.70:
        bar_color = Colors.YELLOW
    else:
        bar_color = Colors.RED

    bar  = bar_color + "█" * filled + Colors.END
    bar += Colors.BLUE + "░" * empty + Colors.END
    pct  = f"{value / max_val * 100:.1f}%"
    print(f"  {Colors.BOLD}{label:<28}{Colors.END} [{bar}]  {Colors.YELLOW}{pct}{Colors.END}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_wine_data(filepath: str = "Wine_Quality_Data.csv"):
    """
    Load the Wine Quality dataset.

    Returns
    -------
    df : raw DataFrame
    X  : feature matrix (numpy, scaled)
    y  : quality labels (integer, 3-9)
    y_binary : binary label – 1 if quality >= 7 (good wine), 0 otherwise
    feature_names : list of column names
    scaler : fitted StandardScaler (for inverse transforms later)
    """
    df = pd.read_csv(filepath)

    # Encode wine colour: red=0, white=1
    df["color_enc"] = (df["color"] == "white").astype(int)

    feature_cols = [
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
        "pH", "sulphates", "alcohol", "color_enc",
    ]

    X_raw = df[feature_cols].values.astype(np.float32)
    y     = df["quality"].values.astype(int)

    # Binary target: good (≥7) vs not-good (<7)
    y_binary = (y >= 7).astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    return df, X, y, y_binary, feature_cols, scaler


def get_train_test(X, y, test_size: float = 0.20, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def split_labeled_unlabeled(X_train, y_train, labeled_fraction: float = 0.10,
                            random_state: int = 42):
    """
    Split training data into a small labeled set and a large unlabeled set.
    Used to simulate semi-supervised scenarios.
    """
    rng = np.random.default_rng(random_state)
    n_labeled = max(int(len(y_train) * labeled_fraction), 20)
    idx = rng.permutation(len(y_train))
    labeled_idx   = idx[:n_labeled]
    unlabeled_idx = idx[n_labeled:]
    return (X_train[labeled_idx],   y_train[labeled_idx],
            X_train[unlabeled_idx], y_train[unlabeled_idx])


# ── Metric explanation helpers ─────────────────────────────────────────────────

def _accuracy_explanation(acc: float) -> str:
    right = int(round(acc * 100))
    wrong = 100 - right
    if acc >= 0.90:
        grade = "That's excellent — like scoring 90+ on a test!"
    elif acc >= 0.80:
        grade = "That's pretty good — like scoring a solid B!"
    elif acc >= 0.70:
        grade = "That's okay — passing, but there's room to improve."
    else:
        grade = "That's below average — this model is struggling a bit."
    return (f"Plain English: out of every 100 wines the model tasted, "
            f"it correctly said 'good' or 'not good' for {right} of them "
            f"and got {wrong} wrong.  {grade}")


def _f1_explanation(f1: float) -> str:
    pct = int(round(f1 * 100))
    if f1 >= 0.85:
        quality = "great"
    elif f1 >= 0.70:
        quality = "decent"
    else:
        quality = "fair"
    return (f"Plain English: the F1 score ({pct}/100) checks two things at once — "
            f"(1) Does it catch all the truly good wines? "
            f"(2) Does it avoid falsely calling bad wines good?  "
            f"A {quality} balance of both.")


# ── Metrics reporters ─────────────────────────────────────────────────────────

def classification_report_short(y_true, y_pred, model_name: str = "") -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    if model_name:
        print_score_bar("Accuracy", acc)
        print_result(f"{model_name} Accuracy", f"{acc:.4f}  ({acc*100:.1f}%)")
        print_explain(_accuracy_explanation(acc))
        print_score_bar("F1 Score (weighted)", f1)
        print_result(f"{model_name} F1 Score (weighted)", f"{f1:.4f}  ({f1*100:.1f}%)")
        print_explain(_f1_explanation(f1))
    return {"accuracy": acc, "f1": f1}


def regression_report_short(y_true, y_pred, model_name: str = "") -> dict:
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    if model_name:
        # RMSE bar: scale 0-2 (wine quality range is 3-9, so error > 2 is really bad)
        print_score_bar("Error (lower=better)", max(0, 2.0 - rmse), max_val=2.0)
        print_result(f"{model_name} RMSE", f"{rmse:.4f}")
        print_explain(
            f"Plain English: RMSE = {rmse:.2f}  —  think of this as the average "
            f"'mistake size' in quality score points.  Wine scores go from 3 to 9, "
            f"so being off by {rmse:.1f} point(s) on average is "
            f"{'very impressive!' if rmse < 0.75 else 'a noticeable error.'}")
        r2_display = max(0.0, r2)
        print_score_bar("R² (variance explained)", r2_display)
        print_result(f"{model_name} R² Score", f"{r2:.4f}  ({max(0,r2)*100:.1f}% explained)")
        print_explain(
            f"Plain English: R² = {r2:.3f}  —  this answers 'how much does the model "
            f"understand why wines get different scores?'  "
            f"{'It explains ' + str(int(r2*100)) + '% of why quality varies — like knowing most of the recipe!' if r2 > 0 else 'A negative value means the model is worse than just guessing the average every time!'}")
    return {"rmse": rmse, "r2": r2}
