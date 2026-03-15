"""
Data Utilities for Wine Quality ML Demo
========================================
Shared data loading, preprocessing, and reporting helpers used by all method modules.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ── Colour codes for terminal output ─────────────────────────────────────────
class Colors:
    HEADER   = "\033[95m"
    BLUE     = "\033[94m"
    CYAN     = "\033[96m"
    GREEN    = "\033[92m"
    YELLOW   = "\033[93m"
    RED      = "\033[91m"
    BOLD     = "\033[1m"
    UNDERLINE = "\033[4m"
    END      = "\033[0m"


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


# ── Metrics helper ────────────────────────────────────────────────────────────

def classification_report_short(y_true, y_pred, model_name: str = "") -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    if model_name:
        print_result(f"{model_name} Accuracy", f"{acc:.4f}")
        print_result(f"{model_name} F1 (weighted)", f"{f1:.4f}")
    return {"accuracy": acc, "f1": f1}


def regression_report_short(y_true, y_pred, model_name: str = "") -> dict:
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    if model_name:
        print_result(f"{model_name} RMSE", f"{rmse:.4f}")
        print_result(f"{model_name} R²",   f"{r2:.4f}")
    return {"rmse": rmse, "r2": r2}
