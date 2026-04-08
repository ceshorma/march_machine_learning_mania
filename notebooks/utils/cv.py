"""Temporal cross-validation for NCAA tournament prediction.

Unlike standard K-fold CV, this uses expanding-window validation where:
- Training: all seasons before year Y
- Validation: season Y's tournament games

This prevents data leakage and mimics the actual prediction task.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def temporal_tournament_cv(
    df: pd.DataFrame,
    season_col: str = "Season",
    min_train_seasons: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window CV splits by tournament season.

    Args:
        df: Training DataFrame (must have a Season column).
        season_col: Name of the season column.
        min_train_seasons: Minimum number of training seasons before first validation.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    seasons = sorted(df[season_col].unique())
    folds = []

    for i in range(min_train_seasons, len(seasons)):
        train_seasons = set(seasons[:i])
        val_season = seasons[i]

        train_idx = df.index[df[season_col].isin(train_seasons)].values
        val_idx = df.index[df[season_col] == val_season].values

        if len(val_idx) > 0 and len(train_idx) > 0:
            folds.append((train_idx, val_idx))

    return folds


def evaluate_temporal_cv(
    model_fn,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    seasons: pd.Series,
    min_train_seasons: int = 5,
    clip_range: tuple[float, float] = (0.05, 0.95),
) -> dict:
    """Run temporal CV and return evaluation results.

    Args:
        model_fn: Callable that takes (X_train, y_train) and returns a fitted model
                  with predict_proba method.
        X: Feature matrix.
        y: Target array.
        seasons: Season series (same length as X).
        min_train_seasons: Min training seasons.
        clip_range: Prediction clipping range for log loss.

    Returns:
        dict with fold_scores, mean_score, std_score, oof_predictions, fold_details.
    """
    # Build a temp DataFrame to use for splitting
    temp_df = pd.DataFrame({"Season": seasons.values}).reset_index(drop=True)
    folds = temporal_tournament_cv(temp_df, min_train_seasons=min_train_seasons)

    X_arr = np.array(X) if isinstance(X, pd.DataFrame) else X
    y_arr = np.array(y)

    fold_scores = []
    oof = np.full(len(y_arr), np.nan)
    fold_details = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        val_season = int(seasons.iloc[val_idx[0]])

        # Train model
        model = model_fn(X_train, y_train)

        # Predict
        if hasattr(model, "predict_proba"):
            val_pred = model.predict_proba(X_val)[:, 1]
        else:
            val_pred = model.predict(X_val)

        # Clip predictions
        val_pred = np.clip(val_pred, clip_range[0], clip_range[1])

        # Evaluate
        score = log_loss(y_val, val_pred)
        fold_scores.append(score)
        oof[val_idx] = val_pred

        fold_details.append({
            "fold": fold_idx,
            "season": val_season,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "log_loss": score,
        })

        print(
            f"  Fold {fold_idx + 1:2d} | Season {val_season} | "
            f"Train: {len(train_idx):5d} | Val: {len(val_idx):3d} | "
            f"LogLoss: {score:.5f}"
        )

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"\n  CV LogLoss: {mean_score:.5f} (+/- {std_score:.5f})")

    return {
        "fold_scores": fold_scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "oof_predictions": oof,
        "fold_details": fold_details,
    }


def multi_seed_cv(
    model_class,
    model_params: dict,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    seasons: pd.Series,
    seeds: list[int] | None = None,
    min_train_seasons: int = 5,
    clip_range: tuple[float, float] = (0.05, 0.95),
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Multi-seed temporal CV averaging.

    Trains with multiple random seeds, averages OOF and test predictions.

    Returns:
        (oof_avg, test_avg, cv_scores_per_seed)
    """
    import copy

    if seeds is None:
        seeds = [42, 2024, 2025, 1234, 5678]

    temp_df = pd.DataFrame({"Season": seasons.values}).reset_index(drop=True)
    folds = temporal_tournament_cv(temp_df, min_train_seasons=min_train_seasons)

    X_arr = np.array(X) if isinstance(X, pd.DataFrame) else X
    y_arr = np.array(y)
    X_test_arr = np.array(X_test) if isinstance(X_test, pd.DataFrame) else X_test

    all_oof = []
    all_test = []
    all_scores = []

    for seed in seeds:
        params = copy.deepcopy(model_params)
        # Set seed depending on model type
        if "random_state" in params:
            params["random_state"] = seed
        elif "random_seed" in params:
            params["random_seed"] = seed

        oof = np.full(len(y_arr), np.nan)
        test_preds = np.zeros(len(X_test_arr))
        fold_scores = []

        for train_idx, val_idx in folds:
            model = model_class(**params)
            model.fit(X_arr[train_idx], y_arr[train_idx])

            if hasattr(model, "predict_proba"):
                val_pred = model.predict_proba(X_arr[val_idx])[:, 1]
                test_pred = model.predict_proba(X_test_arr)[:, 1]
            else:
                val_pred = model.predict(X_arr[val_idx])
                test_pred = model.predict(X_test_arr)

            val_pred = np.clip(val_pred, clip_range[0], clip_range[1])
            test_pred = np.clip(test_pred, clip_range[0], clip_range[1])

            oof[val_idx] = val_pred
            test_preds += test_pred / len(folds)
            fold_scores.append(log_loss(y_arr[val_idx], val_pred))

        valid_mask = ~np.isnan(oof)
        score = log_loss(y_arr[valid_mask], oof[valid_mask])
        all_scores.append(score)
        all_oof.append(oof)
        all_test.append(test_preds)

        print(f"  Seed {seed:5d} | CV LogLoss: {score:.5f}")

    oof_avg = np.nanmean(all_oof, axis=0)
    test_avg = np.mean(all_test, axis=0)

    valid_mask = ~np.isnan(oof_avg)
    avg_score = log_loss(y_arr[valid_mask], oof_avg[valid_mask])
    print(f"\n  Multi-seed avg LogLoss: {avg_score:.5f}")

    return oof_avg, test_avg, all_scores
