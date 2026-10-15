from pathlib import Path
import sys

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import num_features, cat_features


features = num_features + cat_features

train_data = pd.read_csv(
    PROJECT_ROOT / "titanic" / "train.csv"
)

X = train_data[features].copy()
Y = train_data["Survived"]

X[cat_features] = (
    X[cat_features]
    .fillna("Missing")
    .astype(str)
)


CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

depth_values = [
    4,
    6,
    8,
]

learning_rate_values = [
    0.03,
    0.05,
    0.1,
]

l2_leaf_reg_values = [
    3,
    5,
    10,
]

results_catboost = []

for depth in depth_values:
    for learning_rate in learning_rate_values:
        for l2_leaf_reg in l2_leaf_reg_values:
            model_catboost = CatBoostClassifier(
                iterations=300,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                loss_function="Logloss",
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )

            scores = cross_val_score(
                model_catboost,
                X,
                Y,
                cv=CV,
                scoring="accuracy",
                params={
                    "cat_features": cat_features,
                }
            )

            results_catboost.append({
                "depth": depth,
                "learning_rate": learning_rate,
                "l2_leaf_reg": l2_leaf_reg,
                "mean_accuracy": scores.mean(),
                "std_accuracy": scores.std(),
            })
            
results_catboost_df = pd.DataFrame(
    results_catboost
)

results_catboost_df = (
    results_catboost_df
    .sort_values(
        by=["mean_accuracy", "std_accuracy"],
        ascending=[False, True],
    )
    .reset_index(drop=True)
)

print("Top 10 CatBoost configurations:")

print(
    results_catboost_df
    .head(10)
    .round(4)
    .to_string(index=False)
)

iterations_values = [
    100,
    200,
    300,
    500,
    800,
    1000,
]

results_iterations = []

for iterations in iterations_values:
    model_catboost = CatBoostClassifier(
        iterations=iterations,
        depth=4,
        learning_rate=0.1,
        l2_leaf_reg=5,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    scores = cross_val_score(
        model_catboost,
        X,
        Y,
        cv=CV,
        scoring="accuracy",
        params={
            "cat_features": cat_features,
        },
    )

    results_iterations.append({
        "iterations": iterations,
        "mean_accuracy": scores.mean(),
        "std_accuracy": scores.std(),
    })
    
results_iterations_df = (
    pd.DataFrame(results_iterations)
    .sort_values(
        by=["mean_accuracy", "std_accuracy"],
        ascending=[False, True],
    )
    .reset_index(drop=True)
)

print(
    results_iterations_df
    .round(4)
    .to_string(index=False)
)
