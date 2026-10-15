from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import num_features, cat_features


features = num_features + cat_features

train_data = pd.read_csv(
    PROJECT_ROOT / "titanic" / "train.csv"
)

X = train_data[features].copy()
Y = train_data["Survived"]

for column in cat_features:
    X[column] = (
        X[column]
        .fillna("Missing")
        .astype("category")
    )
    

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

max_depth_values = [
    3,
    5,
    7,
]

learning_rate_values = [
    0.03,
    0.05,
    0.1,
]

min_child_weight_values = [
    1,
    3,
    5,
]

results_xgboost = []

for max_depth in max_depth_values:
    for learning_rate in learning_rate_values:
        for min_child_weight in min_child_weight_values:
            model_xgboost = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=300,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_child_weight=min_child_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                tree_method="hist",
                enable_categorical=True,
                random_state=42,
                n_jobs=-1,
            )

            scores = cross_val_score(
                model_xgboost,
                X,
                Y,
                cv=CV,
                scoring="accuracy",
            )

            results_xgboost.append({
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "min_child_weight": min_child_weight,
                "mean_accuracy": scores.mean(),
                "std_accuracy": scores.std(),
            })

results_xgboost_df = (
    pd.DataFrame(results_xgboost)
    .sort_values(
        by=["mean_accuracy", "std_accuracy"],
        ascending=[False, True],
    )
    .reset_index(drop=True)
)

print("Top 10 XGBoost configurations:")

print(
    results_xgboost_df
    .head(5)
)

n_estimators_values = [
    100,
    200,
    300,
    500,
    800,
    1000,
]

results_iterations = []

for n_estimators in n_estimators_values:
    model_xgboost = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
    )

    scores = cross_val_score(
        model_xgboost,
        X,
        Y,
        cv=CV,
        scoring="accuracy",
    )

    results_iterations.append({
        "n_estimators": n_estimators,
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



reg_alpha_values = [
    0.0,
    0.1,
    0.5,
    1.0,
]

reg_lambda_values = [
    0.5,
    1.0,
    5.0,
    10.0,
]

results_regularization = []

for reg_alpha in reg_alpha_values:
    for reg_lambda in reg_lambda_values:
        model_xgboost = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            tree_method="hist",
            enable_categorical=True,
            random_state=42,
            n_jobs=-1,
        )

        scores = cross_val_score(
            model_xgboost,
            X,
            Y,
            cv=CV,
            scoring="accuracy",
        )

        results_regularization.append({
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std(),
        })
        
        
results_regularization_df = (
    pd.DataFrame(results_regularization)
    .sort_values(
        by=["mean_accuracy", "std_accuracy"],
        ascending=[False, True],
    )
    .reset_index(drop=True)
)

print(
    results_regularization_df
    .round(4)
    .to_string(index=False)
)