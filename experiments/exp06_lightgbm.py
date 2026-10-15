from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)   
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
        .astype('category')
    )

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
max_depth_values = [3, 5, 7, 9]
learning_rate_values = [0.01, 0.05, 0.1]
num_leaves_values = [7, 15, 31]

results_lgbm = []
for max_depth in max_depth_values:
    for learning_rate in learning_rate_values:
        for num_leaves in num_leaves_values:
            model_lightgbm = LGBMClassifier(
                objective="binary",
                n_estimators=300,
                max_depth=max_depth,
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                min_child_samples=20,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
            scores = cross_val_score(
                model_lightgbm,
                X,
                Y,
                cv=CV,
                scoring="accuracy",
            )
            results_lgbm.append({
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
                "mean_accuracy": scores.mean(),
                "std_accuracy": scores.std(),
            })
            

results_lgbm_df = pd.DataFrame(results_lgbm)
results_lgbm_df = (results_lgbm_df
    .sort_values(by="mean_accuracy", ascending=False)
    .reset_index(drop=True)
    )
print("Top 5 LightGBM configurations:")
print(results_lgbm_df.head(5))

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
    model_lightgbm = LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        max_depth=9,
        num_leaves=15,
        learning_rate=0.05,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    scores = cross_val_score(
        model_lightgbm,
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

