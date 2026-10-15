from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import (
    build_pipeline,
    cat_features,
    num_features,
)


features = num_features + cat_features

train_data = pd.read_csv(
    PROJECT_ROOT / "titanic" / "train.csv"
)

X = train_data[features].copy()
y = train_data["Survived"]

# Версия данных для CatBoost
X_catboost = X.copy()
X_catboost[cat_features] = (
    X_catboost[cat_features]
    .fillna("Missing")
    .astype(str)
)

# Версия данных для XGBoost
X_xgboost = X.copy()

for column in cat_features:
    X_xgboost[column] = (
        X_xgboost[column]
        .fillna("Missing")
        .astype("category")
    )

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

results = {
    "Random Forest": [],
    "CatBoost": [],
    "XGBoost": [],
    "Soft ensemble": [],
    "Hard ensemble": [],
}

for fold, (train_idx, val_idx) in enumerate(
    CV.split(X, y),
    start=1,
):
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # Random Forest
    random_forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=1,
        max_features=None,
        random_state=42,
        n_jobs=-1,
    )

    random_forest_pipeline = build_pipeline(
        model=random_forest,
        scale_numeric=False,
    )

    random_forest_pipeline.fit(
        X.iloc[train_idx],
        y_train,
    )

    rf_probabilities = (
        random_forest_pipeline
        .predict_proba(X.iloc[val_idx])[:, 1]
    )

    # CatBoost
    catboost = CatBoostClassifier(
        iterations=300,
        depth=4,
        learning_rate=0.1,
        l2_leaf_reg=5,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    catboost.fit(
        X_catboost.iloc[train_idx],
        y_train,
        cat_features=cat_features,
    )

    catboost_probabilities = catboost.predict_proba(
        X_catboost.iloc[val_idx]
    )[:, 1]

    # XGBoost
    xgboost = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
    )

    xgboost.fit(
        X_xgboost.iloc[train_idx],
        y_train,
    )

    xgboost_probabilities = xgboost.predict_proba(
        X_xgboost.iloc[val_idx]
    )[:, 1]

    probability_matrix = np.column_stack([
        rf_probabilities,
        catboost_probabilities,
        xgboost_probabilities,
    ])

    soft_predictions = (
        probability_matrix.mean(axis=1) >= 0.5
    ).astype(int)

    class_matrix = (
        probability_matrix >= 0.5
    ).astype(int)

    hard_predictions = (
        class_matrix.sum(axis=1) >= 2
    ).astype(int)

    predictions = {
        "Random Forest": class_matrix[:, 0],
        "CatBoost": class_matrix[:, 1],
        "XGBoost": class_matrix[:, 2],
        "Soft ensemble": soft_predictions,
        "Hard ensemble": hard_predictions,
    }

    for name, fold_predictions in predictions.items():
        score = accuracy_score(
            y_val,
            fold_predictions,
        )
        results[name].append(score)

    print(f"Fold {fold} complete")

for name, scores in results.items():
    scores = np.array(scores)

    print(
        f"{name}: "
        f"mean={scores.mean():.4f}, "
        f"std={scores.std():.4f}, "
        f"folds={scores}"
    )