from pathlib import Path
import sys

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
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

# Единое представление категорий для CatBoost и XGBoost.
# Random Forest обработает их внутри своего pipeline.
for column in cat_features:
    X[column] = (
        X[column]
        .fillna("Missing")
        .astype("category")
    )

random_forest = build_pipeline(
    model=RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=1,
        max_features=None,
        random_state=42,
        n_jobs=-1,
    ),
    scale_numeric=False,
)

catboost = CatBoostClassifier(
    iterations=300,
    depth=4,
    learning_rate=0.1,
    l2_leaf_reg=5,
    loss_function="Logloss",
    cat_features=tuple(cat_features),
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
    thread_count=-1,
)

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

INNER_CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123,
)

OUTER_CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

stacking_model = StackingClassifier(
    estimators=[
        ("random_forest", random_forest),
        ("catboost", catboost),
        ("xgboost", xgboost),
    ],
    final_estimator=LogisticRegression(
        max_iter=1000,
    ),
    stack_method="predict_proba",
    cv=INNER_CV,
)

scores = cross_val_score(
    stacking_model,
    X,
    y,
    cv=OUTER_CV,
    scoring="accuracy",
)

print(f"Fold scores: {scores}")
print(f"Mean CV accuracy: {scores.mean():.4f}")
print(f"CV std: {scores.std():.4f}")