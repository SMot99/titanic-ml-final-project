from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier


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

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

voting_model = VotingClassifier(
    estimators=[
        (
            "logreg",
            LogisticRegression(
                C=0.1,
                max_iter=1000,
            ),
        ),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=7,
                weights="uniform",
                metric="manhattan",
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=1,
                max_features=None,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ],
    voting="soft",
    weights=[1, 1, 2],
)

model_pipeline = build_pipeline(
    model=voting_model,
    scale_numeric=True,
)

scores = cross_val_score(
    model_pipeline,
    X,
    y,
    cv=CV,
    scoring="accuracy",
)

print(f"Fold scores: {scores}")
print(f"Mean CV accuracy: {scores.mean():.4f}")
print(f"CV std: {scores.std():.4f}")