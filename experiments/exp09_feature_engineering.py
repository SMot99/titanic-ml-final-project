from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import (
    add_features,
    build_pipeline,
    cat_features,
    num_features,
)


engineered_num_features = (
    num_features
    + [
        "FamilySize",
        "IsAlone",
    ]
)

engineered_cat_features = (
    cat_features
    + [
        "Title",
        "CabinDeck",
    ]
)

train_data = pd.read_csv(
    PROJECT_ROOT / "titanic" / "train.csv"
)

train_data = add_features(train_data)

features = (
    engineered_num_features
    + engineered_cat_features
)

X = train_data[features].copy()
y = train_data["Survived"]

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=1,
    max_features=None,
    random_state=42,
    n_jobs=-1,
)

model_pipeline = build_pipeline(
    model=model,
    scale_numeric=False,
    numeric_features=engineered_num_features,
    categorical_features=engineered_cat_features,
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