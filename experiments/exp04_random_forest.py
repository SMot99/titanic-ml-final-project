from pathlib import Path
import sys
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))   
from pipeline import build_pipeline, num_features, cat_features
features = num_features + cat_features
train_data = pd.read_csv(PROJECT_ROOT / 'titanic' / 'train.csv')
X = train_data[features]
Y = train_data['Survived']
CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

max_depth_values = [None, 2, 4, 6, 8, 10]
min_samples_leaf_values = [1, 2, 5, 10, 20]
max_features_values = [None, 'sqrt', 'log2']

results_rf = []

for max_depth in max_depth_values:
    for min_samples_leaf in min_samples_leaf_values:
        for max_features in max_features_values:
            model_rf = RandomForestClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
            )

            pipeline_rf = build_pipeline(
                model=model_rf,
                scale_numeric=False,
            )

            scores = cross_val_score(
                pipeline_rf,
                X,
                Y,
                cv=CV,
                scoring="accuracy",
            )
            results_rf.append({
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "mean_accuracy": scores.mean(),
                "std_accuracy": scores.std(),
            })

results_rf_df = pd.DataFrame(results_rf)
results_rf_df = (results_rf_df
    .sort_values(by="mean_accuracy", ascending=False)
    .reset_index(drop=True)
    )

print("Top 5 Random Forest configurations:")
print(results_rf_df.head(5))


n_estimators_values = [100, 300, 500, 1000]

results_n_estimators = []

for n_estimators in n_estimators_values:
    model_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        min_samples_leaf=1,
        max_features=None,
        criterion="gini",
        random_state=42,
        n_jobs=-1,
    )

    pipeline_rf = build_pipeline(
        model=model_rf,
        scale_numeric=False,
    )

    scores = cross_val_score(
        pipeline_rf,
        X,
        Y,
        cv=CV,
        scoring="accuracy",
    )

    results_n_estimators.append({
        "n_estimators": n_estimators,
        "mean_accuracy": scores.mean(),
        "std_accuracy": scores.std(),
    })
    
results_n_estimators_df = pd.DataFrame(results_n_estimators)
results_n_estimators_df = (results_n_estimators_df
    .sort_values(by="mean_accuracy", ascending=False)
    .reset_index(drop=True)
    )
print("Random Forest configurations with different n_estimators:")
print(results_n_estimators_df)