from pathlib import Path
import sys

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
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

n_neighbors_values = [1, 3, 5, 7, 9, 11, 13, 15]

weights_values = [
    "uniform",
    "distance",
]

metric_values = [
    "euclidean",
    "manhattan",
]


results_knn = []

for n_neighbors in n_neighbors_values:
    for weights in weights_values:
        for metric in metric_values:
            model_knn = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric,
            )

            pipeline_knn = build_pipeline(
                model=model_knn,
                scale_numeric=True,
            )

            scores = cross_val_score(
                pipeline_knn,
                X,
                Y,
                cv=CV,
                scoring="accuracy",
            )

            results_knn.append({
                "n_neighbors": n_neighbors,
                "weights": weights,
                "metric": metric,
                "mean_accuracy": scores.mean(),
                "std_accuracy": scores.std(),
            })
            

results_knn = pd.DataFrame(results_knn)

results_knn = (
    results_knn
    .sort_values(
        by=["mean_accuracy", "std_accuracy"],
        ascending=[False, True],
    )
    .reset_index(drop=True)
)

print(
    results_knn
    .head(10)
    .round(4)
    .to_string(index=False))

