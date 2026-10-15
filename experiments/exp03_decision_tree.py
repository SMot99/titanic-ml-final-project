import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import build_pipeline, num_features, cat_features
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

features = num_features + cat_features
train_data = pd.read_csv(PROJECT_ROOT / 'titanic' / 'train.csv')
X = train_data[features]
Y = train_data['Survived']
CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

max_depth_values = [None, 2, 3, 4, 5, 6, 7, 8, 9, 10]

min_samples_leaf_values = [1, 2, 5, 10, 20]

criterion_values = [
    "gini",
    "entropy",
]
results_tree = []

for max_depth in max_depth_values:
    for min_samples_leaf in min_samples_leaf_values:
        for criterion in criterion_values:
            model_tree = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=42,
            )

            pipeline_tree = build_pipeline(
                model=model_tree,
                scale_numeric=False,
            )

            scores = cross_val_score(
                pipeline_tree,
                X,
                Y,
                cv=CV,
                scoring="accuracy",
            )
            results_tree.append({
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "criterion": criterion,
                "mean_accuracy": scores.mean(),
                "std_accuracy": scores.std(),
            })
            

results_tree_df = pd.DataFrame(results_tree)

results_tree_df = (
    results_tree_df
    .sort_values(by="mean_accuracy", ascending=False)
    .reset_index(drop=True)
)

print(results_tree_df)

