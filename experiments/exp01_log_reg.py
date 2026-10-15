from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline import build_pipeline, num_features, cat_features

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


feautures = num_features + cat_features

train_data = pd.read_csv(PROJECT_ROOT / 'titanic' / 'train.csv')

X = train_data[feautures]
Y = train_data['Survived']
CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
c_values = [0.01, 0.1, 1, 10, 100]

#смотрим Ridge

model_L2 = LogisticRegression(
    max_iter=1000,
)
model_pipeline_L2 = build_pipeline(
    model=model_L2,
    scale_numeric=True,
)


results_L2 = []

for c in c_values:
    candidate_model = clone(model_pipeline_L2)
    candidate_model.set_params(
        model__C=c,
    )
    
    scores = cross_val_score(
        candidate_model,
        X,
        Y,
        cv=CV,
        scoring='accuracy',
    )
    results_L2.append({
        'C': c,
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std(),
    })
    

results_L2 = pd.DataFrame(results_L2)

results_L2 = (
    results_L2
    .sort_values(
        by="mean_accuracy",
        ascending=False,
    )
    .reset_index(drop=True)
)
print(
    results_L2.round(4).to_string(index=False)
)
    
    
 # смотрим Lasso   

model_L1 = LogisticRegression(
    max_iter=1000,
    l1_ratio=1.0,
    solver='liblinear'
    
)

model_pipeline_L1 = build_pipeline(
    model=model_L1,
    scale_numeric=True,
)

results_L1 = []

for c in c_values:
    candidate_model = clone(model_pipeline_L1)
    candidate_model.set_params(
        model__C=c,
    )
    
    scores = cross_val_score(
        candidate_model,
        X,
        Y,
        cv=CV,
        scoring='accuracy',
    )
    results_L1.append({
        'C': c,
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std(),
    })
   
results_L1 = pd.DataFrame(results_L1)
results_L1 = (
    results_L1
    .sort_values(
        by="mean_accuracy",
        ascending=False,
    )
    .reset_index(drop=True)
)
print(
    results_L1.round(4).to_string(index=False)
)

#смотрим ElasticNet

l1_ratios = [
    0.25,
    0.5,
    0.75,
]

model_elasticnet = LogisticRegression(
    solver="saga",
    max_iter=5000,
    random_state=42,
)

pipeline_elasticnet = build_pipeline(
    model=model_elasticnet,
    scale_numeric=True,
)

results_elasticnet = []

for l1_ratio in l1_ratios:
    for c in c_values:
        candidate_model = clone(pipeline_elasticnet)

        candidate_model.set_params(
            model__C=c,
            model__l1_ratio=l1_ratio,
        )

        scores = cross_val_score(
            candidate_model,
            X,
            Y,
            cv=CV,
            scoring="accuracy",
        )

        results_elasticnet.append({
            "l1_ratio": l1_ratio,
            "C": c,
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std(),
        })
        
results_elasticnet = pd.DataFrame(
    results_elasticnet
)

results_elasticnet = (
    results_elasticnet
    .sort_values(
        by="mean_accuracy",
        ascending=False,
    )
    .reset_index(drop=True)
)

print(
    results_elasticnet.round(4).to_string(index=False)
)