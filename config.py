"""Configuration for the reproducible Titanic training pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "titanic"
RESULTS_DIR = PROJECT_ROOT / "results"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
CV_RESULTS_PATH = RESULTS_DIR / "final_cv_results.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_hard_ensemble.csv"

ID_COLUMN = "PassengerId"
TARGET_COLUMN = "Survived"
NUMERIC_FEATURES = ("Age", "Fare", "SibSp", "Parch")
CATEGORICAL_FEATURES = ("Pclass", "Sex", "Embarked")
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
N_SPLITS = 5
RUN_CROSS_VALIDATION = True

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_leaf": 1,
    "max_features": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

CATBOOST_PARAMS = {
    "iterations": 300,
    "depth": 4,
    "learning_rate": 0.1,
    "l2_leaf_reg": 5,
    "loss_function": "Logloss",
    "random_seed": RANDOM_STATE,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}

XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.1,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
