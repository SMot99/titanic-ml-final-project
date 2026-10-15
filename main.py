"""Train, validate, and run inference for the final Titanic ensemble."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from config import (
    CATBOOST_PARAMS,
    CATEGORICAL_FEATURES,
    CV_RESULTS_PATH,
    FEATURES,
    ID_COLUMN,
    N_SPLITS,
    RANDOM_FOREST_PARAMS,
    RANDOM_STATE,
    RESULTS_DIR,
    RUN_CROSS_VALIDATION,
    SUBMISSION_PATH,
    SUBMISSIONS_DIR,
    TARGET_COLUMN,
    TEST_PATH,
    TRAIN_PATH,
    XGBOOST_PARAMS,
)
from pipeline import build_pipeline


MODEL_NAMES = ("Random Forest", "CatBoost", "XGBoost")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test files and validate their required columns."""
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    required_train = set(FEATURES) | {ID_COLUMN, TARGET_COLUMN}
    required_test = set(FEATURES) | {ID_COLUMN}

    missing_train = required_train.difference(train_data.columns)
    missing_test = required_test.difference(test_data.columns)

    if missing_train:
        raise ValueError(
            f"Train data is missing columns: {sorted(missing_train)}"
        )
    if missing_test:
        raise ValueError(
            f"Test data is missing columns: {sorted(missing_test)}"
        )

    return train_data, test_data


def prepare_catboost_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to the representation CatBoost expects."""
    result = frame.copy()
    categorical_columns = list(CATEGORICAL_FEATURES)
    result[categorical_columns] = (
        result[categorical_columns]
        .fillna("Missing")
        .astype(str)
    )
    return result


def prepare_xgboost_frames(
    *frames: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """Give XGBoost frames one shared categorical vocabulary."""
    lengths = [len(frame) for frame in frames]
    combined = pd.concat(frames, ignore_index=True)

    for column in CATEGORICAL_FEATURES:
        combined[column] = (
            combined[column]
            .fillna("Missing")
            .astype("category")
        )

    prepared_frames = []
    start = 0
    for frame, length in zip(frames, lengths, strict=True):
        prepared = combined.iloc[start:start + length].copy()
        prepared.index = frame.index
        prepared_frames.append(prepared)
        start += length

    return tuple(prepared_frames)


def build_models():
    """Create fresh estimators for one training fold or final fit."""
    random_forest = build_pipeline(
        model=RandomForestClassifier(**RANDOM_FOREST_PARAMS),
        scale_numeric=False,
    )
    catboost = CatBoostClassifier(**CATBOOST_PARAMS)
    xgboost = XGBClassifier(**XGBOOST_PARAMS)
    return random_forest, catboost, xgboost


def predict_component_probabilities(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    inference_features: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Fit all final component models and return positive-class probabilities."""
    random_forest, catboost, xgboost = build_models()

    random_forest.fit(train_features, train_target)
    random_forest_probabilities = random_forest.predict_proba(
        inference_features
    )[:, 1]

    train_catboost = prepare_catboost_frame(train_features)
    inference_catboost = prepare_catboost_frame(inference_features)
    catboost.fit(
        train_catboost,
        train_target,
        cat_features=list(CATEGORICAL_FEATURES),
    )
    catboost_probabilities = catboost.predict_proba(
        inference_catboost
    )[:, 1]

    train_xgboost, inference_xgboost = prepare_xgboost_frames(
        train_features,
        inference_features,
    )
    xgboost.fit(train_xgboost, train_target)
    xgboost_probabilities = xgboost.predict_proba(
        inference_xgboost
    )[:, 1]

    return {
        "Random Forest": random_forest_probabilities,
        "CatBoost": catboost_probabilities,
        "XGBoost": xgboost_probabilities,
    }


def combine_predictions(
    probabilities: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Create component, soft-average, and majority-vote predictions."""
    probability_matrix = np.column_stack(
        [probabilities[name] for name in MODEL_NAMES]
    )
    class_matrix = (probability_matrix >= 0.5).astype(int)

    predictions = {
        name: class_matrix[:, index]
        for index, name in enumerate(MODEL_NAMES)
    }
    predictions["Soft ensemble"] = (
        probability_matrix.mean(axis=1) >= 0.5
    ).astype(int)
    predictions["Hard ensemble"] = (
        class_matrix.sum(axis=1) >= 2
    ).astype(int)
    return predictions


def cross_validate(
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Evaluate every component and ensemble on shared stratified folds."""
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores: dict[str, list[float]] = {
        name: []
        for name in (*MODEL_NAMES, "Soft ensemble", "Hard ensemble")
    }

    for fold, (train_index, validation_index) in enumerate(
        splitter.split(features, target),
        start=1,
    ):
        fold_probabilities = predict_component_probabilities(
            train_features=features.iloc[train_index],
            train_target=target.iloc[train_index],
            inference_features=features.iloc[validation_index],
        )
        fold_predictions = combine_predictions(fold_probabilities)

        for name, predictions in fold_predictions.items():
            scores[name].append(
                accuracy_score(
                    target.iloc[validation_index],
                    predictions,
                )
            )

        print(f"Completed CV fold {fold}/{N_SPLITS}")

    rows = []
    for name, fold_scores in scores.items():
        values = np.asarray(fold_scores)
        rows.append({
            "model": name,
            "mean_accuracy": values.mean(),
            "std_accuracy": values.std(),
            "fold_scores": ",".join(f"{score:.6f}" for score in values),
        })

    return pd.DataFrame(rows)


def create_submission(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """Fit the final ensemble on all training rows and predict test rows."""
    train_features = train_data.loc[:, FEATURES].copy()
    test_features = test_data.loc[:, FEATURES].copy()
    train_target = train_data[TARGET_COLUMN]

    probabilities = predict_component_probabilities(
        train_features=train_features,
        train_target=train_target,
        inference_features=test_features,
    )
    predictions = combine_predictions(probabilities)["Hard ensemble"]

    submission = pd.DataFrame({
        ID_COLUMN: test_data[ID_COLUMN],
        TARGET_COLUMN: predictions,
    })

    if len(submission) != len(test_data):
        raise ValueError("Submission row count does not match test data")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values")
    if not set(submission[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Submission target must contain only 0 and 1")

    return submission


def main() -> None:
    """Run validation, train the final ensemble, and save all outputs."""
    train_data, test_data = load_data()
    train_features = train_data.loc[:, FEATURES].copy()
    train_target = train_data[TARGET_COLUMN]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    if RUN_CROSS_VALIDATION:
        cv_results = cross_validate(train_features, train_target)
        cv_results.to_csv(CV_RESULTS_PATH, index=False)
        print("\nCross-validation results:")
        print(
            cv_results.loc[:, ["model", "mean_accuracy", "std_accuracy"]]
            .round(4)
            .to_string(index=False)
        )
        print(f"Saved CV results to: {CV_RESULTS_PATH}")

    submission = create_submission(train_data, test_data)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"Saved submission to: {SUBMISSION_PATH}")
    print("Prediction counts:")
    print(submission[TARGET_COLUMN].value_counts().sort_index())


if __name__ == "__main__":
    main()
