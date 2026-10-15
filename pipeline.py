"""Shared feature engineering and sklearn preprocessing utilities."""

from collections.abc import Sequence

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


# Lowercase aliases are kept for compatibility with the experiment scripts.
num_features = list(NUMERIC_FEATURES)
cat_features = list(CATEGORICAL_FEATURES)


def add_features(data):
    """Return a copy of a Titanic frame with the tested engineered features."""
    result = data.copy()

    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
    result["IsAlone"] = (result["FamilySize"] == 1).astype(int)

    result["Title"] = (
        result["Name"]
        .str.extract(r",\s*([^.]*)\.", expand=False)
        .str.strip()
        .replace({
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
        })
    )

    common_titles = {"Mr", "Miss", "Mrs", "Master"}
    result["Title"] = result["Title"].where(
        result["Title"].isin(common_titles),
        "Rare",
    )

    valid_decks = set("ABCDEFG")
    cabin_deck = result["Cabin"].str[0]
    result["CabinDeck"] = cabin_deck.where(
        cabin_deck.isin(valid_decks),
        "Unknown",
    )

    return result


def build_preprocessor(
    scale_numeric: bool = True,
    dense_output: bool = False,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Build the shared numeric and categorical preprocessing graph."""
    numeric_features = list(numeric_features or num_features)
    categorical_features = list(categorical_features or cat_features)

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense_output,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_features,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )


def build_pipeline(
    model: BaseEstimator,
    scale_numeric: bool,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> Pipeline:
    """Attach a model to the shared preprocessing graph."""
    preprocessor = build_preprocessor(
        scale_numeric=scale_numeric,
        dense_output=False,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
