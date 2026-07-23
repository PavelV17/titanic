# Файл создаёт финальный Soft Voting, который показал
# лучшую среднюю accuracy.

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src import (
    LIGHTGBM_PARAMS,
    LOGISTIC_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
)
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing pipelines for numeric and categorical features."""
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ])


def build_pipeline(classifier) -> Pipeline:
    """Combine preprocessing and a classifier into one sklearn pipeline."""
    return Pipeline([
        (
            "preprocessor",
            build_preprocessor(),
        ),
        (
            "classifier",
            classifier,
        ),
    ])


def build_voting_model() -> VotingClassifier:
    """Build the final soft-voting ensemble."""
    logistic_model = build_pipeline(
        LogisticRegression(**LOGISTIC_PARAMS)
    )

    random_forest_model = build_pipeline(
        RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    )

    lightgbm_model = build_pipeline(
        LGBMClassifier(**LIGHTGBM_PARAMS)
    )

    xgboost_model = build_pipeline(
        XGBClassifier(**XGBOOST_PARAMS)
    )

    return VotingClassifier(
        estimators=[
            ("logistic", logistic_model),
            ("random_forest", random_forest_model),
            ("lightgbm", lightgbm_model),
            ("xgboost", xgboost_model),
        ],
        voting="soft",
    )