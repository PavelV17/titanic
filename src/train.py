# создаёт новые признаки;
# разделяет данные на X и y;
# создаёт Soft Voting;
# обучает его на всех train.csv;
# сохраняет модель в:
# results/voting_model.joblib

import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import (
    METRICS_PATH,
    MODEL_PATH,
    N_SPLITS,
    RANDOM_STATE,
    SCORING,
    TARGET_COLUMN,
)
from src.features import MODEL_FEATURES, add_features
from src.models import build_voting_model


def evaluate_model(
    model: VotingClassifier,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Evaluate the final model using stratified cross-validation."""
    cv_splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv_splitter,
        scoring=SCORING,
        n_jobs=1,
    )

    metrics = pd.DataFrame([
        {
            "model": "Soft Voting",
            "scoring": SCORING,
            "cv_folds": N_SPLITS,
            "score_mean": scores.mean(),
            "score_std": scores.std(),
        }
    ])

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    metrics.to_csv(
        METRICS_PATH,
        index=False,
    )

    print("CV scores:", scores)
    print(f"CV mean: {scores.mean():.4f}")
    print(f"CV std: {scores.std():.4f}")
    print(f"Метрики сохранены: {METRICS_PATH}")

    return metrics


def train_model(
    train_data: pd.DataFrame,
) -> VotingClassifier:
    """Evaluate, train and save the final soft-voting model."""
    train_fe = add_features(train_data)

    X = train_fe[MODEL_FEATURES]
    y = train_fe[TARGET_COLUMN]

    model = build_voting_model()

    evaluate_model(
        model=model,
        X=X,
        y=y,
    )

    model.fit(X, y)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    joblib.dump(model, MODEL_PATH)

    print(f"Модель сохранена: {MODEL_PATH}")

    return model