# Этот файл:
# загружает сохранённую модель;
# создаёт признаки для test.csv;
# получает прогноз;
# сохраняет voting_submission.csv."

import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier

from src import MODEL_PATH, SUBMISSION_PATH
from src.features import MODEL_FEATURES, add_features


def load_model() -> VotingClassifier:
    """Load the trained model from the configured path."""
    return joblib.load(MODEL_PATH)


def create_submission(
    test_data: pd.DataFrame,
    model: VotingClassifier | None = None,
) -> pd.DataFrame:
    """Generate predictions and save a Kaggle submission file."""
    if model is None:
        model = load_model()

    test_fe = add_features(test_data)
    X_test = test_fe[MODEL_FEATURES]

    predictions = model.predict(X_test).astype(int)

    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions,
    })

    SUBMISSION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        SUBMISSION_PATH,
        index=False,
    )

    print(f"Submission сохранён: {SUBMISSION_PATH}")
    print(f"Размер submission: {submission.shape}")

    return submission