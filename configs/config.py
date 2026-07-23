from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_PATH = RESULTS_DIR / "voting_model.joblib"
METRICS_PATH = RESULTS_DIR / "final_model_metrics.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "voting_submission.csv"

TARGET_COLUMN = "Survived"

RANDOM_STATE = 42
N_SPLITS = 5
SCORING = "accuracy"

FINAL_MODEL = "soft_voting"


LOGISTIC_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}

XGBOOST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}