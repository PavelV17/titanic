import pandas as pd

from src import TEST_PATH, TRAIN_PATH


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Titanic train and test datasets from configured paths."""
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    return train_data, test_data