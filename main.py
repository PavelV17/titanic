# Он выполнит всю цепочку:
# загрузка данных
#  feature engineering
#  загрузка данных
#  feature engineering
#  кросс-валидация Soft Voting
#  сохранение метрики
#  обучение на всех данных
#  сохранение модели
#  создание submission

from src.data import load_data
from src.inference import create_submission
from src.train import train_model


def main() -> None:
    """Run data loading, cross-validation, training and submission creation."""
    train_data, test_data = load_data()

    print("Train shape:", train_data.shape)
    print("Test shape:", test_data.shape)

    model = train_model(train_data)

    submission = create_submission(
        test_data=test_data,
        model=model,
    )

    print(submission.head())


if __name__ == "__main__":
    main()