# Это вынесет из eda.ipynb:
# создание новых признаков;
# список числовых признаков;
# список категориальных признаков;
# общий список признаков модели.

import pandas as pd


NUMERIC_FEATURES = [
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
    "CabinKnown",
]

CATEGORICAL_FEATURES = [
    "Pclass",
    "Sex",
    "Embarked",
    "TitleGrouped",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features used by all project models."""
    data = data.copy()

    data["FamilySize"] = (
        data["SibSp"]
        + data["Parch"]
        + 1
    )

    data["IsAlone"] = (
        data["FamilySize"] == 1
    ).astype(int)

    data["CabinKnown"] = (
        data["Cabin"].notna()
    ).astype(int)

    data["Title"] = data["Name"].str.extract(
        r",\s*([^.]*)\."
    )

    common_titles = [
        "Mr",
        "Miss",
        "Mrs",
        "Master",
    ]

    data["TitleGrouped"] = data["Title"].where(
        data["Title"].isin(common_titles),
        "Rare",
    )

    return data