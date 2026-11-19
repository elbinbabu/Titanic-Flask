
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import from the separate Python file
from feature_engineering import TitanicFeatureEngineer


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "train.csv"
MODEL_PATH = BASE_DIR / "titanic_model.pkl"


NUMERIC_COLS = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "FarePerPerson"
]

CATEGORICAL_COLS = [
    "Sex",
    "Embarked",
    "Pclass",
    "Title",
    "IsAlone"
]


def build_pipeline():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS)
    ])

    model = Pipeline([
        ("feature_engineer", TitanicFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ])

    return model


def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError("train.csv not found!")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Survived"])
    y = df["Survived"].astype(int)

    model = build_pipeline()

    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

    print("Model trained successfully!")
    print("Model saved as titanic_model.pkl")


if __name__ == "__main__":
    main()
