
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        data = X.copy()

        # Extract passenger title
        if "Name" in data.columns:
            titles = data["Name"].fillna("").astype(str).str.extract(
                r",\s*([^.]*)\.", expand=False
            ).str.strip()

            titles = titles.replace({
                "Mlle": "Miss",
                "Ms": "Miss",
                "Mme": "Mrs"
            })

            data["Title"] = titles.where(
                titles.isin(["Mr", "Mrs", "Miss", "Master"]),
                "Rare"
            ).fillna("Mr")

        else:
            data["Title"] = "Mr"

        # Calculate family size
        data["FamilySize"] = (
            pd.to_numeric(data["SibSp"], errors="coerce").fillna(0)
            + pd.to_numeric(data["Parch"], errors="coerce").fillna(0)
            + 1
        )

        # Check if passenger is alone
        data["IsAlone"] = (
            data["FamilySize"] == 1
        ).astype(int)

        # Calculate fare per person
        fare = pd.to_numeric(data["Fare"], errors="coerce")

        data["FarePerPerson"] = (
            fare / data["FamilySize"].replace(0, 1)
        )

        return data
