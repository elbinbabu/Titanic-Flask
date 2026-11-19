import __main__
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

from train_model import TitanicFeatureEngineer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "titanic_model.pkl"

app = Flask(__name__)
setattr(__main__, "TitanicFeatureEngineer", TitanicFeatureEngineer)
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None
    form_data = {
        "name": "John Smith, Mr.",
        "pclass": "3",
        "sex": "male",
        "age": "25",
        "sibsp": "0",
        "parch": "0",
        "fare": "15",
        "embarked": "S"
    }

    if request.method == "POST":
        form_data.update(request.form.to_dict())

        if model is None:
            error = "The model file is missing. Run train_model.py first."
        else:
            try:
                passenger = pd.DataFrame([{
                    "PassengerId": 0,
                    "Pclass": int(request.form["pclass"]),
                    "Name": request.form.get("name", "Passenger, Mr."),
                    "Sex": request.form["sex"],
                    "Age": float(request.form["age"]),
                    "SibSp": int(request.form["sibsp"]),
                    "Parch": int(request.form["parch"]),
                    "Fare": float(request.form["fare"]),
                    "Ticket": "WEB-INPUT",
                    "Cabin": None,
                    "Embarked": request.form["embarked"]
                }])

                result = int(model.predict(passenger)[0])
                prediction = "Survived" if result == 1 else "Did not survive"

                if hasattr(model, "predict_proba"):
                    probability = round(float(model.predict_proba(passenger)[0][1]) * 100, 1)

            except (ValueError, KeyError) as exc:
                error = f"Please check the passenger details. {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)