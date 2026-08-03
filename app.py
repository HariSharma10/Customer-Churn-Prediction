from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ==========================
# Load Model Files
# ==========================
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


# ==========================
# Home
# ==========================
@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# About
# ==========================
@app.route("/about")
def about():
    return render_template("about.html")


# ==========================
# Prediction Page
# ==========================
@app.route("/predict")
def predict():
    return render_template("predict.html")


# ==========================
# Result
# ==========================
@app.route("/result", methods=["POST"])
def result():

    try:

        data = {}

        # Read all features
        for col in feature_columns:

            value = request.form[col]

            # Numerical columns
            if col in ["Tenure Months", "Monthly Charges", "Total Charges"]:
                data[col] = float(value)

            # Encode categorical columns
            else:
                data[col] = encoders[col].transform([value])[0]

        # Create dataframe in correct order
        input_df = pd.DataFrame([data])

        input_df = input_df[feature_columns]

        # Scale
        scaled = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(scaled)[0]

        probability = model.predict_proba(scaled)[0].max() * 100

        if prediction == 1:
            result = "Customer is likely to Churn"
            color = "danger"
        else:
            result = "Customer is likely to Stay"
            color = "success"

        return render_template(
            "results.html",
            prediction=result,
            probability=round(probability,2),
            color=color
        )

    except Exception as e:
      return render_template("error.html", error=str(e)), 400


if __name__ == "__main__":
    app.run(debug=True)