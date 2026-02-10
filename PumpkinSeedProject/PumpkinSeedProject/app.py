import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model, scaler and encoder
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/check")
def check():
    return render_template("check.html")



@app.route('/predict', methods=["POST"])
def predict():
    # Get values from form
    Area = float(request.form["Area"])
    Perimeter = float(request.form["Perimeter"])
    Major_Axis_Length = float(request.form["Major_Axis_Length"])
    Minor_Axis_Length = float(request.form["Minor_Axis_Length"])
    Convex_Area = float(request.form["Convex_Area"])
    Equiv_Diameter = float(request.form["Equiv_Diameter"])
    Eccentricity = float(request.form["Eccentricity"])
    Solidity = float(request.form["Solidity"])
    Extent = float(request.form["Extent"])
    Roundness = float(request.form["Roundness"])
    Aspect_Ration = float(request.form["Aspect_Ration"])
    Compactness = float(request.form["Compactness"])

    # Convert into DataFrame
    input_data = pd.DataFrame([[Area, Perimeter, Major_Axis_Length, Minor_Axis_Length,
                                Convex_Area, Equiv_Diameter, Eccentricity, Solidity,
                                Extent, Roundness, Aspect_Ration, Compactness]],
                              columns=["Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length",
                                       "Convex_Area", "Equiv_Diameter", "Eccentricity", "Solidity",
                                       "Extent", "Roundness", "Aspect_Ration", "Compactness"])

    # Scale the input
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]

    # Convert numeric prediction back to class name
    result = le.inverse_transform([prediction])[0]

    return render_template("predict.html", prediction_text="Predicted Seed Class: " + result)


if __name__ == "__main__":
    app.run(debug=True)
