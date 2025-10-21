from flask import Flask, request, render_template
from pickle import load
import numpy as np

# Load all pickle files
f = open("model.pkl", "rb")
model = load(f)
f.close()

f = open("scaler.pkl", "rb")
scaler = load(f)
f.close()

f = open("encoder.pkl", "rb")
encoder = load(f)
f.close()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    res = ""
    if request.method == "POST":
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        Humidity = float(request.form["Humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        data = [N, P, K, temperature, Humidity, ph, rainfall]
        scaled_data = scaler.transform([data])
        pred = model.predict(scaled_data)
        res = encoder.inverse_transform(pred)[0]

    return render_template("home.html", res=res)

if __name__ == "__main__":
    app.run(debug=True)
