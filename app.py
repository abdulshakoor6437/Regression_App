import os
import sys
import json
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

app = Flask(__name__, static_folder="static", template_folder="templates")

# Paths
INSURANCE_MODEL_PATH = "insurance_model.joblib"
HOUSING_MODEL_PATH = "housing_model.joblib"
METRICS_PATH = "metrics.json"

ins_model = None
house_model = None
metrics = {}

def load_resources():
    global ins_model, house_model, metrics
    if os.path.exists(INSURANCE_MODEL_PATH):
        ins_model = joblib.load(INSURANCE_MODEL_PATH)
    if os.path.exists(HOUSING_MODEL_PATH):
        house_model = joblib.load(HOUSING_MODEL_PATH)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

# Initial load
load_resources()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/insurance", methods=["GET", "POST"])
def insurance_route():
    return render_template("index.html", default_tab="insurance")

@app.route("/housing", methods=["GET", "POST"])
def housing_route():
    return render_template("index.html", default_tab="housing")

@app.route("/api/predict/insurance", methods=["POST", "GET"])
def predict_insurance():
    global ins_model
    if ins_model is None:
        load_resources()
    if ins_model is None:
        return jsonify({"status": "error", "message": "Insurance model not trained yet"}), 500

    try:
        data = request.get_json(silent=True) or {}
        if not data and request.form:
            data = request.form.to_dict()
        if not data and request.args:
            data = request.args.to_dict()

        age = float(data.get("age", 30))
        sex = str(data.get("sex", "male"))
        bmi = float(data.get("bmi", 25.0))
        children = int(data.get("children", 0))
        smoker = str(data.get("smoker", "no"))
        region = str(data.get("region", "southwest"))

        row = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])
        pred = float(ins_model.predict(row)[0])
        pred_clamped = max(0.0, pred)
        return jsonify({
            "status": "success",
            "raw": round(pred_clamped, 2),
            "formatted": f"${pred_clamped:,.2f}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/predict/housing", methods=["POST", "GET"])
def predict_housing():
    global house_model
    if house_model is None:
        load_resources()
    if house_model is None:
        return jsonify({"status": "error", "message": "Housing model not trained yet"}), 500

    try:
        data = request.get_json(silent=True) or {}
        if not data and request.form:
            data = request.form.to_dict()
        if not data and request.args:
            data = request.args.to_dict()

        row = pd.DataFrame([{
            "longitude": float(data.get("longitude", -122.23)),
            "latitude": float(data.get("latitude", 37.88)),
            "housing_median_age": float(data.get("housing_median_age", 30)),
            "total_rooms": float(data.get("total_rooms", 880)),
            "total_bedrooms": float(data.get("total_bedrooms", 129)),
            "population": float(data.get("population", 322)),
            "households": float(data.get("households", 126)),
            "median_income": float(data.get("median_income", 8.3)),
            "ocean_proximity": str(data.get("ocean_proximity", "NEAR BAY"))
        }])
        pred = float(house_model.predict(row)[0])
        pred_clamped = max(10000.0, pred)
        return jsonify({
            "status": "success",
            "raw": round(pred_clamped, 2),
            "formatted": f"${pred_clamped:,.2f}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    global metrics
    if not metrics and os.path.exists(METRICS_PATH):
        load_resources()
    return jsonify(metrics)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f">> RegressIQ Studio running on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
