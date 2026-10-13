import os
import sys
import pandas as pd
import joblib
import gradio as gr

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Load pre-trained models
if not os.path.exists("insurance_model.joblib") or not os.path.exists("housing_model.joblib"):
    print("Training models first...")
    import train_models

ins_model = joblib.load("insurance_model.joblib")
house_model = joblib.load("housing_model.joblib")

def predict_insurance(age, sex, bmi, children, smoker, region):
    row = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])
    pred = ins_model.predict(row)[0]
    return f"${max(0.0, pred):,.2f}"

def predict_house(lon, lat, age, rooms, bedrooms, pop, households, income, ocean):
    row = pd.DataFrame([{
        "longitude": lon,
        "latitude": lat,
        "housing_median_age": age,
        "total_rooms": rooms,
        "total_bedrooms": bedrooms,
        "population": pop,
        "households": households,
        "median_income": income,
        "ocean_proximity": ocean
    }])
    pred = house_model.predict(row)[0]
    return f"${max(10000.0, pred):,.2f}"

custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="emerald",
)

with gr.Blocks(title="RegressIQ • Linear Regression Predictors", theme=custom_theme) as demo:
    gr.Markdown("# 📊 RegressIQ Multi-Model Regression Predictor")
    gr.Markdown("Real-time inference using scikit-learn Linear Regression pipelines with ColumnTransformer.")

    with gr.Tab("🏥 Healthcare Insurance Charges"):
        gr.Markdown("### Predict Annual Insurance Premiums")
        with gr.Row():
            with gr.Column():
                a = gr.Slider(18, 65, value=30, step=1, label="Age")
                s = gr.Radio(["male", "female"], value="male", label="Biological Sex")
                b = gr.Number(value=25.0, label="Body Mass Index (BMI)")
                c = gr.Slider(0, 5, value=0, step=1, label="Children / Dependents")
                sm = gr.Radio(["no", "yes"], value="no", label="Smoker Status")
                r = gr.Dropdown(["northeast", "northwest", "southeast", "southwest"],
                                value="southwest", label="US Region")
                btn_ins = gr.Button("Calculate Insurance Charges", variant="primary")
            with gr.Column():
                out1 = gr.Textbox(label="Predicted Annual Charges", scale=2)
                btn_ins.click(predict_insurance, [a, s, b, c, sm, r], out1)

    with gr.Tab("🏡 California House Price"):
        gr.Markdown("### Predict California Census Block Median House Value")
        with gr.Row():
            with gr.Column():
                lon = gr.Number(value=-122.23, label="Longitude")
                lat = gr.Number(value=37.88, label="Latitude")
                hage = gr.Slider(1, 52, value=30, step=1, label="Housing Median Age")
                rooms = gr.Number(value=880, label="Total Rooms")
                beds = gr.Number(value=129, label="Total Bedrooms")
                pop = gr.Number(value=322, label="Population")
                hh = gr.Number(value=126, label="Households")
                inc = gr.Number(value=8.3, label="Median Income (in $10,000s)")
                ocean = gr.Dropdown(["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
                                    value="NEAR BAY", label="Ocean Proximity")
                btn_house = gr.Button("Calculate Median House Value", variant="primary")
            with gr.Column():
                out2 = gr.Textbox(label="Predicted Median House Value", scale=2)
                btn_house.click(predict_house, [lon, lat, hage, rooms, beds, pop, hh, inc, ocean], out2)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
