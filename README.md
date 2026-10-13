# 📈 RegressIQ Studio — Enterprise Linear Regression Application

RegressIQ Studio is a high-performance, full-stack Machine Learning web application designed to predict **Healthcare Insurance Annual Charges** and **California Census Block Median Housing Values** using scikit-learn Linear Regression pipelines with automatic preprocessing (`ColumnTransformer`, `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`).

---

## 🌟 Key Features

1. **Dual Machine Learning Pipelines**:
   - **Healthcare Insurance Charges Predictor**: Estimates annual insurance charges based on Age, Biological Sex, BMI, Number of Children, Smoker status, and US Region.
   - **California Housing Valuation**: Predicts median house value based on Geographic Coordinates (Longitude & Latitude), Housing Age, Room & Bedroom counts, Population, Households, Median Income, and Ocean Proximity.
2. **Enterprise Glassmorphic Web Dashboard**:
   - **Real-Time Interactive Prediction Engine**: Calculate predictions instantly as you adjust sliders and inputs without page reloads.
   - **Pre-Configured Scenario Profiles**: One-click test scenarios (e.g. *Healthy College Student*, *High-Risk Smoker*, *SF Bay Area Luxury*, *Inland Central Valley Home*).
   - **Dynamic Risk & Mortgage Analytics**: Real-time risk tier categorization, estimated monthly breakdown, and 30-year fixed mortgage estimates.
   - **Exploratory Data Analysis (EDA) Hub**: High-resolution Correlation Heatmaps, R² scores, MAE, RMSE metrics, and Actual vs. Predicted scatter plots.
3. **Dual Serving Interfaces**:
   - **Modern Flask Web App (`app.py`)**: Glassmorphism UI + RESTful JSON API.
   - **Interactive Gradio App (`gradio_app.py`)**: Standalone Gradio Blocks UI.
4. **RESTful API**:
   - `POST /api/predict/insurance`
   - `POST /api/predict/housing`
   - `GET /api/metrics`

---

## 🚀 Quick Start Guide

### 1. Run the Flask Web Application (Recommended)
Double-click `run_app.bat` or run:
```bash
.venv\Scripts\python.exe app.py
```
Then visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

### 2. Run the Gradio App
Double-click `run_gradio.bat` or run:
```bash
.venv\Scripts\python.exe gradio_app.py
```
Then visit [http://127.0.0.1:7860](http://127.0.0.1:7860).

### 3. Re-train Models & Regenerate Plots
To retrain models and update the evaluation metrics & plots:
```bash
.venv\Scripts\python.exe train_models.py
```

---

## 📊 Model Evaluation Summary

| Dataset | Target Variable | R² Score | MAE | RMSE | Preprocessing Pipeline |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Insurance** | `charges` | **0.7836** | $4,181.19 | $5,796.28 | Median Imputer + Standard Scaler + One-Hot Encoding |
| **CA Housing** | `median_house_value` | **0.6253** | $50,761.16 | $70,049.62 | Median Imputer + Standard Scaler + One-Hot Encoding |

---

## 📂 Project Architecture

```
d:/Regression App/
│
├── housing.csv                  # California Housing dataset
├── insurance.csv                # Healthcare Insurance dataset
├── train_models.py              # Model training, evaluation, and chart generation script
├── app.py                       # Flask server & REST API
├── gradio_app.py                # Gradio Blocks UI application
├── run_app.bat                  # 1-Click launcher for Flask app
├── run_gradio.bat               # 1-Click launcher for Gradio app
├── requirements.txt             # Dependency manifest
│
├── templates/
│   └── index.html               # Responsive glassmorphism web interface
│
├── static/
│   └── img/                     # Generated correlation heatmaps and regression plots
│       ├── insurance_corr.png
│       ├── insurance_pred.png
│       ├── housing_corr.png
│       └── housing_pred.png
│
├── insurance_model.joblib       # Serialized scikit-learn pipeline for Insurance
├── housing_model.joblib         # Serialized scikit-learn pipeline for Housing
└── metrics.json                 # JSON metrics and coefficients
```
