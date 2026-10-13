import os
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("static/img", exist_ok=True)

# 1. Load Data
ins = pd.read_csv("insurance.csv")
house = pd.read_csv("housing.csv")

print("Insurance shape:", ins.shape)
print("Housing shape:", house.shape)

# 2. EDA Heatmaps
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(ins.select_dtypes("number").corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Insurance Features Correlation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("static/img/insurance_corr.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(house.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("California Housing Features Correlation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("static/img/housing_corr.png", dpi=150)
plt.close()

# 3. Model Training Function
metrics_data = {}

def train_linear_regression(df, target, name, slug):
    X = df.drop(columns=[target])
    y = df[target]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    model = Pipeline([("pre", pre), ("lr", LinearRegression())])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))
    train_r2 = float(r2_score(y_train, train_pred))

    print(f"--- {name} ---")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 4))
    print("Train R2:", round(train_r2, 4), "\n")

    # Generate Actual vs Predicted Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, pred, alpha=0.35, color="#4f46e5", edgecolors='none', s=25)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    plt.plot(lims, lims, color="#ef4444", linestyle="--", linewidth=2, label="Ideal 45° Line")
    plt.xlabel("Actual Values", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted Values", fontsize=11, fontweight="bold")
    plt.title(f"{name}: Actual vs Predicted (R² = {r2:.3f})", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"static/img/{slug}_pred.png", dpi=150)
    plt.close()

    # Extract feature names & coefficients
    ohe = model.named_steps["pre"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist() if cat_cols else []
    all_features = num_cols + cat_feature_names
    coefficients = model.named_steps["lr"].coef_.tolist()
    intercept = float(model.named_steps["lr"].intercept_)

    coef_list = []
    for f, c in zip(all_features, coefficients):
        coef_list.append({"feature": f, "coefficient": round(float(c), 4)})
    coef_list.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

    # Calculate summary stats for input fields
    summary_stats = {}
    for col in num_cols:
        summary_stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "mean": float(df[col].mean()),
        }
    for col in cat_cols:
        summary_stats[col] = {
            "unique": df[col].dropna().unique().tolist(),
            "mode": str(df[col].mode()[0])
        }

    metrics_data[slug] = {
        "name": name,
        "samples_count": len(df),
        "test_samples": len(y_test),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "train_r2": round(train_r2, 4),
        "intercept": round(intercept, 4),
        "coefficients": coef_list,
        "summary": summary_stats,
    }

    return model

ins_model = train_linear_regression(ins, "charges", "Insurance Charges", "insurance")
house_model = train_linear_regression(house, "median_house_value", "California Housing", "housing")

joblib.dump(ins_model, "insurance_model.joblib")
joblib.dump(house_model, "housing_model.joblib")

with open("metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=2)

print("Models successfully trained and saved with evaluation metrics!")
