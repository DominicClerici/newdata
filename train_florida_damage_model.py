import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

PROCESSED_DATA_PATH = r"./florida_tornado_features_countyPop_stateHPI.csv"
MODEL_OUTPUT_PATH = r"./florida_damage_xgb_model.joblib"
RESULTS_DIR = r"./Results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading preprocessed data...")
df = pd.read_csv(PROCESSED_DATA_PATH)

target = "DAMAGE_PROPERTY_LOG"
features = [
    "YEAR",
    "MONTH",
    "EF_SCALE_NUM",
    "TOR_LENGTH",
    "TOR_WIDTH",
    "Population_Density",
    "State_HPI",
]
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# only scaling numerical features. YEAR and MONTH are treated as numerical here.
# EF_SCALE_NUM is ordinal, scaling is generally okay for tree models.
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "xgb",
            xgb.XGBRegressor(
                objective="reg:squarederror", random_state=42, n_estimators=100
            ),
        ),
    ]
)

# cross-validation
print("\nPerforming 5-Fold Cross-Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Use neg_mean_squared_error as XGBoost aims to minimize error
cv_scores_mse = cross_val_score(
    pipeline, X_train, y_train, cv=kf, scoring="neg_mean_squared_error"
)
cv_scores_rmse = np.sqrt(-cv_scores_mse)
cv_scores_r2 = cross_val_score(
    pipeline, X_train, y_train, cv=kf, scoring="r2"
)

print(f"Cross-Validation RMSE: {cv_scores_rmse.mean():.4f} +/- {cv_scores_rmse.std():.4f}")
print(f"Cross-Validation R2:   {cv_scores_r2.mean():.4f} +/- {cv_scores_r2.std():.4f}")

print("\nTraining final model on the full training set...")
pipeline.fit(X_train, y_train)
print("Model training complete.")

print("\nEvaluating model on the test set...")
y_pred_log = pipeline.predict(X_test)

y_pred_original = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

y_pred_original[y_pred_original < 0] = 0

# calculate metrics on the original scale
mae = mean_absolute_error(y_test_original, y_pred_original)
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)

print("\n--- Test Set Performance (Original Scale) ---")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R2): {r2:.4f}")


# 1. Predicted vs. Actual Plot (Original Scale)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_original, y=y_pred_original, alpha=0.6)
plt.plot(
    [y_test_original.min(), y_test_original.max()],
    [y_test_original.min(), y_test_original.max()],
    "--r",
    linewidth=2,
)
plt.xlabel("Actual Damage ($)")
plt.ylabel("Predicted Damage ($)")
plt.title("Predicted vs. Actual Tornado Damage (Original Scale)")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig(os.path.join(RESULTS_DIR, "predicted_vs_actual.png"))
plt.close()
print("   - Saved Predicted vs. Actual plot.")

# 2. Residual Plot (Log Scale)
residuals_log = y_test - y_pred_log
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_log, y=residuals_log, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Damage (Log Scale)")
plt.ylabel("Residuals (Log Scale)")
plt.title("Residual Plot (Log Scale)")
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig(os.path.join(RESULTS_DIR, "residual_plot_log.png"))
plt.close()
print("   - Saved Residual plot.")

# 3. Feature Importance Plot
model = pipeline.named_steps["xgb"]
importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
)
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("XGBoost Feature Importance for Tornado Damage Prediction")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"))
plt.close()
print("   - Saved Feature Importance plot.")

joblib.dump(pipeline, MODEL_OUTPUT_PATH)
print(f"\n✅ Trained model pipeline saved to: {MODEL_OUTPUT_PATH}")

