import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import shap

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("../data/wearables_health_6mo_daily.csv") 
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['user_id', 'date'])
df = df.dropna()

# =========================
# 2. REMOVE LEAKAGE
# =========================
df = df.drop(columns=['stress_score'])  # critical fix

# =========================
# 3. BASIC CLEANING
# =========================
df = df[(df['avg_hr_day_bpm'] > 30) & (df['avg_hr_day_bpm'] < 200)]
df = df[(df['sleep_duration_hours'] > 0) & (df['sleep_duration_hours'] < 15)]
df = df[df['steps'] >= 0]

# =========================
# 4. CREATE TARGET
# =========================
# Target: predict tomorrow's raw HRV
df['target'] = df.groupby('user_id')['hrv_rmssd_ms'].shift(-1)
df = df.dropna(subset=['target'])

# =========================
# 5. ROLLING BASELINES (Adapts over time & prevents future-leakage)
# =========================
# We use a 14-day rolling median so the baseline tracks the user's changing fitness
df['hrv_base'] = df.groupby('user_id')['hrv_rmssd_ms'].transform(
    lambda x: x.rolling(14, min_periods=7).median()
)
df['hr_base'] = df.groupby('user_id')['avg_hr_day_bpm'].transform(
    lambda x: x.rolling(14, min_periods=7).median()
)
df['sleep_base'] = df.groupby('user_id')['sleep_duration_hours'].transform(
    lambda x: x.rolling(14, min_periods=7).median()
)

# We calculate the HRV standard deviation specifically for the App's Traffic Light logic
df['hrv_14d_std'] = df.groupby('user_id')['hrv_rmssd_ms'].transform(
    lambda x: x.rolling(14, min_periods=7).std()
)

# Drop any rows where the rolling windows couldn't calculate a baseline
df = df.dropna(subset=['hrv_base', 'hrv_14d_std'])

# =========================
# 6. NORMALIZED FEATURES
# =========================
df['hrv_z'] = df['hrv_rmssd_ms'] - df['hrv_base']
df['hr_z'] = df['avg_hr_day_bpm'] - df['hr_base']
df['sleep_z'] = df['sleep_duration_hours'] - df['sleep_base']

# =========================
# 7. TIME FEATURES
# =========================
df['strain_ewm'] = df.groupby('user_id')['steps'].transform(
    lambda x: x.ewm(span=3).mean()
)

df['sleep_debt'] = df.groupby('user_id')['sleep_duration_hours'].transform(
    lambda x: x.rolling(5, min_periods=1).mean() - x
)

df['hrv_trend'] = df.groupby('user_id')['hrv_rmssd_ms'].transform(
    lambda x: x.rolling(3, min_periods=1).mean().diff()
)

df['recovery_ratio'] = df['hrv_rmssd_ms'] / (df['avg_hr_day_bpm'] + 1)

# =========================
# 8. FINAL FEATURES
# =========================
features_raw = [
    'hrv_z',
    'hr_z',
    'sleep_z',
    'strain_ewm',
    'sleep_debt',
    'hrv_trend',
    'recovery_ratio'
]

df = df.dropna()

# =========================
# 9. 5-DAY WINDOWS
# =========================
X_windows, y_windows, groups = [], [], []
window_size = 5

for user in df['user_id'].unique():
    user_df = df[df['user_id'] == user].reset_index(drop=True)

    for i in range(len(user_df) - window_size):
        window = user_df.iloc[i:i+window_size]

        X_windows.append(window[features_raw].values.flatten())
        # Change this line in Step 9:
        y_windows.append(user_df.iloc[i+window_size]['target'])
        groups.append(user)

X = np.array(X_windows)
y = np.array(y_windows)
groups = np.array(groups)

print(f"Total samples: {len(X)}")
print(f"Features shape: {X.shape}")

# =========================
# 10. USER SPLIT (stratified by user)
# =========================
users = np.unique(groups)
train_users, test_users = train_test_split(users, test_size=0.2, random_state=42)

train_mask = np.isin(groups, train_users)
test_mask = np.isin(groups, test_users)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
groups_train, groups_test = groups[train_mask], groups[test_mask]

print(f"\nTrain samples: {len(X_train)} | Test samples: {len(X_test)}")

# =========================
# 11. FEATURE SCALING (fit ONLY on train)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 12. BASE MODEL + HYPERPARAMETER TUNING
# =========================
print("\n=== Training Base Model ===")
base_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=20,
    eval_metric='rmse'
)

# Early stopping on validation set
train_size = int(0.8 * len(X_train_scaled))
X_tr, X_val = X_train_scaled[:train_size], X_train_scaled[train_size:]
y_tr, y_val = y_train[:train_size], y_train[train_size:]

base_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)

val_pred = base_model.predict(X_val)
print(f"Base model val R²: {r2_score(y_val, val_pred):.4f}")

# =========================
# 13. SHAP FEATURE SELECTION (without leakage)
# =========================
print("\n=== SHAP Feature Selection ===")
explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(X_val)  # Use validation set, not train

mean_shap = np.abs(shap_values).mean(axis=0)

# Create feature names with time index
feature_names = []
for i in range(window_size):
    for f in features_raw:
        feature_names.append(f"{f}_day{i}")

importance_list = sorted(zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True)

print("Top 15 SHAP Features:")
for fname, val in importance_list[:15]:
    print(f"  {fname}: {val:.4f}")

# Select top 25 features (or all if fewer available)
top_k = min(25, len(importance_list))
top_features = [f[0] for f in importance_list[:top_k]]
feature_index = np.array([feature_names.index(f) for f in top_features])

X_train_sel = X_train_scaled[:, feature_index]
X_test_sel = X_test_scaled[:, feature_index]
X_val_sel = X_val[:, feature_index]

print(f"\nSelected {len(feature_index)} features")

# =========================
# 14. FINAL MODEL (fit on FULL training set)
# =========================
print("\n=== Training Final Model ===")
final_model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=30,
    eval_metric='rmse'
)

# Split for early stopping
split_pt = int(0.85 * len(X_train_sel))
X_tr_full, X_val_full = X_train_sel[:split_pt], X_train_sel[split_pt:]
y_tr_full, y_val_full = y_train[:split_pt], y_train[split_pt:]

final_model.fit(
    X_tr_full, y_tr_full,
    eval_set=[(X_val_full, y_val_full)],
    verbose=False
)

# =========================
# 15. EVALUATE ON TEST SET
# =========================
print("\n=== TEST SET EVALUATION ===")
test_pred = final_model.predict(X_test_sel)

r2_test = r2_score(y_test, test_pred)
mae_test = mean_absolute_error(y_test, test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))

print(f"R²:   {r2_test:.4f}")
print(f"MAE:  {mae_test:.4f}")
print(f"RMSE: {rmse_test:.4f}")

# Estimate residual std for uncertainty
residuals = y_test - test_pred
residual_std = np.std(residuals)
print(f"Residual Std: {residual_std:.4f}")

# =========================
# 16. GROUP K-FOLD CV (with proper SHAP per fold)
# =========================
print("\n=== GROUP K-FOLD CROSS-VALIDATION ===")
gkf = GroupKFold(n_splits=5)

r2_scores, mae_scores = [], []
fold_shap_importance = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups_train)):
    X_tr, X_val_fold = X_train_scaled[tr_idx], X_train_scaled[val_idx]
    y_tr, y_val_fold = y_train[tr_idx], y_train[val_idx]

    # SHAP feature selection happens INSIDE the fold
    fold_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    fold_model.fit(X_tr, y_tr, verbose=False)

    # Get SHAP on fold validation
    fold_explainer = shap.TreeExplainer(fold_model)
    fold_shap = fold_explainer.shap_values(X_val_fold)
    fold_shap_importance.append(np.abs(fold_shap).mean(axis=0))

    # Select top features for this fold
    fold_mean_shap = np.abs(fold_shap).mean(axis=0)
    fold_top_idx = np.argsort(fold_mean_shap)[-top_k:]

    X_tr_fold_sel = X_tr[:, fold_top_idx]
    X_val_fold_sel = X_val_fold[:, fold_top_idx]

    fold_final_model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=42,
    )
    fold_final_model.fit(X_tr_fold_sel, y_tr, verbose=False)

    fold_pred = fold_final_model.predict(X_val_fold_sel)
    r2_scores.append(r2_score(y_val_fold, fold_pred))
    mae_scores.append(mean_absolute_error(y_val_fold, fold_pred))

    print(f"Fold {fold+1} | R²: {r2_scores[-1]:.4f} | MAE: {mae_scores[-1]:.4f}")

print(f"\nMean R²:  {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Mean MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")

# =========================
# 17. SAVE ARTIFACTS
# =========================
print("\n=== SAVING MODEL & ARTIFACTS ===")

import os
import joblib
import json

# Create artifacts directory if needed
os.makedirs("model", exist_ok=True)

# MUST HAVE ALL 6 OF THESE:
joblib.dump(final_model, "model/model.pkl")
joblib.dump(feature_index, "model/feature_index.pkl")
joblib.dump(features_raw, "model/features_raw.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(feature_names, "model/feature_names.pkl")
joblib.dump(residual_std, "model/residual_std.pkl")

# Save metadata
metadata = {
    "window_size": window_size,
    "features_raw": features_raw,
    "top_features": top_features,
    "feature_index": feature_index.tolist(),
    "r2_test": float(r2_test),
    "mae_test": float(mae_test),
    "rmse_test": float(rmse_test),
    "residual_std": float(residual_std),
    "cv_r2_mean": float(np.mean(r2_scores)),
    "cv_r2_std": float(np.std(r2_scores)),
    "feature_ranges": {
        "hrv_z": {"min": float(df['hrv_z'].min()), "max": float(df['hrv_z'].max())},
        "hr_z": {"min": float(df['hr_z'].min()), "max": float(df['hr_z'].max())},
        "sleep_z": {"min": float(df['sleep_z'].min()), "max": float(df['sleep_z'].max())},
        "strain_ewm": {"min": float(df['strain_ewm'].min()), "max": float(df['strain_ewm'].max())},
        "sleep_debt": {"min": float(df['sleep_debt'].min()), "max": float(df['sleep_debt'].max())},
        "hrv_trend": {"min": float(df['hrv_trend'].min()), "max": float(df['hrv_trend'].max())},
        "recovery_ratio": {"min": float(df['recovery_ratio'].min()), "max": float(df['recovery_ratio'].max())},
    },
    "readiness_thresholds": {
        "logic": "z_score_from_14d_baseline",
        "optimal_min_z": -0.5,
        "normal_min_z": -1.5,
        "descriptions": {
            "optimal": "🟢 Optimal: Your body is fully recovered and ready for high strain.",
            "normal": "🟡 Normal: You are adequately recovered. Stick to your normal routine.",
            "fatigued": "🔴 Fatigued: Your predicted HRV is significantly below baseline. Prioritize rest."
        }
    }
}

with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✓ All 6 Model files and Metadata saved!")