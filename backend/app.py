"""
app.py — RecoveryAI Flask Backend
==================================
Accepts 4 raw wearable inputs (5-day history each), computes ALL engineered
features internally (matching train.py exactly), and returns HRV prediction
with confidence interval and readiness classification.

Run:
    pip install flask flask-cors joblib numpy xgboost scikit-learn
    python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os
from datetime import datetime

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Model artifacts live in ./model/ relative to this file
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(BASE), "model")
# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
try:
    model        = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    residual_std = joblib.load(os.path.join(MODEL_DIR, "residual_std.pkl"))

    with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # Use feature_index from metadata.json — this is the authoritative source
    # saved by train.py at the same time as the final model.
    # The top-level feature_index.pkl may differ (duplicate from nested folder).
    feature_index = np.array(metadata["feature_index"])
    features_raw  = metadata["features_raw"]   # 7 raw feature names
    window_size   = metadata["window_size"]    # 5

    print(f"✅ Model loaded | expects {model.n_features_in_} features")
    print(f"   window_size={window_size}, features_raw={features_raw}")
    print(f"   feature_index length={len(feature_index)}")

except Exception as e:
    print(f"❌ Failed to load model artifacts: {e}")
    raise SystemExit(1)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING  (mirrors train.py exactly)
# ─────────────────────────────────────────────

def compute_rolling_median(values, window=14):
    """Rolling median over a list; returns last value (most recent day)."""
    if len(values) == 0:
        return np.nan
    arr = np.array(values, dtype=float)
    start = max(0, len(arr) - window)
    return float(np.median(arr[start:]))


def compute_ewm(values, span=3):
    """Exponentially weighted mean (span=3) — returns last value."""
    arr = np.array(values, dtype=float)
    alpha = 2.0 / (span + 1)
    ewm_val = arr[0]
    for v in arr[1:]:
        ewm_val = alpha * v + (1 - alpha) * ewm_val
    return float(ewm_val)


def compute_rolling_mean(values, window=5):
    """Rolling mean over last `window` values — returns last value."""
    arr = np.array(values, dtype=float)
    start = max(0, len(arr) - window)
    return float(np.mean(arr[start:]))


def engineer_features(hrv_history, hr_history, sleep_history, steps_history):
    """
    Compute the 7 engineered features for each of the 5 window days,
    producing a flat 35-element vector matching the scaler's training input.

    Parameters
    ----------
    hrv_history   : list[float], length >= window_size + 14 ideally, min window_size
    hr_history    : list[float], same length
    sleep_history : list[float], same length
    steps_history : list[float], same length

    Returns
    -------
    np.ndarray, shape (35,)  — [feat0_day0, feat1_day0, ..., feat6_day4]
    """
    n = len(hrv_history)
    if n < window_size:
        raise ValueError(f"Need at least {window_size} days of history, got {n}")

    # We'll compute features for each day in the last `window_size` days
    feature_rows = []  # will be window_size rows of 7 features each

    for day_offset in range(window_size):
        # Index into history: day 0 = oldest in window, day 4 = most recent
        # The window covers positions [n - window_size + day_offset]
        # History up to and including this day:
        idx = n - window_size + day_offset  # 0-based index of this day

        hrv_up_to   = hrv_history[:idx + 1]
        hr_up_to    = hr_history[:idx + 1]
        sleep_up_to = sleep_history[:idx + 1]
        steps_up_to = steps_history[:idx + 1]

        hrv_today   = float(hrv_history[idx])
        hr_today    = float(hr_history[idx])
        sleep_today = float(sleep_history[idx])

        # Rolling 14-day baselines (median, min_periods=7 → use what we have)
        hrv_base   = compute_rolling_median(hrv_up_to[-14:])
        hr_base    = compute_rolling_median(hr_up_to[-14:])
        sleep_base = compute_rolling_median(sleep_up_to[-14:])

        # Z-scores (deviation from personal baseline)
        hrv_z   = hrv_today   - hrv_base
        hr_z    = hr_today    - hr_base
        sleep_z = sleep_today - sleep_base

        # Strain EWM: 3-day exponentially weighted mean of steps
        strain_ewm = compute_ewm(steps_up_to[-3:] if len(steps_up_to) >= 3 else steps_up_to, span=3)

        # Sleep debt: 5-day rolling mean of sleep minus today's sleep
        sleep_5d_mean = compute_rolling_mean(sleep_up_to[-5:] if len(sleep_up_to) >= 5 else sleep_up_to, window=5)
        sleep_debt = sleep_5d_mean - sleep_today

        # HRV trend: first difference of 3-day rolling mean of HRV
        # rolling mean at current day
        hrv_roll_now  = compute_rolling_mean(hrv_up_to[-3:], window=3)
        # rolling mean at previous day (if available)
        if idx > 0:
            hrv_prev = hrv_history[:idx]
            hrv_roll_prev = compute_rolling_mean(hrv_prev[-3:] if len(hrv_prev) >= 3 else hrv_prev, window=3)
            hrv_trend = hrv_roll_now - hrv_roll_prev
        else:
            hrv_trend = 0.0

        # Recovery ratio: HRV / (avg_hr + 1)
        recovery_ratio = hrv_today / (hr_today + 1.0)

        feature_rows.append([hrv_z, hr_z, sleep_z, strain_ewm, sleep_debt, hrv_trend, recovery_ratio])

    # Flatten: [day0_feat0, day0_feat1, ..., day0_feat6, day1_feat0, ...]
    # This matches: for i in range(window_size): for f in features_raw: vec.append(...)
    flat = []
    for row in feature_rows:
        flat.extend(row)

    return np.array(flat, dtype=float)


def classify_readiness(predicted_hrv, hrv_history):
    """
    Classify readiness using z-score relative to 14-day personal baseline.
    Returns zone string and z_score.
    """
    baseline = compute_rolling_median(hrv_history[-14:])
    std_val  = float(np.std(hrv_history[-14:])) if len(hrv_history) >= 2 else 1.0
    if std_val < 0.01:
        std_val = 1.0

    z = (predicted_hrv - baseline) / std_val

    if z >= -0.5:
        zone = "excellent"
    elif z >= -1.0:
        zone = "good"
    elif z >= -1.5:
        zone = "fair"
    else:
        zone = "poor"

    return zone, float(z), float(baseline)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({"status": "RecoveryAI API running", "version": "2.0"})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "XGBoost HRV Regressor",
        "window_size": window_size,
        "n_model_features": int(model.n_features_in_)
    }), 200


@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    """Return model performance metrics for the dashboard."""
    return jsonify({
        "r2_test":      metadata["r2_test"],
        "mae_test":     metadata["mae_test"],
        "rmse_test":    metadata["rmse_test"],
        "residual_std": metadata["residual_std"],
        "cv_r2_mean":   metadata["cv_r2_mean"],
        "cv_r2_std":    metadata["cv_r2_std"],
        "window_size":  metadata["window_size"],
        "features_raw": metadata["features_raw"],
        "top_features": metadata["top_features"],
        "feature_ranges": metadata["feature_ranges"],
        "readiness_thresholds": metadata["readiness_thresholds"]
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts 4 raw wearable inputs (each a list of at least 5 daily values).
    Computes all engineered features internally and returns HRV prediction.

    Request body (JSON):
    {
        "hrv_rmssd_ms":        [float, ...],  // >= 5 values, most recent last
        "avg_hr_day_bpm":      [float, ...],
        "sleep_duration_hours":[float, ...],
        "steps":               [float, ...]
    }

    Response:
    {
        "prediction":           float,   // predicted next-day HRV in ms
        "confidence_interval":  { "lower": float, "upper": float },
        "readiness_zone":       str,     // "excellent" | "good" | "fair" | "poor"
        "readiness_z_score":    float,
        "personal_baseline_hrv":float,
        "shap_insight":         str,
        "timestamp":            str
    }
    """
    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "Request body must be JSON"}), 400

        # ── Validate required fields ──────────────────────────────────────
        required = ["hrv_rmssd_ms", "avg_hr_day_bpm", "sleep_duration_hours", "steps"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing required field: '{field}'"}), 400
            if not isinstance(data[field], list) or len(data[field]) < window_size:
                return jsonify({
                    "error": f"'{field}' must be a list with at least {window_size} values"
                }), 400

        hrv_hist   = [float(v) for v in data["hrv_rmssd_ms"]]
        hr_hist    = [float(v) for v in data["avg_hr_day_bpm"]]
        sleep_hist = [float(v) for v in data["sleep_duration_hours"]]
        steps_hist = [float(v) for v in data["steps"]]

        # ── Basic sanity checks (mirrors train.py cleaning) ───────────────
        for hr in hr_hist:
            if not (30 <= hr <= 200):
                return jsonify({"error": f"avg_hr_day_bpm value {hr} out of range (30–200)"}), 400
        for sl in sleep_hist:
            if not (0 < sl < 15):
                return jsonify({"error": f"sleep_duration_hours value {sl} out of range (0–15)"}), 400
        for st in steps_hist:
            if st < 0:
                return jsonify({"error": f"steps value {st} cannot be negative"}), 400

        # ── Feature engineering ───────────────────────────────────────────
        X_raw = engineer_features(hrv_hist, hr_hist, sleep_hist, steps_hist)
        # shape: (35,)

        # ── Scale (fit was on 35-feature vectors) ─────────────────────────
        X_scaled = scaler.transform(X_raw.reshape(1, -1))
        # shape: (1, 35)

        # ── SHAP feature selection ────────────────────────────────────────
        X_selected = X_scaled[:, feature_index]
        # shape: (1, 25)

        # ── Predict ───────────────────────────────────────────────────────
        pred = float(model.predict(X_selected)[0])

        # ── Confidence interval (±1.96σ) ─────────────────────────────────
        margin = 1.96 * float(residual_std)
        lower  = pred - margin
        upper  = pred + margin

        # ── Readiness classification ──────────────────────────────────────
        zone, z_score, baseline = classify_readiness(pred, hrv_hist)

        # ── SHAP insight (rule-based, no re-computation needed) ───────────
        # Use the most recent day's engineered features for a plain-language note
        last_hrv_z   = X_raw[7 * (window_size - 1) + 0]   # hrv_z day4
        last_rr      = X_raw[7 * (window_size - 1) + 6]   # recovery_ratio day4
        last_trend   = X_raw[7 * (window_size - 1) + 5]   # hrv_trend day4

        insights = []
        if last_hrv_z > 5:
            insights.append("HRV is well above your baseline — strong recovery signal")
        elif last_hrv_z < -5:
            insights.append("HRV is below your baseline — body may still be recovering")
        if last_rr > 0.8:
            insights.append("recovery ratio is high, indicating good cardiac efficiency")
        elif last_rr < 0.4:
            insights.append("recovery ratio is low — consider reducing training load")
        if last_trend > 3:
            insights.append("HRV trend is rising — positive adaptation")
        elif last_trend < -3:
            insights.append("HRV trend is declining — monitor fatigue")

        shap_insight = "; ".join(insights) if insights else "Features are within normal ranges"

        return jsonify({
            "prediction":            round(pred, 2),
            "confidence_interval":   {"lower": round(lower, 2), "upper": round(upper, 2)},
            "readiness_zone":        zone,
            "readiness_z_score":     round(z_score, 3),
            "personal_baseline_hrv": round(baseline, 2),
            "shap_insight":          shap_insight,
            "timestamp":             datetime.now().isoformat()
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": "Internal prediction error — check server logs"}), 500


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
