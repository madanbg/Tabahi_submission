"""
CPRI Hackathon - Interactive Web Dashboard Backend
Flask Server with Live Machine Learning Inference & Analytics
Team: Tabahi
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add Tabahi_submission to path to import solution pipeline
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(BASE_DIR, "Tabahi_submission")):
    SUBMISSION_DIR = os.path.join(BASE_DIR, "Tabahi_submission")
else:
    SUBMISSION_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(SUBMISSION_DIR, "output")
DATASET_PATH = os.path.join(SUBMISSION_DIR, "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx")

if SUBMISSION_DIR not in sys.path:
    sys.path.append(SUBMISSION_DIR)
import solution

app = Flask(__name__, template_folder="templates", static_folder="static")

# In-memory model cache
MODELS = {
    "clf": None,
    "reg": None,
    "iso_model": None,
    "impute_values": None,
    "feat_cols": None,
    "clf_importance": None,
    "reg_importance": None,
    "summary": None,
    "test_df": None,
    "cv_metrics": {}
}

def initialize_models():
    """Load dataset, train models, and cache predictions on startup."""
    print("[INIT] Training machine learning models...")
    train, test = solution.load_data(DATASET_PATH)

    # Feature engineering
    train_feat, impute_values, iso_model = solution.engineer_features(train, fit_iso=True)
    test_feat, _, _ = solution.engineer_features(
        test, impute_values=impute_values, iso_model=iso_model, fit_iso=False
    )
    feat_cols = solution.get_feature_columns()

    # Train Classifier & Regressor
    clf, cv_report, cv_acc, cv_f1, clf_importance = solution.train_validity_classifier(train_feat, feat_cols)
    reg, r2, mae, rmse, reg_importance = solution.train_regressor(train_feat, feat_cols)

    # Predict test data
    test_validity = clf.predict(test_feat[feat_cols])
    test_feat["Validity_Label"] = np.where(test_validity == 1, "Invalid", "Valid")
    test_feat["Invalid_Probability"] = clf.predict_proba(test_feat[feat_cols])[:, 1]
    test_feat["Predicted_Reference_Parameter"] = reg.predict(test_feat[feat_cols]).round(4)

    # Load summary if exists or build
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        summary = solution.build_summary(test_feat, clf, feat_cols, test_feat, solution.METHOD_NOTE_TEXT)

    MODELS["clf"] = clf
    MODELS["reg"] = reg
    MODELS["iso_model"] = iso_model
    MODELS["impute_values"] = impute_values
    MODELS["feat_cols"] = feat_cols
    MODELS["clf_importance"] = clf_importance.to_dict()
    MODELS["reg_importance"] = reg_importance.to_dict()
    MODELS["summary"] = summary
    MODELS["test_df"] = test_feat
    MODELS["cv_metrics"] = {
        "clf_accuracy": round(float(cv_acc) * 100, 2),
        "clf_f1": round(float(cv_f1), 4),
        "reg_r2": round(float(r2), 4),
        "reg_mae": round(float(mae), 4),
        "reg_rmse": round(float(rmse), 4)
    }
    print("[INIT] Models loaded and ready!")

@app.route("/")
def index():
    """Render main dashboard view."""
    return render_template("index.html")

@app.route("/api/data")
def api_data():
    """Return summary, test records, and feature importances for charts."""
    test_df = MODELS["test_df"]
    records = []
    if test_df is not None:
        display_cols = [
            "Test_ID", "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C",
            "Test_Duration_min", "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4",
            "anomaly_score", "Predicted_Reference_Parameter", "Validity_Label", "Invalid_Probability"
        ]
        subset = test_df[[c for c in display_cols if c in test_df.columns]].copy()
        subset["anomaly_score"] = subset["anomaly_score"].round(4)
        subset["Invalid_Probability"] = (subset["Invalid_Probability"] * 100).round(1)
        records = subset.to_dict(orient="records")

    return jsonify({
        "summary": MODELS["summary"],
        "cv_metrics": MODELS["cv_metrics"],
        "clf_importance": MODELS["clf_importance"],
        "reg_importance": MODELS["reg_importance"],
        "records": records
    })

@app.route("/api/predict_single", methods=["POST"])
def api_predict_single():
    """Live inference endpoint for interactive 'What-If' simulator."""
    try:
        data = request.json or {}
        raw = {
            "Applied_Voltage_kV": float(data.get("Applied_Voltage_kV", 11.0)),
            "Load_Current_A": float(data.get("Load_Current_A", 630.0)),
            "Ambient_Temperature_C": float(data.get("Ambient_Temperature_C", 30.0)),
            "Test_Duration_min": float(data.get("Test_Duration_min", 120.0)),
            "Sensor_S1": float(data.get("Sensor_S1", 45.0)),
            "Sensor_S2": float(data.get("Sensor_S2", 44.5)),
            "Sensor_S3": float(data.get("Sensor_S3", 45.2)),
            "Sensor_S4": float(data.get("Sensor_S4", 25.0)),
        }
        df_single = pd.DataFrame([raw])

        # Feature engineering using cached fitted models
        df_feat, _, _ = solution.engineer_features(
            df_single,
            impute_values=MODELS["impute_values"],
            iso_model=MODELS["iso_model"],
            fit_iso=False
        )
        feat_cols = MODELS["feat_cols"]

        clf = MODELS["clf"]
        reg = MODELS["reg"]

        is_invalid = int(clf.predict(df_feat[feat_cols])[0])
        prob_invalid = float(clf.predict_proba(df_feat[feat_cols])[0, 1])
        predicted_temp = float(reg.predict(df_feat[feat_cols])[0])
        anomaly_score = float(df_feat["anomaly_score"].iloc[0])
        hard_error = int(df_feat["hard_sensor_error"].iloc[0])
        s_spread = float(df_feat["S_spread"].iloc[0])

        return jsonify({
            "status": "success",
            "validity": "Invalid" if is_invalid == 1 else "Valid",
            "is_valid": is_invalid == 0,
            "confidence": round((1 - prob_invalid if is_invalid == 0 else prob_invalid) * 100, 1),
            "invalid_probability": round(prob_invalid * 100, 1),
            "predicted_reference_parameter": round(predicted_temp, 2),
            "anomaly_score": round(anomaly_score, 4),
            "hard_sensor_error": hard_error == 1,
            "sensor_spread": round(s_spread, 2)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/download/<path:filename>")
def download_file(filename):
    """Allow downloading submission files."""
    allowed_output_files = ["Tabahi.csv", "summary.json", "summary.csv", "model_report.txt"]
    allowed_root_files = ["your-tabahi-submission.zip", "your-teamname-submission.zip"]
    
    if filename in allowed_output_files:
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)
    elif filename in allowed_root_files:
        return send_from_directory(BASE_DIR, filename, as_attachment=True)
    return jsonify({"error": "File not allowed"}), 404

if __name__ == "__main__":
    initialize_models()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" Web Dashboard available at: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
