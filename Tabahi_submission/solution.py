"""
CPRI State-Level Hackathon - Screening Round
Team: Tabahi

End-to-end, fully automated solution for:
  Task 1 - Identify abnormal / invalid records
  Task 2 - Predict the Reference_Parameter (hot-spot temperature rise)
  Task 3 - Auto-generate a test summary (summary.json / summary.csv)

Run:
    python solution.py --input CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx --team Tabahi

Outputs (written to ./output/):
    Tabahi.csv        -> Test_ID, Predicted_Reference_Parameter, Validity_Label
    summary.json       -> automated test summary (Task 3)
    summary.csv         -> same summary, flat CSV form
    model_report.txt   -> cross-validated performance metrics (for the methodology note)

No record is touched or edited by hand anywhere in this script - every step
(imputation, feature engineering, anomaly detection, classification,
regression, summary generation) is fully programmatic and reproducible.
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_val_predict,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

RAW_FEATURES = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4",
]
SENSOR_COLS = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def load_data(path):
    xl = pd.ExcelFile(path)
    train = pd.read_excel(xl, "Training_Data")
    test = pd.read_excel(xl, "Test_Data")
    return train, test


# --------------------------------------------------------------------------- #
# 2. Feature engineering (identical transform applied to train AND test so
#    the pipeline is leak-free and fully reproducible on unseen data)
# --------------------------------------------------------------------------- #
def engineer_features(df, impute_values=None, iso_model=None, fit_iso=False):
    df = df.copy()

    # --- missing-sensor flags (a missing reading is itself informative of a
    #     faulty sensor / corrupted record) ---
    for col in SENSOR_COLS:
        df[col + "_missing"] = df[col].isna().astype(int)

    # --- median imputation (fit on training data only, reused on test) ---
    if impute_values is None:
        impute_values = {c: df[c].median() for c in SENSOR_COLS}
    for c in SENSOR_COLS:
        df[c] = df[c].fillna(impute_values[c])

    # --- physically-implausible readings ---
    # A "temperature rise above ambient" can never be negative, and an
    # exact 0.0 on a live terminal sensor is not physically realistic given
    # the current flowing (S1/S2/S3). These are flagged as hard sensor
    # errors regardless of anything else.
    df["hard_sensor_error"] = (
        (df["Sensor_S1"] <= 0)
        | (df["Sensor_S2"] <= 0)
        | (df["Sensor_S3"] <= 0)
    ).astype(int)

    # --- engineered, target-free (no Reference_Parameter used anywhere
    #     here) sensor-consistency & electrical features ---
    df["S1_minus_S2"] = df["Sensor_S1"] - df["Sensor_S2"]
    df["S1_minus_S3"] = df["Sensor_S1"] - df["Sensor_S3"]
    df["S2_minus_S3"] = df["Sensor_S2"] - df["Sensor_S3"]
    df["current_per_voltage"] = df["Load_Current_A"] / df["Applied_Voltage_kV"]
    df["power_proxy"] = df["Applied_Voltage_kV"] * df["Load_Current_A"]
    df["S_mean"] = df[["Sensor_S1", "Sensor_S2", "Sensor_S3"]].mean(axis=1)
    df["S_spread"] = df[["Sensor_S1", "Sensor_S2", "Sensor_S3"]].std(axis=1)

    # --- multivariate anomaly score (unsupervised, IsolationForest) on the
    #     core operating + sensor variables. Sensor_S4 is intentionally
    #     EXCLUDED (see EDA in the methodology note: near-zero correlation
    #     with the verified Reference_Parameter -> treated as an auxiliary,
    #     largely irrelevant channel, consistent with the brief's hint that
    #     "not every sensor is useful"). This score is only ONE input among
    #     many into the supervised classifier below - it is not used on its
    #     own to declare a record invalid, precisely so that a genuine
    #     operating-regime shift (rare but valid) is not auto-flagged just
    #     for being statistically unusual. ---
    iso_feats = [
        "Applied_Voltage_kV",
        "Load_Current_A",
        "Ambient_Temperature_C",
        "Test_Duration_min",
        "Sensor_S1",
        "Sensor_S2",
        "Sensor_S3",
    ]
    if fit_iso:
        iso_model = IsolationForest(
            n_estimators=300, contamination=0.15, random_state=RANDOM_STATE
        )
        iso_model.fit(df[iso_feats])
    df["anomaly_score"] = -iso_model.score_samples(df[iso_feats])  # higher = more unusual

    return df, impute_values, iso_model


def get_feature_columns():
    return (
        RAW_FEATURES
        + [c + "_missing" for c in SENSOR_COLS]
        + [
            "hard_sensor_error",
            "S1_minus_S2",
            "S1_minus_S3",
            "S2_minus_S3",
            "current_per_voltage",
            "power_proxy",
            "S_mean",
            "S_spread",
            "anomaly_score",
        ]
    )


# --------------------------------------------------------------------------- #
# 3. Task 1 - Validity classifier
# --------------------------------------------------------------------------- #
def train_validity_classifier(train_feat, feat_cols):
    X = train_feat[feat_cols]
    y = (train_feat["Validity_Label"] == "Invalid").astype(int)

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    # 5-fold stratified cross-validation for an honest, reported estimate
    # of generalisation performance (used in the methodology note)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_pred = cross_val_predict(clf, X, y, cv=skf)
    cv_report = classification_report(
        y, cv_pred, target_names=["Valid", "Invalid"], digits=3
    )
    cv_acc = accuracy_score(y, cv_pred)
    cv_f1 = f1_score(y, cv_pred)

    # final model fit on ALL training data
    clf.fit(X, y)

    importances = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(
        ascending=False
    )

    return clf, cv_report, cv_acc, cv_f1, importances


# --------------------------------------------------------------------------- #
# 4. Task 2 - Reference_Parameter regressor
#    Trained on VERIFIED "Valid" records only, since those are the ones
#    engineers confirmed represent reliable test conditions.
# --------------------------------------------------------------------------- #
def train_regressor(train_feat, feat_cols):
    valid_rows = train_feat[train_feat["Validity_Label"] == "Valid"]
    X = valid_rows[feat_cols]
    y = valid_rows["Reference_Parameter"]

    reg = RandomForestRegressor(
        n_estimators=500, max_depth=10, random_state=RANDOM_STATE
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_pred = cross_val_predict(reg, X, y, cv=kf)
    r2 = r2_score(y, cv_pred)
    mae = mean_absolute_error(y, cv_pred)
    rmse = mean_squared_error(y, cv_pred) ** 0.5

    reg.fit(X, y)
    importances = pd.Series(reg.feature_importances_, index=feat_cols).sort_values(
        ascending=False
    )

    return reg, r2, mae, rmse, importances


# --------------------------------------------------------------------------- #
# 5. Task 3 - Automated summary
# --------------------------------------------------------------------------- #
def build_summary(test_out, clf, feat_cols, test_feat, methodology_note):
    n_records = len(test_out)
    n_invalid = int((test_out["Validity_Label"] == "Invalid").sum())

    pred = test_out["Predicted_Reference_Parameter"]
    invalid_proba = clf.predict_proba(test_feat[feat_cols])[:, 1]
    top3_idx = np.argsort(-invalid_proba)[:3]
    top3_ids = test_out.iloc[top3_idx]["Test_ID"].tolist()

    summary = {
        "records_analysed": int(n_records),
        "abnormal_invalid_records_identified": n_invalid,
        "predicted_reference_parameter_min": round(float(pred.min()), 4),
        "predicted_reference_parameter_max": round(float(pred.max()), 4),
        "predicted_reference_parameter_average": round(float(pred.mean()), 4),
        "test_ids_requiring_highest_attention": top3_ids,
        "approach_summary": methodology_note,
    }
    return summary


METHOD_NOTE_TEXT = (
    "We engineered target-free features (sensor differences, current/voltage "
    "ratio, missing-sensor flags, hard physical-range checks, and an "
    "IsolationForest anomaly score) so the pipeline works identically on "
    "unseen test data. A Random Forest classifier, trained on engineer-"
    "verified Valid/Invalid labels, learns to separate genuine operating-"
    "regime shifts from true sensor faults, rather than flagging statistical "
    "outliers alone. A second Random Forest, trained only on Valid records, "
    "predicts the Reference_Parameter. Sensor_S4 was down-weighted after EDA "
    "showed near-zero correlation with ground truth."
)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx",
        help="Path to the provided dataset workbook",
    )
    parser.add_argument("--team", default="Tabahi", help="Team name for output naming")
    parser.add_argument("--outdir", default="output", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1. Load
    train, test = load_data(args.input)

    # 2. Feature engineering (fit transformers on TRAIN only)
    train_feat, impute_values, iso_model = engineer_features(train, fit_iso=True)
    test_feat, _, _ = engineer_features(
        test, impute_values=impute_values, iso_model=iso_model, fit_iso=False
    )
    feat_cols = get_feature_columns()

    # 3. Task 1: validity classifier
    clf, cv_report, cv_acc, cv_f1, clf_importance = train_validity_classifier(
        train_feat, feat_cols
    )
    test_validity = clf.predict(test_feat[feat_cols])
    test_feat["Validity_Label"] = np.where(test_validity == 1, "Invalid", "Valid")

    # also record which TRAIN rows the model flags, for the report
    train_feat["Validity_Pred"] = np.where(
        clf.predict(train_feat[feat_cols]) == 1, "Invalid", "Valid"
    )

    # 4. Task 2: reference-parameter regressor (trained on Valid rows only)
    reg, r2, mae, rmse, reg_importance = train_regressor(train_feat, feat_cols)
    test_feat["Predicted_Reference_Parameter"] = reg.predict(test_feat[feat_cols])

    # 5. Assemble required submission file
    test_out = test_feat[
        ["Test_ID", "Predicted_Reference_Parameter", "Validity_Label"]
    ].copy()
    test_out["Predicted_Reference_Parameter"] = test_out[
        "Predicted_Reference_Parameter"
    ].round(4)

    submission_path = os.path.join(args.outdir, f"{args.team}.csv")
    test_out.to_csv(submission_path, index=False)

    # 6. Task 3: automated summary
    summary = build_summary(test_out, clf, feat_cols, test_feat, METHOD_NOTE_TEXT)
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # flat CSV version too
    flat = {
        "Records_Analysed": summary["records_analysed"],
        "Abnormal_Invalid_Records_Identified": summary[
            "abnormal_invalid_records_identified"
        ],
        "Min_Predicted_Reference_Parameter": summary[
            "predicted_reference_parameter_min"
        ],
        "Max_Predicted_Reference_Parameter": summary[
            "predicted_reference_parameter_max"
        ],
        "Average_Predicted_Reference_Parameter": summary[
            "predicted_reference_parameter_average"
        ],
        "Test_ID_Highest_Attention_1": summary["test_ids_requiring_highest_attention"][
            0
        ],
        "Test_ID_Highest_Attention_2": summary["test_ids_requiring_highest_attention"][
            1
        ],
        "Test_ID_Highest_Attention_3": summary["test_ids_requiring_highest_attention"][
            2
        ],
        "Approach_Summary": summary["approach_summary"],
    }
    pd.DataFrame([flat]).to_csv(os.path.join(args.outdir, "summary.csv"), index=False)

    # 7. Model performance report (used to write the methodology note)
    report_lines = []
    report_lines.append("=== Task 1: Validity Classifier (5-fold CV on Training_Data) ===")
    report_lines.append(f"Accuracy: {cv_acc:.4f}   F1 (Invalid class): {cv_f1:.4f}")
    report_lines.append(cv_report)
    report_lines.append("Top feature importances (classifier):")
    report_lines.append(clf_importance.head(8).to_string())
    report_lines.append("")
    report_lines.append("=== Task 2: Reference_Parameter Regressor (5-fold CV on Valid rows) ===")
    report_lines.append(f"R2: {r2:.4f}   MAE: {mae:.4f}   RMSE: {rmse:.4f}")
    report_lines.append("Top feature importances (regressor):")
    report_lines.append(reg_importance.head(8).to_string())
    report_lines.append("")
    report_lines.append("=== Test set outcome ===")
    report_lines.append(f"Records analysed: {summary['records_analysed']}")
    report_lines.append(
        f"Predicted invalid on test set: {summary['abnormal_invalid_records_identified']}"
    )
    report_lines.append(
        f"Predicted_Reference_Parameter range: "
        f"[{summary['predicted_reference_parameter_min']}, "
        f"{summary['predicted_reference_parameter_max']}], "
        f"avg={summary['predicted_reference_parameter_average']}"
    )
    report_lines.append(
        f"Top-3 attention Test IDs: {summary['test_ids_requiring_highest_attention']}"
    )

    with open(os.path.join(args.outdir, "model_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"\nWritten: {submission_path}")
    print(f"Written: {os.path.join(args.outdir, 'summary.json')}")
    print(f"Written: {os.path.join(args.outdir, 'summary.csv')}")


if __name__ == "__main__":
    main()
