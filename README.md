# CPRI State-Level Hackathon - Screening Round Submission

**Team:** Tabahi  
**Project:** Transformer Hot-Spot Temperature Rise AI & Anomaly Detection Dashboard

---

## Interactive Intelligence Dashboard

An interactive web frontend dashboard built with **Flask**, **Tailwind CSS**, and **Chart.js**.

### Key Features
- **Executive KPI Cards:** Real-time metrics for total test records, validity ratios, temperature ranges, and CV performance.
- **Interactive "What-If" Simulator:** Live sliders for electrical operating conditions ($V, I, T_{amb}, t$) and terminal sensors ($S_1, S_2, S_3, S_4$) with real-time AI classification and temperature rise prediction.
- **Visual Analytics:** Interactive Chart.js graphs for temperature rise distribution, multivariate anomaly score vs. sensor spread, and feature importance rankings for both the Classifier and Regressor.
- **Dataset Explorer:** Searchable, filterable table for all 350 screening test records with status pills and instant export/download options.
- **One-Click Artifact Export:** Instant download for `Tabahi.csv`, `summary.json`, `summary.csv`, and `model_report.txt`.

### How to Run the Web Dashboard
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the web application
python app.py
```
Open your browser at **`http://localhost:5000`**.

---

## Headline ML Results (5-Fold Cross-Validation on Training Data)

| Metric / Task | Result |
|---|---|
| **Task 1: Validity Classifier** | Accuracy: **96.5%**, F1-score (Invalid class): **0.8511** |
| **Task 2: Reference_Parameter Regressor** | $R^2$: **0.9866**, MAE: **0.7316**, RMSE: **1.2450** |
| **Test Set Predictions (350 records)** | **37** flagged Invalid; Predicted range: `[12.97, 55.45]`, Avg: `26.29` |

---

## Repository Contents

```text
├── app.py                                            <- Flask dashboard backend & live inference API
├── templates/
│   └── index.html                                    <- Modern glassmorphism dashboard UI
├── README.md                                         <- Repository documentation
├── requirements.txt                                  <- Python dependencies (Flask, pandas, scikit-learn, etc.)
├── .gitignore                                        <- Git ignore rules
└── Tabahi_submission/
    ├── CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx <- Dataset workbook
    ├── Methodology_Note_Tabahi.docx                  <- 2-page methodology note
    ├── README.txt                                    <- Original submission instructions
    ├── solution.py                                   <- End-to-end ML pipeline script (CLI)
    ├── Tabahi.csv                                    <- Required prediction file
    ├── model_report.txt                              <- Model metrics & feature importances
    ├── summary.json                                  <- Task 3 summary (JSON)
    ├── summary.csv                                   <- Task 3 summary (CSV)
    └── output/                                       <- Auto-generated output directory
```

---

## Running the CLI Pipeline Directly

To re-run the pure CLI pipeline without the web interface:

```bash
cd Tabahi_submission
python solution.py --input CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx --team Tabahi
```

This regenerates:
- `output/Tabahi.csv`
- `output/summary.json` & `output/summary.csv`
- `output/model_report.txt`
