# CPRI State-Level Hackathon - Screening Round Submission

**Team:** Tabahi  

## Overview
This repository contains the end-to-end reproducible machine learning pipeline and screening round submission files for the CPRI State-Level Hackathon.

The solution addresses:
- **Task 1:** Identify abnormal and invalid test records using engineered physical features and an anomaly detection classifier.
- **Task 2:** Accurately predict the `Reference_Parameter` (hot-spot temperature rise) on verified valid test records.
- **Task 3:** Automated test summary generation (`summary.json` & `summary.csv`).

---

## Headline Results (5-Fold Cross-Validation on Training Data)

| Metric / Task | Result |
|---|---|
| **Task 1: Validity Classifier** | Accuracy: **96.5%**, F1-score (Invalid class): **0.8511** |
| **Task 2: Reference_Parameter Regressor** | $R^2$: **0.9866**, MAE: **0.7316**, RMSE: **1.2450** |
| **Test Set Predictions (350 records)** | **37** flagged Invalid; Predicted range: `[12.97, 55.45]`, Avg: `26.29` |

---

## Repository Contents

```text
├── README.md                                         <- Repository documentation
├── requirements.txt                                  <- Python dependencies
├── .gitignore                                        <- Standard Git ignore rules
└── Tabahi_submission/
    ├── CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx <- Dataset workbook
    ├── Methodology_Note_Tabahi.docx                  <- 2-page methodology note
    ├── README.txt                                    <- Original submission instructions
    ├── solution.py                                   <- End-to-end ML pipeline script
    ├── Tabahi.csv                                    <- Required prediction file
    ├── model_report.txt                              <- Model metrics & feature importances
    ├── summary.json                                  <- Task 3 summary (JSON)
    ├── summary.csv                                   <- Task 3 summary (CSV)
    └── output/                                       <- Auto-generated output directory
```

---

## Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Solution Pipeline
Navigate to `Tabahi_submission/` (or specify paths) and run:

```bash
cd Tabahi_submission
python solution.py --input CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx --team Tabahi
```

This regenerates:
- `output/Tabahi.csv` (predictions with `Test_ID`, `Predicted_Reference_Parameter`, `Validity_Label`)
- `output/summary.json` & `output/summary.csv`
- `output/model_report.txt`
