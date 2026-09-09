Team: Tabahi
CPRI State-Level Hackathon - Screening Round Submission

CONTENTS
--------
1. Tabahi.csv                 -> Required prediction file (Test_ID, Predicted_Reference_Parameter, Validity_Label)
2. summary.json / summary.csv -> Task 3 automated test summary
3. solution.py                -> Complete, end-to-end, reproducible source code (Tasks 1-3)
4. model_report.txt           -> Full cross-validation performance metrics
5. Methodology_Note_Tabahi.docx -> 2-page methodology note (approach, key parameters,
                                    anomaly-detection method, assumptions, digital-twin roadmap)
6. CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx -> original dataset, included for
                                    convenience so the script can be re-run as-is.

HOW TO RUN
----------
Requires Python 3.9+ with: pandas, numpy, scikit-learn, openpyxl

    pip install pandas numpy scikit-learn openpyxl
    python solution.py --input CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx --team Tabahi

This regenerates Tabahi.csv, summary.json, summary.csv and model_report.txt inside an
"output/" folder, with no manual editing of any individual record at any step.

HEADLINE RESULTS (5-fold cross-validation on Training_Data)
-------------------------------------------------------------
Task 1 - Validity classifier : Accuracy 96.5%, F1 (Invalid class) 0.85
Task 2 - Reference_Parameter regressor : R2 = 0.987, MAE = 0.73, RMSE = 1.25

Test set (350 records): 37 flagged Invalid; Predicted_Reference_Parameter
ranges 12.97-55.45 (avg 26.29).
