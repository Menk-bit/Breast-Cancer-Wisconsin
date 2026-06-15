# Breast Cancer Classification

Five classic machine-learning algorithms evaluated consistently on Kaggle
dataset `uciml/breast-cancer-wisconsin-data`:

- K-nearest neighbors
- Logistic regression
- Support vector machine
- Random Forest
- XGBoost

## Consistent Experimental Protocol

`Preprocessing.py` is the only data preparation entry point. It removes the
identifier and empty column, maps benign/malignant labels to `0/1`, and
persists one stratified 70/30 train-test split with random seed `42`.

Every algorithm:

1. Uses the same 70% training and 30% held-out test samples.
2. Runs stratified 5-fold cross-validation only on the training portion.
3. Selects hyperparameters by recall, then F1, ROC-AUC, and accuracy.
4. Fits the selected configuration on all training data.
5. Saves the model, CV table, test metrics, and result plot under `artifacts/`.

The test set is never used for hyperparameter selection.

## Run

```powershell
..\venv\Scripts\python.exe -m pip install -r requirements.txt
..\venv\Scripts\python.exe Preprocessing.py
..\venv\Scripts\python.exe RunAll.py --force-train
```

Later runs load the saved models:

```powershell
..\venv\Scripts\python.exe RunAll.py
```

Each model can also be run individually, for example:

```powershell
..\venv\Scripts\python.exe SVM.py --force-train
```

Candidate hyperparameter grids are declared near the top of each model script
for later expansion.

Original implementations extracted from the two unrelated repository histories
are retained under `legacy/`.
