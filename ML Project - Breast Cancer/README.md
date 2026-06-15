# Five-Year Breast Cancer Survival Classification

Seven machine-learning algorithms are evaluated consistently on the
200,000-row model-ready breast-cancer survival dataset:

- K-nearest neighbors
- Logistic regression
- Gradient-descent logistic regression from `LR + NB/v03.ipynb`
- Gaussian Naive Bayes from `LR + NB/v03.ipynb`
- Linear SVM trained only with SGD
- Random Forest
- XGBoost

## Consistent Experimental Protocol

`src/Preprocessing.py` is the shared data entry point. The binary target is
`survive_after_5`, where `1` means the patient survived at least five years.
It creates one deterministic stratified 70/30 split with random seed `42`.

KNN, both logistic implementations, Naive Bayes, and SVM use
`ML Project - Breast Cancer/data/model_ready_scaled.csv`. Random Forest and
XGBoost use `ML Project - Breast Cancer/data/model_ready_tree.csv`.

Every algorithm:

1. Uses the same 70% training and 30% held-out test samples.
2. Runs stratified 5-fold cross-validation only on the training portion.
3. Selects hyperparameters by recall, then F1, ROC-AUC, and accuracy.
4. Fits the selected configuration on all training data.
5. Saves the model, CV table, test metrics, and result plot under the root
   `artifacts/` directory.

The test set is never used for hyperparameter selection.

## Run

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe src\Preprocessing.py
venv\Scripts\python.exe src\RunAll.py --force-train
```

Later runs load the saved models:

```powershell
venv\Scripts\python.exe src\RunAll.py
```

Each model can also be run individually, for example:

```powershell
venv\Scripts\python.exe src\SVM.py --force-train
```

Candidate hyperparameter grids are declared near the top of each model script
for later expansion.

The incoming `Preprocess/` folder is copied unchanged from the IT3190E
repository. Unified training code lives under `src/`.
