# IT3190E-ML-PROJECT2025.2

This branch combines the original project histories and keeps all Python source
code in `src/`.

- `main`: preprocessing, KNN, logistic regression, linear regression, and
  Naive Bayes work.
- `master`: SVM implementations, experiment suites, and generated results.

## Local data

The repository uses the existing files in `data/`:

- `data/preprocessed_breast_cancer.csv`
- `data/model_ready_tree.csv`
- `data/model_ready_scaled.csv`

The data directory is intentionally ignored by Git so these large local files
are not replaced or committed during branch assembly.

The model scripts use the scaled or tree-ready file as appropriate.

## Shared Method

All model runners use:

- `src/data_splitting.py` for stratified 60% train, 20% validation, and 20%
  test splits.
- `src/metrics_utils.py` for Accuracy, Precision Class 0, Precision Class 1,
  Recall Class 0, Recall Class 1, F1 Class 0, F1 Class 1, and ROC AUC.

The test set is used only for final evaluation. Validation data selects model
settings and classification thresholds.

## Model Commands

```powershell
python src/knn.py
python src/logistic_regression.py
python src/lr_naive_bayes.py
python src/seer_compact_two_custom_svm_vs_sklearn.py
python src/tree_ensemble_models.py
```

## Preprocessing

Preprocessing code is grouped under `src/preprocess/`:

```powershell
python src/preprocess/clean.py
python src/preprocess/encode.py
python src/preprocess/feature_selection.py
python src/preprocess/selected.py
```

## Random Forest and XGBoost

Train and evaluate both tree ensemble models on the local tree-ready dataset:

The tree ensemble command writes models, standard metrics, test predictions,
validation threshold results, and feature importance tables to
`tree_ensemble_outputs/`.

For a faster smoke run:

```powershell
python src/tree_ensemble_models.py --sample-size 20000
```
