# IT3190E-ML-PROJECT2025.2

This assembled branch combines the unrelated `main` and `master` histories:

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

The newer KNN and logistic scripts use `model_ready_scaled.csv`. The SVM
experiment suites select either the scaled or tree-ready file as appropriate.
The original `KNN/` and `Logistic/` folders are retained as legacy Wisconsin
dataset implementations; use `KNN new/` and `Logistic new/` for the root SEER
dataset.
