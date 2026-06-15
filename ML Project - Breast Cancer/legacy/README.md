# Extracted Historical Implementations

These files preserve the algorithm implementations from the nested repository's
two unrelated Git histories before that repository was removed.

- `main/`: commit `56eea99`, containing scratch KNN and Logistic Regression.
- `master/`: commit `21e39c6` (`origin/master`), containing the SVM comparison
  implementation.

The historical files retain their original preprocessing and evaluation logic.
Use the root-level scripts for consistent data splitting, K-fold tuning, model
selection, persistence, and held-out evaluation.
