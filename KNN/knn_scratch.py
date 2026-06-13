import numpy as np


class KNNClassifierScratch:
    def __init__(
        self,
        k=5,
        distance_metric="euclidean",
        weights="uniform",
        tie_break=1,
    ):
        if k <= 0:
            raise ValueError("k phải > 0.")

        if distance_metric not in ("euclidean", "manhattan"):
            raise ValueError("distance_metric phải là 'euclidean' hoặc 'manhattan'.")

        if weights not in ("uniform", "distance"):
            raise ValueError("weights phải là 'uniform' hoặc 'distance'.")

        if tie_break not in (0, 1):
            raise ValueError("tie_break phải là 0 hoặc 1.")

        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.tie_break = tie_break

        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=int)

        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X_train và y_train không cùng số mẫu.")

        return self

    def _compute_distances(self, X_test):
        X_test = np.asarray(X_test, dtype=float)

        if self.distance_metric == "euclidean":
            X_test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)
            X_train_sq = np.sum(self.X_train ** 2, axis=1).reshape(1, -1)

            distances_sq = X_test_sq + X_train_sq - 2 * X_test @ self.X_train.T
            distances_sq = np.maximum(distances_sq, 0)

            return np.sqrt(distances_sq)

        return np.sum(
            np.abs(X_test[:, None, :] - self.X_train[None, :, :]),
            axis=2,
        )

    def _get_neighbors(self, X_test):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model chưa được fit.")

        distances = self._compute_distances(X_test)
        k_eff = min(self.k, self.X_train.shape[0])

        neighbor_indices = np.argpartition(
            distances,
            kth=k_eff - 1,
            axis=1,
        )[:, :k_eff]

        row_indices = np.arange(distances.shape[0])[:, None]

        neighbor_distances = distances[row_indices, neighbor_indices]
        neighbor_labels = self.y_train[neighbor_indices]

        return neighbor_labels, neighbor_distances

    def _predict_uniform(self, neighbor_labels):
        pos_count = np.sum(neighbor_labels == 1, axis=1)
        neg_count = np.sum(neighbor_labels == 0, axis=1)

        y_pred = np.where(pos_count > neg_count, 1, 0)

        tie_mask = pos_count == neg_count
        y_pred[tie_mask] = self.tie_break

        return y_pred

    def _predict_distance_weighted(self, neighbor_labels, neighbor_distances):
        eps = 1e-9
        voting_weights = 1.0 / (neighbor_distances + eps)

        pos_score = np.sum(voting_weights * (neighbor_labels == 1), axis=1)
        neg_score = np.sum(voting_weights * (neighbor_labels == 0), axis=1)

        y_pred = np.where(pos_score > neg_score, 1, 0)

        tie_mask = pos_score == neg_score
        y_pred[tie_mask] = self.tie_break

        return y_pred

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=float)

        neighbor_labels, neighbor_distances = self._get_neighbors(X_test)

        if self.weights == "uniform":
            return self._predict_uniform(neighbor_labels)

        return self._predict_distance_weighted(
            neighbor_labels,
            neighbor_distances,
        )