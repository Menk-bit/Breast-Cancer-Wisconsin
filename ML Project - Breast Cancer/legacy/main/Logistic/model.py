import numpy as np


class LogisticRegressionScratch:
    def __init__(
        self,
        learning_rate=0.05,
        epochs=300,
        C=1.0,
        class_weight=None,
        print_every=50,
        verbose=False,
    ):
        if C <= 0:
            raise ValueError("C phải > 0.")

        if class_weight not in (None, "balanced"):
            raise ValueError("class_weight chỉ được là None hoặc 'balanced'.")

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C
        self.class_weight = class_weight
        self.print_every = print_every
        self.verbose = verbose

        self.w = None
        self.b = 0.0
        self.history = []
        self.n_train_samples = None

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_sample_weight(self, y):
        y = np.asarray(y)

        if self.class_weight is None:
            return np.ones(y.shape[0])

        n_samples = y.shape[0]
        n_class_0 = np.sum(y == 0)
        n_class_1 = np.sum(y == 1)

        if n_class_0 == 0 or n_class_1 == 0:
            raise ValueError("Mỗi fold phải có đủ hai lớp 0 và 1.")

        weight_0 = n_samples / (2 * n_class_0)
        weight_1 = n_samples / (2 * n_class_1)

        return np.where(y == 1, weight_1, weight_0)

    def compute_loss(self, y_true, y_proba, sample_weight=None):
        y_true = np.asarray(y_true)
        y_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)

        bce = -(
            y_true * np.log(y_proba)
            + (1 - y_true) * np.log(1 - y_proba)
        )

        if sample_weight is None:
            data_loss = np.mean(bce)
        else:
            data_loss = np.sum(sample_weight * bce) / np.sum(sample_weight)

        l2_loss = np.sum(self.w ** 2) / (2 * self.C * self.n_train_samples)

        return data_loss + l2_loss

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        n_samples, n_features = X_train.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.history = []
        self.n_train_samples = n_samples

        sample_weight = self.compute_sample_weight(y_train)
        weight_sum = np.sum(sample_weight)

        for epoch in range(1, self.epochs + 1):
            y_train_proba = self.predict_proba(X_train)

            error = y_train_proba - y_train
            weighted_error = sample_weight * error

            dw = X_train.T @ weighted_error / weight_sum
            dw += self.w / (self.C * n_samples)

            db = np.sum(weighted_error) / weight_sum

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            train_loss = self.compute_loss(
                y_train,
                self.predict_proba(X_train),
                sample_weight,
            )

            valid_loss = np.nan

            if X_valid is not None and y_valid is not None:
                X_valid = np.asarray(X_valid, dtype=float)
                y_valid = np.asarray(y_valid, dtype=float)

                valid_loss = self.compute_loss(
                    y_valid,
                    self.predict_proba(X_valid),
                )

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                }
            )

            if self.verbose and (epoch == 1 or epoch % self.print_every == 0):
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train Loss = {train_loss:.6f} | "
                    f"Valid Loss = {valid_loss:.6f}"
                )

        return self

    def predict_proba(self, X):
        if self.w is None:
            raise RuntimeError("Model chưa được fit.")

        X = np.asarray(X, dtype=float)
        z = X @ self.w + self.b

        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_history(self):
        return list(self.history)