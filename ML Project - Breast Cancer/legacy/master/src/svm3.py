import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

path = "./data/data.csv"

# ═══════════════════════════════════════════════════════════
# DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['id'])
    df["diagnosis"] = df["diagnosis"].map({'M': 1, 'B': 0})
    if df.isnull().sum().sum() == 0:
        print("No missing values in the dataset.")
    else:
        df = df.fillna(df.median())
    return df


def stratified_split(df: pd.DataFrame, target_col: str,
                     test_size: float, random_state: int):
    """Chia tập train/test theo tỉ lệ lớp (stratified)."""
    B_data = df[df[target_col] == 0].sample(frac=1, random_state=random_state)
    M_data = df[df[target_col] == 1].sample(frac=1, random_state=random_state)

    num_B_test = int(len(B_data) * test_size)
    num_M_test = int(len(M_data) * test_size)

    training_set = pd.concat([B_data.iloc[num_B_test:], M_data.iloc[num_M_test:]])
    testing_set  = pd.concat([B_data.iloc[:num_B_test], M_data.iloc[:num_M_test]])

    training_set = training_set.sample(frac=1, random_state=random_state).reset_index(drop=True)
    testing_set  = testing_set.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Training size : {training_set.shape[0]}")
    print(f"Testing  size : {testing_set.shape[0]}")
    return training_set, testing_set


def k_fold_cross_validation(training_set: pd.DataFrame, k: int = 5):
    """K-fold: chia đều n mẫu thành k phần không chồng lấp."""
    n         = len(training_set)
    fold_size = n // k
    folds     = []
    for i in range(k):
        start = i * fold_size
        end   = start + fold_size if i < k - 1 else n
        folds.append(training_set.iloc[start:end].reset_index(drop=True))
    return folds


def normalize_features(X_train: np.ndarray, X_other: np.ndarray):
    """Z-score normalization — fit trên train, transform cả hai."""
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1
    return (X_train - mean) / std, (X_other - mean) / std, mean, std


# ═══════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    accuracy  = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return dict(accuracy=accuracy, precision=precision,
                recall=recall, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)


# ═══════════════════════════════════════════════════════════
# KERNEL FUNCTIONS
# ═══════════════════════════════════════════════════════════

def linear_kernel(x1, x2):
    return float(np.dot(x1, x2))

def polynomial_kernel(x1, x2, degree=3, coef0=1.0):
    return float((np.dot(x1, x2) + coef0) ** degree)

def gaussian_kernel(x1, x2, gamma=0.05):
    diff = x1 - x2
    return float(np.exp(-gamma * np.dot(diff, diff)))

def build_kernel_matrix(X1, X2, kernel_fn):
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    symmetric = (n1 == n2) and np.array_equal(X1, X2)
    if symmetric:
        for i in range(n1):
            for j in range(i, n2):
                v = kernel_fn(X1[i], X2[j])
                K[i, j] = v
                K[j, i] = v
    else:
        for i in range(n1):
            for j in range(n2):
                K[i, j] = kernel_fn(X1[i], X2[j])
    return K


# ═══════════════════════════════════════════════════════════
# LINEAR SVM — SGD (primal, from scratch)
# ═══════════════════════════════════════════════════════════

def train_sgd_svm(X: np.ndarray, y: np.ndarray,
                  learning_rate: float = 0.001,
                  lambda_param: float  = 0.001,
                  n_epochs: int        = 2000) -> tuple:
    """
    Linear SVM giải bài toán primal bằng SGD.
    Mỗi epoch shuffle dữ liệu và dùng learning rate decay.

    Bài toán primal:
        min  ½||w||² + C·Σ max(0, 1 - yᵢ(w·xᵢ + b))
    Gradient:
        ngoài margin (yᵢf(xᵢ) ≥ 1): dw = λw,       db = 0
        trong margin / sai:           dw = λw - yᵢxᵢ, db = -yᵢ
    """
    n_samples, n_features = X.shape
    y_ = np.where(y == 0, -1, 1)
    w  = np.zeros(n_features)
    b  = 0.0

    for epoch in range(n_epochs):
        # Learning rate decay + shuffle mỗi epoch
        lr      = learning_rate / (1 + lambda_param * epoch)
        indices = np.random.permutation(n_samples)
        for idx in indices:
            x_i = X[idx]
            if y_[idx] * (x_i @ w + b) >= 1:
                dw = lambda_param * w
                db = 0.0
            else:
                dw = lambda_param * w - y_[idx] * x_i
                db = -y_[idx]
            w -= lr * dw
            b -= lr * db
    return w, b


def predict_sgd_svm(X, w, b):
    return np.where(np.sign(X @ w + b) == -1, 0, 1)


# ═══════════════════════════════════════════════════════════
# SVM — SMO (dual, from scratch)
# ═══════════════════════════════════════════════════════════

class SMO_SVM:
    """
    SVM huấn luyện bằng SMO (Platt, 1998) — giải bài toán dual.
        max  Σαᵢ − ½ ΣᵢΣⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
        s.t. 0 ≤ αᵢ ≤ C,   Σαᵢyᵢ = 0
    Hỗ trợ mọi kernel.
    """

    def __init__(self, kernel_fn, C=1.0, tol=1e-3, max_iter=200):
        self.kernel_fn = kernel_fn
        self.C         = C
        self.tol       = tol
        self.max_iter  = max_iter

    def _decision(self, K_col):
        return float((self.alphas_ * self.y_) @ K_col + self.b_)

    def _bounds(self, i, j):
        if self.y_[i] != self.y_[j]:
            L = max(0.0, self.alphas_[j] - self.alphas_[i])
            H = min(self.C, self.C + self.alphas_[j] - self.alphas_[i])
        else:
            L = max(0.0, self.alphas_[i] + self.alphas_[j] - self.C)
            H = min(self.C, self.alphas_[i] + self.alphas_[j])
        return L, H

    def _optimize_pair(self, i, j, K):
        if i == j:
            return False
        Ei = self._decision(K[:, i]) - self.y_[i]
        Ej = self._decision(K[:, j]) - self.y_[j]
        ai_old, aj_old = self.alphas_[i], self.alphas_[j]
        L, H = self._bounds(i, j)
        if L >= H:
            return False
        eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
        if eta >= 0:
            return False
        aj_new = float(np.clip(aj_old - self.y_[j] * (Ei - Ej) / eta, L, H))
        if abs(aj_new - aj_old) < 1e-5:
            return False
        ai_new = ai_old + self.y_[i] * self.y_[j] * (aj_old - aj_new)
        di, dj = ai_new - ai_old, aj_new - aj_old
        b1 = self.b_ - Ei - self.y_[i]*di*K[i,i] - self.y_[j]*dj*K[i,j]
        b2 = self.b_ - Ej - self.y_[i]*di*K[i,j] - self.y_[j]*dj*K[j,j]
        if 0 < ai_new < self.C:
            self.b_ = b1
        elif 0 < aj_new < self.C:
            self.b_ = b2
        else:
            self.b_ = (b1 + b2) / 2.0
        self.alphas_[i] = ai_new
        self.alphas_[j] = aj_new
        return True

    def fit(self, X, y):
        n = len(X)
        self.y_      = np.where(y == 0, -1.0, 1.0)
        self.alphas_ = np.zeros(n)
        self.b_      = 0.0
        self.X_      = X
        print(f"    Building kernel matrix ({n}×{n}) …", flush=True)
        K = build_kernel_matrix(X, X, self.kernel_fn)

        entire_pass       = True
        passes_no_change  = 0

        for _ in range(self.max_iter):
            changed = 0
            indices = range(n) if entire_pass else [
                i for i in range(n) if 0 < self.alphas_[i] < self.C
            ]
            for i in indices:
                Ei = self._decision(K[:, i]) - self.y_[i]
                kkt_ok = (
                    (self.y_[i]*Ei >= -self.tol and self.alphas_[i] <= self.C) and
                    (self.y_[i]*Ei <=  self.tol and self.alphas_[i] >= 0)
                )
                if kkt_ok:
                    continue
                # Heuristic: chọn j có |Ei-Ej| lớn nhất
                errors = np.abs(
                    (self.alphas_ * self.y_) @ K + self.b_ - self.y_ - Ei
                )
                errors[i] = -1
                j = int(np.argmax(errors))
                if self._optimize_pair(i, j, K):
                    changed += 1
                else:
                    for j in np.random.permutation(n):
                        if j != i and self._optimize_pair(i, j, K):
                            changed += 1
                            break
            if changed == 0:
                if entire_pass:
                    passes_no_change += 1
                    if passes_no_change >= 3:
                        break
                entire_pass = not entire_pass
            else:
                passes_no_change = 0
                entire_pass = not entire_pass

        sv_mask         = self.alphas_ > 1e-5
        self.sv_alphas_ = self.alphas_[sv_mask]
        self.sv_y_      = self.y_[sv_mask]
        self.sv_X_      = X[sv_mask]
        self.n_sv_      = int(sv_mask.sum())
        print(f"    Support vectors: {self.n_sv_} / {n}", flush=True)
        return self

    def decision_function(self, X):
        K_pred = build_kernel_matrix(X, self.sv_X_, self.kernel_fn)
        return K_pred @ (self.sv_alphas_ * self.sv_y_) + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, 0)


# ═══════════════════════════════════════════════════════════
# PIPELINE HELPERS
# ═══════════════════════════════════════════════════════════

def _make_result(label, impl, kernel_name, cv_scores, test_metrics,
                 train_time, n_sv=None):
    return {
        "label"       : label,
        "impl"        : impl,          # "SGD" | "SMO" | "sklearn"
        "kernel"      : kernel_name,
        "cv_acc_mean" : float(np.mean(cv_scores)),
        "cv_acc_std"  : float(np.std(cv_scores)),
        "cv_scores"   : cv_scores,
        "train_time"  : train_time,
        "n_sv"        : n_sv if n_sv is not None else "—",
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "confusion"   : np.array([[test_metrics["tn"], test_metrics["fp"]],
                                   [test_metrics["fn"], test_metrics["tp"]]]),
    }


# ─── 1. SGD linear (primal) ─────────────────────────────────

def run_sgd_linear(training_set, testing_set, k=5,
                   lr=0.001, lambda_param=0.001, n_epochs=2000):
    print(f"\n{'═'*55}")
    print(f"  [1] Linear SVM — SGD (primal, from scratch)")
    print(f"{'═'*55}")
    target_col = "diagnosis"
    folds      = k_fold_cross_validation(training_set, k=k)
    cv_scores  = []
    t0         = time.time()

    for i in range(k):
        val_fold   = folds[i]
        train_fold = pd.concat([folds[j] for j in range(k) if j != i], ignore_index=True)
        X_tr = train_fold.drop(columns=[target_col]).values
        y_tr = train_fold[target_col].values
        X_vl = val_fold.drop(columns=[target_col]).values
        y_vl = val_fold[target_col].values
        X_tr_n, X_vl_n, _, _ = normalize_features(X_tr, X_vl)
        print(f"  Fold {i+1}/{k} …", end=" ", flush=True)
        w, b   = train_sgd_svm(X_tr_n, y_tr, lr, lambda_param, n_epochs)
        y_pred = predict_sgd_svm(X_vl_n, w, b)
        m      = compute_metrics(y_vl, y_pred)
        cv_scores.append(m["accuracy"])
        print(f"acc={m['accuracy']:.4f}")

    X_tr_full = training_set.drop(columns=[target_col]).values
    y_tr_full = training_set[target_col].values
    X_te      = testing_set.drop(columns=[target_col]).values
    y_te      = testing_set[target_col].values
    X_tr_n, X_te_n, _, _ = normalize_features(X_tr_full, X_te)
    print("  Training final model …", flush=True)
    w, b      = train_sgd_svm(X_tr_n, y_tr_full, lr, lambda_param, n_epochs)
    y_pred_te = predict_sgd_svm(X_te_n, w, b)
    train_time = time.time() - t0
    test_m = compute_metrics(y_te, y_pred_te)
    _print_result(test_m, np.mean(cv_scores), np.std(cv_scores))
    return _make_result("SGD Linear", "SGD", "Linear (primal)",
                        cv_scores, test_m, train_time)


# ─── 2–4. SMO (dual, from scratch) ──────────────────────────

def run_smo(training_set, testing_set, svm_type="linear",
            k=5, C=1.0, degree=3, coef0=1.0, gamma=0.05):
    idx_map   = {"linear": 2, "polynomial": 3, "gaussian": 4}
    name_map  = {
        "linear"    : "Linear SVM  — SMO (dual, from scratch)  kernel: x·y",
        "polynomial": f"Poly SVM    — SMO (dual, from scratch)  kernel: (x·y+{coef0})^{degree}",
        "gaussian"  : f"Gaussian SVM— SMO (dual, from scratch)  kernel: RBF γ={gamma}",
    }
    kname_map = {
        "linear"    : "Linear (dual)",
        "polynomial": f"Poly d={degree}",
        "gaussian"  : f"RBF γ={gamma}",
    }
    print(f"\n{'═'*55}")
    print(f"  [{idx_map[svm_type]}] {name_map[svm_type]}")
    print(f"{'═'*55}")

    if svm_type == "linear":
        kfn = linear_kernel
    elif svm_type == "polynomial":
        kfn = lambda a, b: polynomial_kernel(a, b, degree, coef0)
    else:
        kfn = lambda a, b: gaussian_kernel(a, b, gamma)

    target_col = "diagnosis"
    folds      = k_fold_cross_validation(training_set, k=k)
    cv_scores  = []
    t0         = time.time()

    for i in range(k):
        val_fold   = folds[i]
        train_fold = pd.concat([folds[j] for j in range(k) if j != i], ignore_index=True)
        X_tr = train_fold.drop(columns=[target_col]).values
        y_tr = train_fold[target_col].values
        X_vl = val_fold.drop(columns=[target_col]).values
        y_vl = val_fold[target_col].values
        X_tr_n, X_vl_n, _, _ = normalize_features(X_tr, X_vl)
        print(f"  Fold {i+1}/{k} …", end=" ", flush=True)
        model  = SMO_SVM(kfn, C=C).fit(X_tr_n, y_tr)
        y_pred = model.predict(X_vl_n)
        m      = compute_metrics(y_vl, y_pred)
        cv_scores.append(m["accuracy"])
        print(f"acc={m['accuracy']:.4f}")

    X_tr_full = training_set.drop(columns=[target_col]).values
    y_tr_full = training_set[target_col].values
    X_te      = testing_set.drop(columns=[target_col]).values
    y_te      = testing_set[target_col].values
    X_tr_n, X_te_n, _, _ = normalize_features(X_tr_full, X_te)
    print("  Training final model …", flush=True)
    final     = SMO_SVM(kfn, C=C).fit(X_tr_n, y_tr_full)
    y_pred_te = final.predict(X_te_n)
    train_time = time.time() - t0
    test_m = compute_metrics(y_te, y_pred_te)
    _print_result(test_m, np.mean(cv_scores), np.std(cv_scores), final.n_sv_)
    label = {"linear":"SMO Linear","polynomial":"SMO Poly","gaussian":"SMO Gaussian"}[svm_type]
    return _make_result(label, "SMO", kname_map[svm_type],
                        cv_scores, test_m, train_time, final.n_sv_)


# ─── 5–7. sklearn SVC ────────────────────────────────────────

def run_sklearn(training_set, testing_set, kernel="linear",
                k=5, C=1.0, degree=3, gamma=0.05):
    idx_map  = {"linear": 5, "poly": 6, "rbf": 7}
    kname_map = {"linear": "Linear", "poly": f"Poly d={degree}", "rbf": f"RBF γ={gamma}"}
    print(f"\n{'═'*55}")
    print(f"  [{idx_map[kernel]}] sklearn SVC — kernel={kernel}  C={C}")
    print(f"{'═'*55}")

    target_col = "diagnosis"
    X_tr_full = training_set.drop(columns=[target_col]).values
    y_tr_full = training_set[target_col].values
    X_te      = testing_set.drop(columns=[target_col]).values
    y_te      = testing_set[target_col].values

    # Normalize toàn bộ train set một lần cho sklearn
    scaler    = StandardScaler()
    X_tr_n    = scaler.fit_transform(X_tr_full)
    X_te_n    = scaler.transform(X_te)

    svc_kwargs = dict(kernel=kernel, C=C, random_state=42)
    if kernel == "poly":
        svc_kwargs["degree"]  = degree
        svc_kwargs["coef0"]   = 1.0
    if kernel == "rbf":
        svc_kwargs["gamma"]   = gamma

    t0  = time.time()
    clf = SVC(**svc_kwargs)

    # CV với sklearn (stratified)
    skf       = StratifiedKFold(n_splits=k, shuffle=False)
    cv_scores = cross_val_score(clf, X_tr_n, y_tr_full, cv=skf,
                                scoring="accuracy").tolist()
    print(f"  CV scores: {[f'{s:.4f}' for s in cv_scores]}")

    clf.fit(X_tr_n, y_tr_full)
    y_pred_te  = clf.predict(X_te_n)
    train_time = time.time() - t0
    n_sv       = int(clf.support_vectors_.shape[0])
    test_m     = compute_metrics(y_te, y_pred_te)
    _print_result(test_m, np.mean(cv_scores), np.std(cv_scores), n_sv)

    label = {"linear":"sklearn Linear","poly":"sklearn Poly","rbf":"sklearn Gaussian"}[kernel]
    return _make_result(label, "sklearn", kname_map[kernel],
                        cv_scores, test_m, train_time, n_sv)


def _print_result(m, cv_mean, cv_std, n_sv=None):
    print(f"  Accuracy   : {m['accuracy']:.4f}")
    print(f"  Precision  : {m['precision']:.4f}")
    print(f"  Recall     : {m['recall']:.4f}")
    print(f"  F1         : {m['f1']:.4f}")
    print(f"  CV Acc     : {cv_mean:.4f} ± {cv_std:.4f}")
    if n_sv is not None:
        print(f"  #SV        : {n_sv}")


# ═══════════════════════════════════════════════════════════
# VISUALIZATION  (7 phương pháp)
# ═══════════════════════════════════════════════════════════

# Palette: SGD=grey, SMO-linear=blue, SMO-poly=orange, SMO-gauss=green,
#          SK-linear=cyan, SK-poly=red, SK-gauss=yellow
COLORS = ["#8B8FA8", "#4E9AF1", "#F1A44E", "#4EF1A0",
          "#A78BFA", "#F16A6A", "#F1E44E"]

def plot_comparison(results: list, save_path="svm_comparison.png"):
    n       = len(results)
    labels  = [r["label"] for r in results]
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
    mnames  = ["Accuracy", "Precision", "Recall", "F1"]
    colors  = COLORS[:n]

    fig = plt.figure(figsize=(22, 20), facecolor="#0D1117")
    fig.suptitle("SVM Comparison — 7 Methods\n"
                 "(SGD primal · SMO dual · sklearn)",
                 fontsize=20, fontweight="bold", color="white", y=0.99)

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.38)

    # ── Row 0: grouped bar — test metrics ────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor("#161B22")
    x  = np.arange(len(metrics))
    bw = 0.11
    for idx, r in enumerate(results):
        vals = [r[m] for m in metrics]
        bars = ax0.bar(x + idx*bw, vals, width=bw, label=labels[idx],
                       color=colors[idx], alpha=0.9,
                       edgecolor="#0D1117", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax0.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.003,
                     f"{v:.2f}", ha="center", va="bottom",
                     fontsize=6, color="white", rotation=90)
    ax0.set_xticks(x + bw * (n-1)/2)
    ax0.set_xticklabels(mnames, color="white", fontsize=12)
    ax0.set_ylim(0, 1.22)
    ax0.set_ylabel("Score", color="white")
    ax0.set_title("Test Set — All Metrics, All 7 Methods",
                  color="white", fontsize=13, pad=8)
    ax0.legend(facecolor="#21262D", labelcolor="white",
               edgecolor="#444", fontsize=7.5, ncol=4,
               loc="upper right")
    ax0.tick_params(colors="white")
    for sp in ax0.spines.values(): sp.set_edgecolor("#444")
    ax0.yaxis.grid(True, color="#2A2F38", linewidth=0.5)

    # ── Row 1: CV line per fold ───────────────────────────
    ax1 = fig.add_subplot(gs[1, :3])
    ax1.set_facecolor("#161B22")
    k_f    = len(results[0]["cv_scores"])
    x_fold = np.arange(1, k_f + 1)
    for idx, r in enumerate(results):
        ax1.plot(x_fold, r["cv_scores"], "o-",
                 color=colors[idx], linewidth=1.8,
                 markersize=6, label=r["label"])
        ax1.axhline(r["cv_acc_mean"], color=colors[idx],
                    linestyle="--", linewidth=0.8, alpha=0.45)
    ax1.set_xticks(x_fold)
    ax1.set_xticklabels([f"Fold {i}" for i in x_fold], color="white", fontsize=9)
    ax1.set_ylim(0.6, 1.08)
    ax1.set_ylabel("CV Accuracy", color="white")
    ax1.set_title("5-Fold CV Accuracy per Fold  (dashed = mean)",
                  color="white", fontsize=12)
    ax1.legend(facecolor="#21262D", labelcolor="white",
               edgecolor="#444", fontsize=7, ncol=2)
    ax1.tick_params(colors="white")
    for sp in ax1.spines.values(): sp.set_edgecolor("#444")
    ax1.yaxis.grid(True, color="#2A2F38", linewidth=0.5)

    # ── Row 1 right: CV mean ± std bar ───────────────────
    ax1b = fig.add_subplot(gs[1, 3])
    ax1b.set_facecolor("#161B22")
    cv_means = [r["cv_acc_mean"] for r in results]
    cv_stds  = [r["cv_acc_std"]  for r in results]
    bars1b   = ax1b.barh(labels[::-1], cv_means[::-1],
                          color=colors[::-1], alpha=0.85,
                          edgecolor="#0D1117", linewidth=0.4)
    ax1b.errorbar(cv_means[::-1], labels[::-1],
                  xerr=cv_stds[::-1], fmt="none",
                  color="white", capsize=4, linewidth=1.2)
    for bar, v in zip(bars1b, cv_means[::-1]):
        ax1b.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                  f"{v:.4f}", va="center", fontsize=7, color="white")
    ax1b.set_xlim(0.5, 1.05)
    ax1b.set_title("CV Mean ± Std", color="white", fontsize=10)
    ax1b.tick_params(colors="white", labelsize=7)
    for sp in ax1b.spines.values(): sp.set_edgecolor("#444")
    ax1b.xaxis.grid(True, color="#2A2F38", linewidth=0.5)

    # ── Row 2: confusion matrices (7 = 4+3 layout) ───────
    cm_positions = [
        gs[2, 0], gs[2, 1], gs[2, 2], gs[2, 3],
        gs[3, 0], gs[3, 1], gs[3, 2],
    ]
    cell_lbl = [["TN","FP"],["FN","TP"]]
    for idx, r in enumerate(results):
        ax = fig.add_subplot(cm_positions[idx])
        ax.set_facecolor("#161B22")
        cm   = r["confusion"]
        cmap = LinearSegmentedColormap.from_list("c", ["#161B22", colors[idx]])
        ax.imshow(cm, cmap=cmap, vmin=0)
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii,
                        f"{cell_lbl[ii][jj]}\n{cm[ii,jj]}",
                        ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["B","M"], color="white", fontsize=8)
        ax.set_yticklabels(["B","M"], color="white", fontsize=8)
        sv_txt = f"SV={r['n_sv']}" if r["n_sv"] != "—" else ""
        ax.set_title(f"{r['label']}\nacc={r['test_accuracy']:.3f}  {sv_txt}",
                     color="white", fontsize=8.5)
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    # ── Row 3 right: training time ────────────────────────
    ax_t = fig.add_subplot(gs[3, 3])
    ax_t.set_facecolor("#161B22")
    times  = [r["train_time"] for r in results]
    bars_t = ax_t.barh(labels[::-1], times[::-1],
                        color=colors[::-1], alpha=0.85,
                        edgecolor="#0D1117", linewidth=0.4)
    for bar, v in zip(bars_t, times[::-1]):
        ax_t.text(v + max(times)*0.01,
                  bar.get_y() + bar.get_height()/2,
                  f"{v:.1f}s", va="center", fontsize=7, color="white")
    ax_t.set_title("Training Time (s)", color="white", fontsize=10)
    ax_t.tick_params(colors="white", labelsize=7)
    for sp in ax_t.spines.values(): sp.set_edgecolor("#444")
    ax_t.xaxis.grid(True, color="#2A2F38", linewidth=0.5)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    print(f"\n  Chart saved → {save_path}")


# ═══════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════

def print_summary_table(results: list):
    W = 100
    print(f"\n{'═'*W}")
    print("  SVM COMPARISON — 7 METHODS")
    print(f"  {'#':<3} {'Method':<18} {'Impl':<8} {'Kernel':<14} "
          f"{'CV Acc':>8} {'CV Std':>7} {'Test Acc':>9} "
          f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'#SV':>6} {'Time(s)':>8}")
    print("─"*W)
    for i, r in enumerate(results, 1):
        sv = str(r["n_sv"]) if r["n_sv"] != "—" else "  —"
        print(f"  {i:<3} {r['label']:<18} {r['impl']:<8} {r['kernel']:<14} "
              f"{r['cv_acc_mean']:>8.4f} {r['cv_acc_std']:>7.4f} "
              f"{r['test_accuracy']:>9.4f} "
              f"{r['test_precision']:>7.4f} {r['test_recall']:>7.4f} "
              f"{r['test_f1']:>7.4f} {sv:>6} {r['train_time']:>8.2f}")
    print("─"*W)
    # Best per metric
    best_acc = max(results, key=lambda r: r["test_accuracy"])
    best_f1  = max(results, key=lambda r: r["test_f1"])
    best_cv  = max(results, key=lambda r: r["cv_acc_mean"])
    print(f"\n  Best Test Accuracy : {best_acc['label']}  ({best_acc['test_accuracy']:.4f})")
    print(f"  Best F1-Score      : {best_f1['label']}  ({best_f1['test_f1']:.4f})")
    print(f"  Best CV Accuracy   : {best_cv['label']}  ({best_cv['cv_acc_mean']:.4f})")
    print("═"*W)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    df = pd.read_csv(path)
    df = handle_missing_values(df)
    training_set, testing_set = stratified_split(
        df, target_col="diagnosis", test_size=0.2, random_state=42
    )

    results = []

    # ── 1. Linear SGD (primal, from scratch) ─────────────
    results.append(run_sgd_linear(
        training_set, testing_set,
        lr=0.001, lambda_param=0.001, n_epochs=2000
    ))

    # ── 2. Linear SMO (dual, from scratch) ───────────────
    results.append(run_smo(training_set, testing_set,
                           svm_type="linear", C=1.0))

    # ── 3. Polynomial SMO (dual, from scratch) ───────────
    results.append(run_smo(training_set, testing_set,
                           svm_type="polynomial", C=1.0, degree=3, coef0=1.0))

    # ── 4. Gaussian SMO (dual, from scratch) ─────────────
    results.append(run_smo(training_set, testing_set,
                           svm_type="gaussian", C=1.0, gamma=0.05))

    # ── 5. sklearn SVC — linear ──────────────────────────
    results.append(run_sklearn(training_set, testing_set,
                               kernel="linear", C=1.0))

    # ── 6. sklearn SVC — polynomial ──────────────────────
    results.append(run_sklearn(training_set, testing_set,
                               kernel="poly", C=1.0, degree=3))

    # ── 7. sklearn SVC — RBF (gaussian) ──────────────────
    results.append(run_sklearn(training_set, testing_set,
                               kernel="rbf", C=1.0, gamma=0.05))

    print_summary_table(results)
    plot_comparison(results, save_path="svm_comparison.png")


if __name__ == "__main__":
    main()