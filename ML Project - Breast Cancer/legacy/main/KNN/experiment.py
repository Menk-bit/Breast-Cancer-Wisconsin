from sklearn.model_selection import StratifiedKFold

from data_utils import (
    get_selected_features,
    prepare_data,
)

from knn_scratch import KNNClassifierScratch

from metrics_utils import (
    calculate_metrics,
    select_best_config,
    summarize_fold_metrics,
)


def build_configs_for_baseline(baseline_name, baseline_definition, k_values):
    configs = []

    for k in k_values:
        for corr_threshold in baseline_definition["corr_thresholds"]:
            configs.append(
                {
                    "baseline": baseline_name,
                    "k": k,
                    "feature_mode": baseline_definition["feature_mode"],
                    "corr_threshold": corr_threshold,
                    "use_scaling": baseline_definition["use_scaling"],
                    "distance_metric": baseline_definition["distance_metric"],
                    "weights": baseline_definition["weights"],
                }
            )

    return configs


def build_search_space(enabled_baselines, baseline_definitions, k_values):
    search_space = []

    for baseline_name in enabled_baselines:
        if baseline_name not in baseline_definitions:
            raise ValueError(f"Baseline chưa được định nghĩa: {baseline_name}")

        baseline_configs = build_configs_for_baseline(
            baseline_name=baseline_name,
            baseline_definition=baseline_definitions[baseline_name],
            k_values=k_values,
        )

        search_space.extend(baseline_configs)

    return search_space


def run_cv_for_config(
    X,
    y,
    config,
    n_splits,
    random_state,
    tie_break,
):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_metrics = []
    n_features_list = []

    for train_idx, valid_idx in skf.split(X, y):
        X_fold_train = X.iloc[train_idx]
        X_fold_valid = X.iloc[valid_idx]

        y_fold_train = y.iloc[train_idx]
        y_fold_valid = y.iloc[valid_idx]

        selected_features, _ = get_selected_features(
            X_fold_train,
            y_fold_train,
            feature_mode=config["feature_mode"],
            corr_threshold=config["corr_threshold"],
        )

        X_fold_train_prepared, X_fold_valid_prepared, _ = prepare_data(
            X_fold_train,
            X_fold_valid,
            selected_features,
            use_scaling=config["use_scaling"],
        )

        model = KNNClassifierScratch(
            k=config["k"],
            distance_metric=config["distance_metric"],
            weights=config["weights"],
            tie_break=tie_break,
        )

        model.fit(X_fold_train_prepared, y_fold_train)

        y_valid_pred = model.predict(X_fold_valid_prepared)

        metrics = calculate_metrics(
            y_true=y_fold_valid,
            y_pred=y_valid_pred,
        )

        fold_metrics.append(metrics)
        n_features_list.append(len(selected_features))

    summary = summarize_fold_metrics(fold_metrics)

    return {
        **config,
        **summary,
        "n_features_mean": sum(n_features_list) / len(n_features_list),
    }


def run_knn_cv_tuning(
    X_train,
    y_train,
    search_space,
    n_splits,
    random_state,
    tie_break,
):
    all_cv_results = []
    results_by_baseline = {}

    for config in search_space:
        result = run_cv_for_config(
            X=X_train,
            y=y_train,
            config=config,
            n_splits=n_splits,
            random_state=random_state,
            tie_break=tie_break,
        )

        all_cv_results.append(result)

        baseline_name = config["baseline"]
        results_by_baseline.setdefault(baseline_name, [])
        results_by_baseline[baseline_name].append(result)

    best_config, sorted_results = select_best_config(all_cv_results)

    return all_cv_results, results_by_baseline, best_config, sorted_results


def train_final_knn(
    X_train,
    y_train,
    X_test,
    best_config,
    tie_break,
):
    selected_features, corr_table = get_selected_features(
        X_train,
        y_train,
        feature_mode=best_config["feature_mode"],
        corr_threshold=best_config["corr_threshold"],
    )

    X_train_final, X_test_final, _ = prepare_data(
        X_train,
        X_test,
        selected_features,
        use_scaling=best_config["use_scaling"],
    )

    model = KNNClassifierScratch(
        k=int(best_config["k"]),
        distance_metric=best_config["distance_metric"],
        weights=best_config["weights"],
        tie_break=tie_break,
    )

    model.fit(X_train_final, y_train)

    return model, X_test_final, selected_features, corr_table


def evaluate_final_knn(
    model,
    X_test_final,
    y_test,
):
    y_test_pred = model.predict(X_test_final)

    return calculate_metrics(
        y_true=y_test,
        y_pred=y_test_pred,
    )
