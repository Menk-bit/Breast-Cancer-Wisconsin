from config import (
    BASELINE_DEFINITIONS,
    CSV_PATH,
    ENABLED_BASELINES,
    K_VALUES,
    N_SPLITS,
    RANDOM_STATE,
    TEST_SIZE,
    TIE_BREAK,
    TOP_N_GLOBAL,
    TOP_N_PER_BASELINE,
)

from data_utils import (
    load_data,
    preprocess_data,
    split_train_test,
)

from experiment import (
    build_search_space,
    evaluate_final_knn,
    run_knn_cv_tuning,
    train_final_knn,
)

from metrics_utils import (
    print_baseline_cv_summary,
    print_best_cv_config,
    print_global_cv_summary,
    print_test_result,
)


def main():
    df = load_data(CSV_PATH)
    X, y, feature_names = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("\nDataset")
    print(f"Samples       : {len(X)}")
    print(f"Features      : {len(feature_names)}")
    print(f"Train samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")
    print("Test set      : held out, not used for tuning")

    search_space = build_search_space(
        enabled_baselines=ENABLED_BASELINES,
        baseline_definitions=BASELINE_DEFINITIONS,
        k_values=K_VALUES,
    )

    print("\nSearch space")
    print(f"Enabled baselines : {len(ENABLED_BASELINES)}")
    print(f"Total configs     : {len(search_space)}")

    all_cv_results, results_by_baseline, best_config, _ = run_knn_cv_tuning(
        X_train=X_train,
        y_train=y_train,
        search_space=search_space,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        tie_break=TIE_BREAK,
    )

    for baseline_name in ENABLED_BASELINES:
        print_baseline_cv_summary(
            baseline_name=baseline_name,
            cv_results=results_by_baseline[baseline_name],
            top_n=TOP_N_PER_BASELINE,
        )

    print_global_cv_summary(
        cv_results=all_cv_results,
        top_n=TOP_N_GLOBAL,
    )

    print_best_cv_config(best_config)

    final_model, X_test_final, selected_features, corr_table = train_final_knn(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        best_config=best_config,
        tie_break=TIE_BREAK,
    )

    test_metrics = evaluate_final_knn(
        model=final_model,
        X_test_final=X_test_final,
        y_test=y_test,
    )

    print("\n\n================================================")
    print("FINAL SELECTED FEATURES")
    print("================================================")
    print(f"Number of features: {len(selected_features)}")
    print(", ".join(selected_features))

    print_test_result(test_metrics)


if __name__ == "__main__":
    main()
    