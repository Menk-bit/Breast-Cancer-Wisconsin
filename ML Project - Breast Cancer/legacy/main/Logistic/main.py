from config import (
    BASELINE_C,
    BASELINE_CLASS_WEIGHT,
    BASELINE_LEARNING_RATE,
    BASELINE_THRESHOLD,
    CLASS_WEIGHT_SURVEY_VALUES,
    C_SURVEY_VALUES,
    CSV_PATH,
    EPOCHS,
    LEARNING_RATE_SURVEY_VALUES,
    N_SPLITS,
    PRINT_EVERY,
    RANDOM_STATE,
    SHOW_LOSS_CURVE,
    TEST_SIZE,
    THRESHOLD_SURVEY_VALUES,
)

from data_utils import (
    load_data,
    preprocess_data,
    split_train_test,
)

from experiment import (
    run_baseline_report,
    run_c_survey,
    run_class_weight_survey,
    run_learning_rate_survey,
    run_threshold_survey,
)

from metrics_utils import print_survey_table
from visualize import plot_loss_curve
from evaluation import (
    train_final_model,
    evaluate_on_test,
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

    base_config = {
        "C": BASELINE_C,
        "learning_rate": BASELINE_LEARNING_RATE,
        "class_weight": BASELINE_CLASS_WEIGHT,
        "threshold": BASELINE_THRESHOLD,
    }

    print("\nDataset")
    print(f"Samples       : {len(X)}")
    print(f"Features      : {len(feature_names)}")
    print(f"Train samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")
    print("Test set      : held out, not used for tuning")

    _, _, baseline_history = run_baseline_report(
        X=X_train,
        y=y_train,
        C=BASELINE_C,
        learning_rate=BASELINE_LEARNING_RATE,
        class_weight=BASELINE_CLASS_WEIGHT,
        threshold=BASELINE_THRESHOLD,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        epochs=EPOCHS,
        print_every=PRINT_EVERY,
    )

    if SHOW_LOSS_CURVE:
        plot_loss_curve(
            baseline_history,
            title="Baseline Logistic Regression Loss Curve",
        )

    c_survey_df = run_c_survey(
        X=X_train,
        y=y_train,
        C_values=C_SURVEY_VALUES,
        base_config=base_config,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        epochs=EPOCHS,
        print_every=PRINT_EVERY,
    )

    print_survey_table(
        "C SURVEY",
        c_survey_df,
    )

    learning_rate_survey_df = run_learning_rate_survey(
        X=X_train,
        y=y_train,
        learning_rate_values=LEARNING_RATE_SURVEY_VALUES,
        base_config=base_config,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        epochs=EPOCHS,
        print_every=PRINT_EVERY,
    )

    print_survey_table(
        "LEARNING RATE SURVEY",
        learning_rate_survey_df,
    )

    class_weight_survey_df = run_class_weight_survey(
        X=X_train,
        y=y_train,
        class_weight_values=CLASS_WEIGHT_SURVEY_VALUES,
        base_config=base_config,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        epochs=EPOCHS,
        print_every=PRINT_EVERY,
    )

    print_survey_table(
        "CLASS WEIGHT SURVEY",
        class_weight_survey_df,
    )

    threshold_survey_df = run_threshold_survey(
        X=X_train,
        y=y_train,
        threshold_values=THRESHOLD_SURVEY_VALUES,
        base_config=base_config,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        epochs=EPOCHS,
        print_every=PRINT_EVERY,
    )

    print_survey_table(
        "THRESHOLD SURVEY",
        threshold_survey_df,
    )
        # ========================================================
    # 8. Final test evaluation
    # ========================================================
    # Cấu hình này được chọn từ CV, không chọn theo test set.

    final_config = {
        "C": 1.0,
        "learning_rate": 0.05,
        "class_weight": None,
        "threshold": 0.4,
    }

    final_model, final_imputer, final_scaler = train_final_model(
        X_train=X_train,
        y_train=y_train,
        C=final_config["C"],
        learning_rate=final_config["learning_rate"],
        class_weight=final_config["class_weight"],
        epochs=EPOCHS,
        print_every=PRINT_EVERY,
    )

    test_metrics = evaluate_on_test(
        model=final_model,
        imputer=final_imputer,
        scaler=final_scaler,
        X_test=X_test,
        y_test=y_test,
        threshold=final_config["threshold"],
    )

    print("\n\n================================================")
    print("FINAL SELECTED CONFIG")
    print("================================================")
    print(f"C            : {final_config['C']}")
    print(f"learning_rate: {final_config['learning_rate']}")
    print(f"class_weight : {final_config['class_weight']}")
    print(f"threshold    : {final_config['threshold']}")

    print_test_result(
        "FINAL TEST RESULT",
        test_metrics,
    )


if __name__ == "__main__":
    main()