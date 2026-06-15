CSV_PATH = "/Users/minhdt/Desktop/ML breast cancer/KNN/breast-cancer-wisconsin-data_data 2.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]
CORR_THRESHOLDS = [0.2, 0.3, 0.4]

TIE_BREAK = 1
TOP_N_PER_BASELINE = 5
TOP_N_GLOBAL = 10

ENABLED_BASELINES = [
    "FULL_SCALE_EUCLIDEAN_UNIFORM",
    "SELECTED_SCALE_EUCLIDEAN_UNIFORM",
    "SELECTED_SCALE_EUCLIDEAN_DISTANCE",
]

BASELINE_DEFINITIONS = {
    "FULL_SCALE_EUCLIDEAN_UNIFORM": {
        "feature_mode": "full",
        "corr_thresholds": [None],
        "use_scaling": True,
        "distance_metric": "euclidean",
        "weights": "uniform",
    },

    "FULL_SCALE_MANHATTAN_UNIFORM": {
        "feature_mode": "full",
        "corr_thresholds": [None],
        "use_scaling": True,
        "distance_metric": "manhattan",
        "weights": "uniform",
    },

    "FULL_NO_SCALE_EUCLIDEAN_UNIFORM": {
        "feature_mode": "full",
        "corr_thresholds": [None],
        "use_scaling": False,
        "distance_metric": "euclidean",
        "weights": "uniform",
    },

    "SELECTED_SCALE_EUCLIDEAN_UNIFORM": {
        "feature_mode": "correlation",
        "corr_thresholds": CORR_THRESHOLDS,
        "use_scaling": True,
        "distance_metric": "euclidean",
        "weights": "uniform",
    },

    "SELECTED_SCALE_EUCLIDEAN_DISTANCE": {
        "feature_mode": "correlation",
        "corr_thresholds": CORR_THRESHOLDS,
        "use_scaling": True,
        "distance_metric": "euclidean",
        "weights": "distance",
    },

    "SELECTED_SCALE_MANHATTAN_UNIFORM": {
        "feature_mode": "correlation",
        "corr_thresholds": CORR_THRESHOLDS,
        "use_scaling": True,
        "distance_metric": "manhattan",
        "weights": "uniform",
    },

    "SELECTED_SCALE_MANHATTAN_DISTANCE": {
        "feature_mode": "correlation",
        "corr_thresholds": CORR_THRESHOLDS,
        "use_scaling": True,
        "distance_metric": "manhattan",
        "weights": "distance",
    },
}