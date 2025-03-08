import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import KFold

problem_title = "Predicting Legal Assistance Costs"
_target_column_name = "Montant total NDURINT"
_ignore_column_names = []

Predictions = rw.prediction_types.make_regression()

workflow = rw.workflows.Estimator()

score_types = [
    rw.score_types.RMSE(name="rmse", precision=4),
]


def get_cv(X, y):
    """Define cross-validation strategy (KFold)."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    return cv.split(X, y)


def _read_data(path, df_filename):
    """Load dataset from CSV and split into features and target."""
    df = pd.read_csv(os.path.join(path, "data", df_filename))
    y_array = df[_target_column_name].values
    X_dict = df.drop(columns=[_target_column_name])
    return X_dict, y_array


def get_train_data(path="."):
    """Load training data."""
    df_filename = "train.csv"
    return _read_data(path, df_filename)


def get_test_data(path="."):
    """Load test data."""
    df_filename = "test.csv"
    return _read_data(path, df_filename)