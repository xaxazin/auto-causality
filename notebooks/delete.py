import os
import sys
import numpy as np
import warnings
import pandas as pd
import matplotlib
from scipy import special
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

root_path = root_path = os.path.realpath('../..')
try:
    import auto_causality
except ModuleNotFoundError:
    sys.path.append(os.path.join(root_path, "auto-causality"))

from auto_causality import AutoCausality
from auto_causality.data_utils import preprocess_dataset
from auto_causality.scoring import Scorer

warnings.filterwarnings("ignore")


def iv_dgp_econml(n=5000, p=10, true_effect=10):

    X = np.random.normal(0, 1, size=(n, p))
    Z = np.random.binomial(1, 0.5, size=(n,))
    nu = np.random.uniform(0, 5, size=(n,))
    coef_Z = 0.8
    C = np.random.binomial(
        1, coef_Z * special.expit(0.4 * X[:, 0] + nu)
    )  # Compliers when recomended
    C0 = np.random.binomial(
        1, 0.006 * np.ones(X.shape[0])
    )  # Non-compliers when not recommended
    T = C * Z + C0 * (1 - Z)
    y = (
        true_effect(X) * T
        + 2 * nu
        + 5 * (X[:, 3] > 0)
        + 1.5 * np.random.uniform(0, 1, size=(n,))
    )
    cov = [f"x{i}" for i in range(1, X.shape[1] + 1)]
    df = pd.DataFrame(X, columns=cov)

    df["y"] = y
    df["treatment"] = T
    df["Z"] = Z

    return df


if __name__ == "__main__":

    # Dataset parameters
    treatment = "treatment"
    targets = ["y"]
    instruments = ["Z"]
    outcome = targets[0]

    TRUE_EFFECT = 7.5

    CONSTANT_EFFECT = lambda X: TRUE_EFFECT

    data = iv_dgp_econml(n=10000, p=15, true_effect=CONSTANT_EFFECT)
    data_df, features_X, features_W = preprocess_dataset(
        data, treatment, targets, instruments
    )
    outcome = targets[0]
    train_df, test_df = train_test_split(data_df, test_size=0.33)

    Xtest_constant_te = test_df[features_X]
    
    # estimator_list = ["SimpleIV", "SparseLinearDRIV", "DMLIV", "OrthoIV", "LinearDRIV", "LinearIntentToTreatDRIV"]
    estimator_list = ["OrthoIV"]

    ac_constant_te = AutoCausality(
        estimator_list=estimator_list,
        verbose=3,
        components_verbose=2,
        components_time_budget=300,
        propensity_model="dummy",
        train_size=0.5
    )

    ac_constant_te.fit(train_df, treatment, outcome, features_W, features_X, instruments)

    print(ac_constant_te.results.results)


    