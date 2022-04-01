from typing import List

import numpy as np
import pandas as pd

from auto_causality.models.wrapper import DoWhyMethods, DoWhyWrapper
from auto_causality.scoring import ate


class DummyModel(DoWhyMethods):
    def __init__(
        self,
        propensity_modifiers: List[str],
        outcome_modifiers: List[str],
        treatment: str,
        outcome: str,
    ):
        self.propensity_modifiers = propensity_modifiers
        self.outcome_modifiers = outcome_modifiers
        self.treatment = treatment
        self.outcome = outcome

    def fit(
        self,
        df: pd.DataFrame,
    ):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        mean_, _, _ = ate(X[self.treatment], X[self.outcome])
        return np.ones(len(X)) * mean_ * (1 + 0.01 * np.random.normal(size=(len(X),)))


class Dummy(DoWhyWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, inner_class=DummyModel, **kwargs)