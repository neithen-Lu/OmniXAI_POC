import os
import unittest
import numpy as np

from omnixai.utils.misc import set_random_seed
from omnixai.data.timeseries import Timeseries
from omnixai.explainers.timeseries.agnostic.lime import LimeTimeseries


class TestLimeTimeseries(unittest.TestCase):
    def setUp(self) -> None:
        set_random_seed()
        y = np.random.randn(5, 5)

        self.test_data = Timeseries(
            data=y,
            timestamps=list(range(y.shape[0])),
            variable_names=list('x' * (i + 1) for i in range(y.shape[1]))
        )
        self.predict_function = lambda _x: np.sum(_x.values[:, 0])

    def test(self):
        set_random_seed()
        explainer = LimeTimeseries(
            model=self.predict_function,
            mode='forecasting'
        )
        explanations = explainer.explain(self.test_data)
        scores = explanations.get_explanations(index=0)["scores"].values

        # validate explanation structure
        self.assertEqual(scores.shape, self.test_data.shape)


if __name__ == "__main__":
    unittest.main()
