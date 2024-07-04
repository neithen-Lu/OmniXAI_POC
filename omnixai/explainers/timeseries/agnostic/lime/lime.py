import warnings
import numpy as np
import pandas as pd
from typing import List, Callable

from sklearn.linear_model import LinearRegression

from ....base import ExplainerBase
from .....data.timeseries import Timeseries
from .....explanations.timeseries.feature_importance import FeatureImportance
from .perturber import BlockSelector,PerturbedDataGenerator


class LimeTimeseries(ExplainerBase):
    """Time Series Local Interpretable Model-agnostic Explainer (TSLime) is a model-agnostic local time series
    explainer. LIME (Locally interpretable Model agnostic explainer) is a popular algorithm for local
    explanation. LIME explains the model behavior by approximating the model response with linear models.
    LIME algorithm specifically assumes tabular data format, where each row is a data point, and columns
    are features. A generalization of LIME algorithm for image data uses super pixel based perturbation.
    TSLIME generalizes LIME algorithm for time series context.

    TSLIME uses time series perturbation methods to produce a local input perturbation, and linear model
    surrogate which best approximates the model response. TSLime produces an interpretable explanation.
    The explanation weights produced by the TSLime explanation indicates model local sensitivity.

    References:
        .. [#0] `Ribeiro et al. '"Why Should I Trust You?": Explaining the Predictions of Any Classifier'
            <https://arxiv.org/abs/1602.04938>`_

    """

    explanation_type = "local"
    alias = ["lime"]

    def __init__(
        self,
        model: Callable,
        n_perturbations: int = 2000,
        mode: str = "anomaly_detection",
    ):
        """Initializer for TSLimeExplainer

        Args:
            model (Callable): Callable object produces a prediction as numpy array
                for a given input as numpy array. It can be a model prediction (predict/
                predict_proba) function that results a real value like probability or regressed value.
                This function must accept numpy array of shape (input_length x len(feature_names)) as
                input and result in numpy array of shape (1, -1). Currently, TSLime supports sinlge output
                models only. For multi-output models, you can aggregate the output using a custom
                model_wrapper. Use model wrapper classes from aix360.algorithms.tsutils.model_wrappers.
            input_length (int): Input (history) length used for input model.
            n_perturbations (int): Number of perturbed instance for TSExplanation. Defaults to 25.
            relevant_history (int): Interested window size for explanations. The explanation is
                computed for selected latest window of length `relevant_history`. If `input_length=20`
                and `relevant_history=10`, explanation is computed for last 10 time points. If None,
                relevant_history is set to input_length. Defaults to None.
            perturbers (List[TSPerturber, dict]): data perturbation algorithm specification by TSPerturber
                instance or dict. Allowed values for "type" key in dictionary are block-bootstrap, frequency,
                moving-average, shift. Block-bootstrap split the time series into contiguous
                chunks called blocks, for each block noise is estimated and noise is exchanged
                and added to the signal between randomly selected blocks. Moving-average perturbation
                maintains the moving mean of the time series data with the specified window length,
                but add perturbed noise with similar distribution as the data. Frequency
                perturber performs FFT on the noise, and removes random high frequency
                components from the noise estimates. Number of frequencies to be removed
                is specified by the truncate_frequencies argument. Shift perturber adds
                random upward or downward shift in the data value over time continuous
                blocks. If not provided default perturber is block-bootstrap. Defaults to None.
        """
        self.mode = mode
        self.model = model

        # Surrogate training params
        self.surrogate_model = LinearRegression()
        self.n_perturbations = n_perturbations
        self.perturber = None

    def _ts_perturb(self, x):
        # create perturbations
        x_perturbations = None

        x_perturbations, _ = self.perturber.fit_transform(
            x, None, n=self.n_perturbations
        )

        x_perturbations = np.asarray(x_perturbations).astype("float")
        return x_perturbations

    def _batch_predict(self, x_perturbations):
        f_predict_samples = None

        try:
            f_predict_samples = self.model(x_perturbations)
        except Exception as ex:
            warnings.warn(
                "Batch scoring failed with error: {}. Scoring sequentially...".format(
                    ex
                )
            )
            f_predict_samples = [
                self.model(x_perturbations[i]) for i in range(x_perturbations.shape[0])
            ]
            f_predict_samples = np.array(f_predict_samples)

        return f_predict_samples

    def explain(self, X: Timeseries, **explain_params):
        """Explain the prediction made by the time series model at a certain point in time
        (**local explanation**).

        :param X: An instance of `Timeseries` representing one input instance or
            a batch of input instances.

        :param explain_params: Arbitrary explainer parameters.

        Returns:
            dict: explanation object
                Dictionary with keys: input_data, history_weights, model_prediction,
                surrogate_prediction, x_perturbations, y_perturbations.
        """
        explanations = FeatureImportance(self.mode)

        ### input validation
        X_numpy = X.to_numpy()
        input_length = X.shape[0]

        # build perturber
        if self.perturber is None:
            perturbers = [
                dict(type="block-bootstrap"),
            ]

        # build perturber
        block_selector = BlockSelector(start=-input_length, end=None)
        self.perturber = PerturbedDataGenerator(
            perturber_engines=perturbers,
            block_selector=block_selector,
        )

        ### generate time series perturbations
        x_perturbations = self._ts_perturb(x=X_numpy)

        ### generate y
        y_perturbations = self._batch_predict(x_perturbations)
        if y_perturbations is None:
            raise Exception(
                "Model prediction could not be computed for gradient samples."
            )

        y_perturbations = np.asarray(y_perturbations).astype("float")

        ### compute weights using a linear model
        x_perturbations = x_perturbations.reshape(self.n_perturbations,-1)
        self.surrogate_model.fit(
            X=x_perturbations,
            y=y_perturbations,
        )
        lime_values = self.surrogate_model.coef_
        lime_values = lime_values.reshape(X.shape)

        scores = pd.DataFrame(
            lime_values,
            columns=X.columns,
            index=X.index
        )
        explanations.add(X.to_pd(), scores)
        return explanations