import numpy as np
import pandas as pd
from typing import Union, List
from abc import ABC, abstractmethod
from collections import Counter

class BlockSelector:
    """
    BlockSelector is used to prepare the effective window to compute explanation
    from the user provided parameters. This is used while computing timeseries perturbations.
    """

    def __init__(self, start: int, end: int):
        self._start = start
        self._end = end

    def select_start_point(
        self, x, n: int = 1, margin: int = None, block_length: int = 5
    ):
        start, end = self._start, self._end
        x = np.array(x)
        t = x.shape[0]

        if (margin is not None) and (margin < 0):
            margin = t - margin

        if margin is None:
            margin = t

        if margin < 0:
            raise ValueError(
                f"Error: margin should be a valid point with in data length!"
            )

        if start < 0:
            start = t + start
            if start < 0:
                start = 0

        if (end is not None) and (end < 0):
            end = t + end
            if end < 0:
                raise ValueError(f"Error: end must be within the index range!")
        elif end is None:
            end = t

        end = min(end, margin)

        return np.random.randint(low=start, high=end, size=n)


class TSPerturber(ABC):
    """Abstract interface for time series perturbation."""

    def __init__(self):
        self._fitted = False

    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        x: Union[pd.DataFrame, np.ndarray],
    ):
        try:
            self._fit(x)
            self._fitted = True
        except RuntimeError as e:
            raise e
        return self

    def fit_transform(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        n_perturbations: int = 1,
        block_selector: BlockSelector = None,
    ):
        self.fit(x)
        return self.transform(
            n_perturbations=n_perturbations, block_selector=block_selector
        )

    def transform(
        self,
        n_perturbations: int = 1,
        block_selector: BlockSelector = None,
    ):
        if not self.is_fitted():
            raise RuntimeError(
                "Error: transform must be called after fitting the data!"
            )
        return self._transform(n_perturbations, block_selector)

    @abstractmethod
    def _fit(self, x: Union[pd.DataFrame, np.ndarray]):
        pass

    @abstractmethod
    def _transform(
        self, n_perturbations: int = 1, block_selector: BlockSelector = None
    ):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, **kwargs):
        pass


class BlockBootstrapPerturber(TSPerturber):
    """BlockBootstrapPerturber split the time series into contiguous chunks
    called blocks, for each block noise is estimated and noise is exchanged
    and added to the signal (mean) between randomly selected blocks.

    References:
        .. [#0] `Bühlmann, Peter. "Bootstraps for time series."
            Statistical science (2002): 52-72.
            <https://projecteuclid.org/journals/statistical-science/volume-17/issue-1/Bootstraps-for-Time-Series/10.1214/ss/1023798998.full>`_
    """

    def __init__(
        self, window_length: int = 5, block_length: int = 5, block_swap: int = 2
    ):
        """BlockBootstrapPerturber initialization.

        Args:
            window_length (int): window length used for noise estimation. Defaults to 5.
            block_length (int): block length, perturber swaps noise between blocks. Defaults to 5.
            block_swap (int): number of block pairs for perturbation. Defaults to 2.
        """
        super(BlockBootstrapPerturber, self).__init__()
        self._mean = None
        self._residual = None
        self._data_length = None
        self._parameters = dict()
        self._parameters["window_length"] = window_length
        self._parameters["block_length"] = block_length
        self._parameters["block_swap"] = block_swap

    def get_params(self):
        return self._parameters.copy()

    def set_params(self, **kwargs):
        self._parameters.update(kwargs)
        return self

    def _fit(
        self,
        x: np.ndarray,
    ):
        window_length = self._parameters.get("window_length")
        self._mean, self._residual = ts_split_mean_residual(
            x, window_size=window_length
        )
        self._data_length = x.shape[0]
        return self

    def _transform(
        self,
        n_perturbations: int = 1,
        block_selector: BlockSelector = None,
    ):
        block_length = self._parameters.get("block_length")
        block_swap = self._parameters.get("block_swap")

        x_res = [self._residual.copy() for _ in range(n_perturbations)]
        margin = self._residual.shape[0] - block_length + 1
        for _ in range(block_swap):
            if block_selector is None:
                from_point = np.random.randint(
                    0, self._data_length - block_length, n_perturbations
                )

                to_point = np.random.randint(
                    0, self._data_length - block_length, n_perturbations
                )
            else:
                from_point = block_selector.select_start_point(
                    x=self._residual, n=n_perturbations, margin=margin
                )
                to_point = block_selector.select_start_point(
                    x=self._residual, n=n_perturbations, margin=margin
                )

            for j, start in enumerate(zip(from_point, to_point)):
                start_1, start_2 = start
                x_res[j][start_1 : (start_1 + block_length)] = self._residual[
                    start_2 : (start_2 + block_length)
                ]
                x_res[j][start_2 : (start_2 + block_length)] = self._residual[
                    start_1 : (start_1 + block_length)
                ]

        return [self._mean + res for res in x_res]
    
def ts_rolling_mean(
    ts: np.ndarray,
    window_size: int,
):
    """ts_rolling_mean computes rolling mean for numpy ndarray objects.
    The reported rolling mean is of same dimension that of input time series,
    the boundary are adaptively adjusted, for valid window length only.

    Args:
        ts (numpy ndarray): Time series data, as numpy ndarray object.
        window_size (int): number of consecutive data points over which the averaging
            will be performed.

    Returns:
        df (numpy ndarray)
    """
    # shape check
    if ts.shape[0] < ts.shape[1]:
        raise NotImplementedError('Only support n_obs >= n_variables')
    df = ts.copy()
    if isinstance(ts, np.ndarray):
        if len(ts.shape) == 1:
            ts = ts.reshape(-1, 1)
        ts = ts.astype("float")
        n_obs, n_vars = ts.shape
        den = np.convolve(
            np.ones(n_obs), np.ones(window_size, dtype="float"), "same"
        ).astype("float")
        df = np.asarray(
            [
                np.convolve(ts[:, i], np.ones(window_size), "same") / den
                for i in range(n_vars)
            ]
        ).T
    elif isinstance(ts, pd.DataFrame):
        dfv = ts_rolling_mean(ts.values, window_size=window_size)
        df.loc[:, ts.columns] = dfv
    return df


def ts_split_mean_residual(
    ts: np.ndarray,
    window_size: int,
):
    """Split the input data into moving average and residual component.
    The API supports both tsFrame (DataFrame) and numpy ndarray (numeric Array)
    format.

    Args:
        ts (numpy ndarray): input time series as tsFrame or numpy array
        window_size (int): number of observations for averaging.

    Returns:
        tuple (Tuple[numpy ndarray, numpy ndarray]): of same dimension as
            input.
    """
    ts_avg = ts_rolling_mean(ts, window_size)
    ts_res = ts - ts_avg
    return ts_avg, ts_res


class PerturbedDataGenerator:
    """
    PerturbedDataGenerator is a wrapping class to prepare various kinds of
    perturbers and generate specified number of perturbations using these
    perturbers.
    """

    def __init__(
        self,
        perturber_engines: List[Union[TSPerturber, dict]] = None,
        block_selector: BlockSelector = None,
    ):
        """
        Constructor method, initializes the explainer

        Args:
            perturber_engines (List[TSPerturber, dict]): data perturbation algorithm specification
                by TSPerturber instance or dict. Allowed values for "type" key in dictionary are
                block-bootstrap, frequency, moving-average, shift. Block-bootstrap split the time series
                into contiguous chunks called blocks, for each block noise is estimated and noise is exchanged
                and added to the signal between randomly selected blocks. Moving-average perturbation
                maintains the moving mean of the time series data with the specified window length,
                but add perturbed noise with similar distribution as the data. Frequency
                perturber performs FFT on the noise, and removes random high frequency
                components from the noise estimates. Number of frequencies to be removed
                is specified by the truncate_frequencies argument. Shift perturber adds
                random upward or downward shift in the data value over time continuous
                blocks. If not provided default perturber is combination of block-bootstrap,
                moving-average, and frequency. Default: None
            block_selector (BlockSelector): The block_selector is used to prepare the effective window to
                compute explanation from the user provided parameters. This is used while computing timeseries
                perturbations.
        """
        self._perturbers = []
        self._block_selector = block_selector

        if (perturber_engines is None) or (len(perturber_engines) == 0):
            perturber_engines = [
                dict(type="block-bootstrap"),
                dict(type="moving_average"),
                dict(type="frequency"),
            ]

        for engine in perturber_engines:
            if isinstance(engine, TSPerturber):
                self._perturbers.append(engine)
            elif isinstance(engine, dict):
                assert all([f in engine for f in ["type"]])
                if engine.get("type") == "block-bootstrap":
                    self._perturbers.append(
                        BlockBootstrapPerturber(
                            window_length=engine.get("window_length", 5),
                            block_length=engine.get("block_length", 5),
                            block_swap=engine.get("block_swaps", 2),
                        )
                    )
                elif engine.get("type") == "frequency":
                    raise NotImplementedError
                
                elif engine.get("type") == "moving-average":
                    raise NotImplementedError
                
                elif engine.get("type") == "shift":
                    raise NotImplementedError
                
                elif engine.get("type") == "impute":
                    raise NotImplementedError

        if len(self._perturbers) == 0:
            raise RuntimeError(f"Error: no valid perturber specified!")

    def fit_transform(
        self,
        x: np.ndarray,
        x_exog: np.ndarray = None,
        n: int = 10,
    ):
        counter = Counter(np.random.choice(len(self._perturbers), n))
        data = []
        data_exog = []
        for idx in counter:
            ni = counter.get(idx)
            data += self._perturbers[idx].fit_transform(
                x, n_perturbations=ni, block_selector=self._block_selector
            )
            if x_exog is not None:
                data_exog += self._perturbers[idx].fit_transform(
                    x_exog, n_perturbations=ni
                )
        return data, data_exog