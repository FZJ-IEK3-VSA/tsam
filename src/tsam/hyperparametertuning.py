import copy
import warnings

import numpy as np
import tqdm

from tsam.exceptions import LegacyAPIWarning
from tsam.timeseriesaggregation import TimeSeriesAggregation


def get_no_periods_for_data_reduction(
    n_raw_timesteps, segments_per_period, data_reduction
):
    """
    Identifies the maximum number of periods which can be set to achieve the required data reduction.

    :param n_raw_timesteps: Number of original time steps. required
    :type n_raw_timesteps: int

    :param segments_per_period: Segments per period. required
    :type segments_per_period: int

    :param data_reduction: Factor by which the resulting dataset should be reduced. required
    :type data_reduction: float

    :returns: **no_typical_periods** --  Number of typical periods that can be set.

    .. deprecated::
        This function is deprecated along with the HyperTunedAggregations class.
    """
    warnings.warn(
        "getNoPeriodsForDataReduction will be removed in tsam v4.0. "
        "Use tsam.tuning.find_optimal_combination() instead.",
        LegacyAPIWarning,
        stacklevel=2,
    )
    return int(np.floor(data_reduction * float(n_raw_timesteps) / segments_per_period))


def get_no_segments_for_data_reduction(
    n_raw_timesteps, typical_periods, data_reduction
):
    """
    Identifies the maximum number of segments which can be set to achieve the required data reduction.

    :param n_raw_timesteps: Number of original time steps. required
    :type n_raw_timesteps: int

    :param typical_periods: Number of typical periods. required
    :type typical_periods: int

    :param data_reduction: Factor by which the resulting dataset should be reduced. required
    :type data_reduction: float

    :returns: **segments_per_period** --  Number of segments per period that can be set.

    .. deprecated::
        This function is deprecated along with the HyperTunedAggregations class.
    """
    warnings.warn(
        "getNoSegmentsForDataReduction will be removed in tsam v4.0. "
        "Use tsam.tuning.find_optimal_combination() instead.",
        LegacyAPIWarning,
        stacklevel=2,
    )
    return int(np.floor(data_reduction * float(n_raw_timesteps) / typical_periods))


# Backward-compatible function aliases (deprecated)
getNoPeriodsForDataReduction = get_no_periods_for_data_reduction
getNoSegmentsForDataReduction = get_no_segments_for_data_reduction


class HyperTunedAggregations:
    def __init__(self, base_aggregation, save_aggregation_history=True, **kwargs):
        """
        A class that does a parameter variation and tuning of the aggregation itself.

        :param base_aggregation: TimeSeriesAggregation object which is used as basis for tuning the hyper parameters. required
        :type base_aggregation: TimeSeriesAggregation

        :param save_aggregation_history: Defines if all aggregations that are created during the tuning and iterations shall be saved under self.aggregation_history.
        :type save_aggregation_history: boolean

        .. deprecated::
            Use :func:`tsam.tuning.find_optimal_combination` or
            :func:`tsam.tuning.find_pareto_front` instead.
        """
        # Translate deprecated camelCase kwargs
        if "saveAggregationHistory" in kwargs:
            warnings.warn(
                "'saveAggregationHistory' is deprecated, use 'save_aggregation_history'.",
                FutureWarning,
                stacklevel=2,
            )
            if "save_aggregation_history" in kwargs:
                raise TypeError(
                    "Cannot specify both 'saveAggregationHistory' and 'save_aggregation_history'"
                )
            save_aggregation_history = kwargs.pop("saveAggregationHistory")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {set(kwargs)}")

        warnings.warn(
            "HyperTunedAggregations will be removed in tsam v4.0. "
            "Use tsam.tuning.find_optimal_combination() or tsam.tuning.find_pareto_front() instead.",
            LegacyAPIWarning,
            stacklevel=2,
        )
        self.base_aggregation = base_aggregation

        if not isinstance(self.base_aggregation, TimeSeriesAggregation):
            raise ValueError(
                "base_aggregation has to be an TimeSeriesAggregation object"
            )

        self._alterable_aggregation = copy.deepcopy(self.base_aggregation)

        self.save_aggregation_history = save_aggregation_history

        self._segment_history = []

        self._period_history = []

        self._rmse_history = []

        if self.save_aggregation_history:
            self.aggregation_history = []

    def _test_aggregation(self, no_typical_periods, no_segments):
        """
        Tests the aggregation for a set of typical periods and segments and returns the RMSE
        """
        self._segment_history.append(no_segments)

        self._period_history.append(no_typical_periods)

        self._alterable_aggregation.no_typical_periods = no_typical_periods

        self._alterable_aggregation.no_segments = no_segments

        self._alterable_aggregation.create_typical_periods()

        self._alterable_aggregation.predict_original_data()

        rmse = self._alterable_aggregation.total_accuracy_indicators()["RMSE"]

        self._rmse_history.append(rmse)

        if self.save_aggregation_history:
            self.aggregation_history.append(copy.copy(self._alterable_aggregation))

        return rmse

    def _delete_test_history(self, index):
        """
        Deletes the defined index from the test history
        """
        del self._segment_history[index]
        del self._period_history[index]
        del self._rmse_history[index]

        if self.save_aggregation_history:
            del self.aggregation_history[index]

    def identify_optimal_segment_period_combination(self, data_reduction):
        """
        Identifies the optimal combination of number of typical periods and number of segments for a given data reduction set.

        :param data_reduction: Factor by which the resulting dataset should be reduced. required
        :type data_reduction: float

        :returns: **no_segments, no_typical_periods** --  The optimal combination of segments and typical periods for the given optimization set.
        """
        if not self.base_aggregation.segmentation:
            raise ValueError(
                "This function does only make sense in combination with 'segmentation' activated."
            )

        n_raw_timesteps = len(self.base_aggregation.time_series.index)

        _max_periods = int(
            float(n_raw_timesteps) / self.base_aggregation.time_steps_per_period
        )
        _max_segments = self.base_aggregation.time_steps_per_period

        # save RMSE
        rmse_history = []

        # correct 0 index of python
        possible_segments = np.arange(_max_segments) + 1
        possible_periods = np.arange(_max_periods) + 1

        # number of time steps of all combinations of segments and periods
        combined_timesteps = np.outer(possible_segments, possible_periods)
        # reduce to valid combinations for targeted data reduction
        reduction_valid_combinations = (
            combined_timesteps <= n_raw_timesteps * data_reduction
        )

        # number of time steps for all feasible combinations
        reduction_valid_timesteps = combined_timesteps * reduction_valid_combinations

        # identify max segments and max period combination
        optimal_periods = np.zeros_like(reduction_valid_timesteps)
        optimal_periods[
            np.arange(reduction_valid_timesteps.shape[0]),
            reduction_valid_timesteps.argmax(axis=1),
        ] = 1
        optimal_segments = np.zeros_like(reduction_valid_timesteps)
        optimal_segments[
            reduction_valid_timesteps.argmax(axis=0),
            np.arange(reduction_valid_timesteps.shape[1]),
        ] = 1

        optimal_index_combo = np.nonzero(optimal_periods * optimal_segments)

        for segment_ix, period_ix in tqdm.tqdm(
            zip(optimal_index_combo[0], optimal_index_combo[1])
        ):
            # derive new typical periods and derive rmse
            rmse_history.append(
                self._test_aggregation(
                    possible_periods[period_ix], possible_segments[segment_ix]
                )
            )

        # take the negative backwards index with the minimal RMSE
        min_index = -list(reversed(rmse_history)).index(min(rmse_history)) - 1
        rmse_min = rmse_history[min_index]

        no_typical_periods = self._period_history[min_index]
        no_segments = self._segment_history[min_index]

        # and return the segment and typical period pair
        return no_segments, no_typical_periods, rmse_min

    def identify_pareto_optimal_aggregation(self, until_total_timesteps=None):
        """
        Identifies the pareto-optimal combination of number of typical periods and number of segments along with a steepest decent approach, starting from the aggregation to a single period and a single segment up to the representation of the full time series.

        :param until_total_timesteps: Number of timesteps until which the pareto-front should be determined. If None, the maximum number of timesteps is chosen.
        :type until_total_timesteps: int


        :returns: None. Check aggregation history for results. All typical Periods in scaled form.
        """
        if not self.base_aggregation.segmentation:
            raise ValueError(
                "This function does only make sense in combination with 'segmentation' activated."
            )

        n_raw_timesteps = len(self.base_aggregation.time_series.index)

        _max_periods = int(
            float(n_raw_timesteps) / self.base_aggregation.time_steps_per_period
        )
        _max_segments = self.base_aggregation.time_steps_per_period

        if until_total_timesteps is None:
            until_total_timesteps = n_raw_timesteps

        progress_bar = tqdm.tqdm(total=until_total_timesteps)

        # starting point
        no_typical_periods = 1
        no_segments = 1
        _rmse_0 = self._test_aggregation(no_typical_periods, no_segments)

        # loop until either segments or periods have reached their maximum
        while (
            no_typical_periods < _max_periods
            and no_segments < _max_segments
            and (no_segments + 1) * no_typical_periods <= until_total_timesteps
            and no_segments * (no_typical_periods + 1) <= until_total_timesteps
        ):
            # test for more segments
            rmse_segments = self._test_aggregation(no_typical_periods, no_segments + 1)
            # test for more periods
            rmse_periods = self._test_aggregation(no_typical_periods + 1, no_segments)

            # RMSE old
            rmse_old = self._rmse_history[-3]

            # segment gradient (RMSE improvement per increased time step number)
            # for segments: for each period on segment added
            rmse_segment_gradient = (rmse_old - rmse_segments) / no_typical_periods
            # for periods: one period with no of segments
            rmse_periods_gradient = (rmse_old - rmse_periods) / no_segments

            # go along the steeper gradient
            if rmse_periods_gradient > rmse_segment_gradient:
                no_typical_periods += 1
                # and delete the search direction which was not pursued
                self._delete_test_history(-2)
            else:
                no_segments += 1
                self._delete_test_history(-1)
            progress_bar.update(no_segments * no_typical_periods - progress_bar.n)

        # afterwards loop over periods and segments exclusively until maximum is reached
        while (
            no_typical_periods < _max_periods
            and no_segments * (no_typical_periods + 1) <= until_total_timesteps
        ):
            no_typical_periods += 1
            self._test_aggregation(no_typical_periods, no_segments)
            progress_bar.update(no_segments * no_typical_periods - progress_bar.n)

        while (
            no_segments < _max_segments
            and (no_segments + 1) * no_typical_periods <= until_total_timesteps
        ):
            no_segments += 1
            self._test_aggregation(no_typical_periods, no_segments)
            progress_bar.update(no_segments * no_typical_periods - progress_bar.n)
        return

    # Backward-compatible method aliases (deprecated)
    identifyOptimalSegmentPeriodCombination = (
        identify_optimal_segment_period_combination
    )
    identifyParetoOptimalAggregation = identify_pareto_optimal_aggregation

    # Backward-compatible property aliases (deprecated)
    @property
    def aggregationHistory(self):
        return self.aggregation_history

    @property
    def _RMSEHistory(self):
        return self._rmse_history

    @property
    def _segmentHistory(self):
        return self._segment_history

    @property
    def _periodHistory(self):
        return self._period_history
