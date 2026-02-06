import copy
import time
import warnings

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tsam.exceptions import LegacyAPIWarning
from tsam.period_aggregation import aggregate_periods
from tsam.representations import representations

pd.set_option("mode.chained_assignment", None)

# max iterator while resacling cluster profiles
MAX_ITERATOR = 20

# tolerance while rescaling cluster periods to meet the annual sum of the original profile
TOLERANCE = 1e-6


# minimal weight that overwrites a weighting of zero in order to carry the profile through the aggregation process
MIN_WEIGHT = 1e-6


def unstack_to_periods(time_series, time_steps_per_period):
    """
    Extend the timeseries to an integer multiple of the period length and
    groups the time series to the periods.

    :param time_series:
    :type time_series: pandas DataFrame

    :param time_steps_per_period: The number of discrete timesteps which describe one period. required
    :type time_steps_per_period: integer

    :returns: - **unstacked_time_series** (pandas DataFrame) -- is stacked such that each row represents a
                candidate period
              - **time_index** (pandas Series index) -- is the modification of the original
                timeseriesindex in case an integer multiple was created
    """
    # init new grouped timeindex
    unstacked_time_series = time_series.copy()

    # initialize new indices
    period_index = []
    step_index = []

    # extend to integer multiple of period length
    if len(time_series) % time_steps_per_period == 0:
        attached_timesteps = 0
    else:
        # calculate number of timesteps which get attached
        attached_timesteps = (
            time_steps_per_period - len(time_series) % time_steps_per_period
        )

        # take these from the head of the original time series
        rep_data = unstacked_time_series.head(attached_timesteps)

        # append them at the end of the time series
        unstacked_time_series = pd.concat([unstacked_time_series, rep_data])

    # create period and step index
    for ii in range(0, len(unstacked_time_series)):
        period_index.append(int(ii / time_steps_per_period))
        step_index.append(ii - int(ii / time_steps_per_period) * time_steps_per_period)

    # save old index
    time_index = copy.deepcopy(unstacked_time_series.index)

    # create new double index and unstack the time series
    unstacked_time_series.index = pd.MultiIndex.from_arrays(
        [step_index, period_index], names=["TimeStep", "PeriodNum"]
    )
    unstacked_time_series = unstacked_time_series.unstack(level="TimeStep")

    return unstacked_time_series, time_index


# Legacy alias
unstackToPeriods = unstack_to_periods


_PARAM_ALIASES = {
    "timeSeries": "time_series",
    "noTypicalPeriods": "no_typical_periods",
    "noSegments": "no_segments",
    "hoursPerPeriod": "hours_per_period",
    "clusterMethod": "cluster_method",
    "evalSumPeriods": "eval_sum_periods",
    "sortValues": "sort_values",
    "sameMean": "same_mean",
    "rescaleClusterPeriods": "rescale_cluster_periods",
    "rescaleExcludeColumns": "rescale_exclude_columns",
    "weightDict": "weight_dict",
    "extremePeriodMethod": "extreme_period_method",
    "representationMethod": "representation_method",
    "representationDict": "representation_dict",
    "distributionPeriodWise": "distribution_period_wise",
    "segmentRepresentationMethod": "segment_representation_method",
    "predefClusterOrder": "predef_cluster_order",
    "predefClusterCenterIndices": "predef_cluster_center_indices",
    "predefExtremeClusterIdx": "predef_extreme_cluster_idx",
    "predefSegmentOrder": "predef_segment_order",
    "predefSegmentDurations": "predef_segment_durations",
    "predefSegmentCenters": "predef_segment_centers",
    "numericalTolerance": "numerical_tolerance",
    "roundOutput": "round_output",
    "addPeakMin": "add_peak_min",
    "addPeakMax": "add_peak_max",
    "addMeanMin": "add_mean_min",
    "addMeanMax": "add_mean_max",
}


class TimeSeriesAggregation:
    """
    Clusters time series data to typical periods.
    """

    CLUSTER_METHODS = [
        "averaging",
        "k_means",
        "k_medoids",
        "k_maxoids",
        "hierarchical",
        "adjacent_periods",
    ]

    REPRESENTATION_METHODS = [
        "meanRepresentation",
        "medoidRepresentation",
        "maxoidRepresentation",
        "minmaxmeanRepresentation",
        "durationRepresentation",
        "distributionRepresentation",
        "distributionAndMinMaxRepresentation",
    ]

    EXTREME_PERIOD_METHODS = [
        "None",
        "append",
        "new_cluster_center",
        "replace_cluster_center",
    ]

    def __init__(
        self,
        time_series=None,
        resolution=None,
        no_typical_periods=10,
        no_segments=10,
        hours_per_period=24,
        cluster_method="hierarchical",
        eval_sum_periods=False,
        sort_values=False,
        same_mean=False,
        rescale_cluster_periods=True,
        rescale_exclude_columns=None,
        weight_dict=None,
        segmentation=False,
        extreme_period_method="None",
        representation_method=None,
        representation_dict=None,
        distribution_period_wise=True,
        segment_representation_method=None,
        predef_cluster_order=None,
        predef_cluster_center_indices=None,
        predef_extreme_cluster_idx=None,
        predef_segment_order=None,
        predef_segment_durations=None,
        predef_segment_centers=None,
        solver="highs",
        numerical_tolerance=1e-13,
        round_output=None,
        add_peak_min=None,
        add_peak_max=None,
        add_mean_min=None,
        add_mean_max=None,
        **kwargs,
    ):
        """
        Initialize the periodly clusters.

        :param time_series: DataFrame with the datetime as index and the relevant
            time series parameters as columns. required
        :type time_series: pandas.DataFrame() or dict

        :param resolution: Resolution of the time series in hours [h]. If time_series is a
            pandas.DataFrame() the resolution is derived from the datetime
            index. optional, default: delta_T in time_series
        :type resolution: float

        :param hours_per_period: Value which defines the length of a cluster period. optional, default: 24
        :type hours_per_period: integer

        :param no_typical_periods: Number of typical Periods - equivalent to the number of clusters. optional, default: 10
        :type no_typical_periods: integer

        :param no_segments: Number of segments in which the typical periods should be subdivided - equivalent to the
            number of inner-period clusters. optional, default: 10
        :type no_segments: integer

        :param cluster_method: Chosen clustering method. optional, default: 'hierarchical'
        :type cluster_method: string

        :param eval_sum_periods: Boolean if in the clustering process also the averaged periodly values
            shall be integrated additional to the periodly profiles as parameters. optional, default: False
        :type eval_sum_periods: boolean

        :param same_mean: Boolean which is used in the normalization procedure. If true, all time series get normalized
            such that they have the same mean value. optional, default: False
        :type same_mean: boolean

        :param sort_values: Boolean if the clustering should be done by the periodly duration
            curves (true) or the original shape of the data. optional (default: False)
        :type sort_values: boolean

        :param rescale_cluster_periods: Decides if the cluster Periods shall get rescaled such that their
            weighted mean value fits the mean value of the original time series. optional (default: True)
        :type rescale_cluster_periods: boolean

        :param weight_dict: Dictionary which weights the profiles. optional (default: None)
        :type weight_dict: dict

        :param extreme_period_method: Method how to integrate extreme Periods. optional, default: 'None'
        :type extreme_period_method: string

        :param representation_method: Chosen representation. optional
        :type representation_method: string

        :param representation_dict: Dictionary which states for each attribute whether the profiles in each cluster
            should be represented by the minimum value or maximum value of each time step.
        :type representation_dict: dict

        :param distribution_period_wise: If duration representation is chosen, you can choose whether the distribution of
            each cluster should be separately preserved or that of the original time series only (default: True)
        :type distribution_period_wise: boolean

        :param numerical_tolerance: Tolerance for numerical issues. optional (default: 1e-13)
        :type numerical_tolerance: float

        :param round_output: Decimals to what the output time series get round. optional (default: None)
        :type round_output: integer

        :param add_peak_min: List of column names which's minimal value shall be added. optional, default: []
        :type add_peak_min: list

        :param add_peak_max: List of column names which's maximal value shall be added. optional, default: []
        :type add_peak_max: list

        :param add_mean_min: List of column names where the period with the cumulative minimal value
            shall be added. optional, default: []
        :type add_mean_min: list

        :param add_mean_max: List of column names where the period with the cumulative maximal value
            shall be added. optional, default: []
        :type add_mean_max: list
        """
        # Translate deprecated camelCase kwargs to snake_case
        for old_name, new_name in _PARAM_ALIASES.items():
            if old_name in kwargs:
                warnings.warn(
                    f"'{old_name}' is deprecated, use '{new_name}'.",
                    FutureWarning,
                    stacklevel=2,
                )
                if new_name in kwargs:
                    raise TypeError(
                        f"Cannot specify both '{old_name}' and '{new_name}'"
                    )
                kwargs[new_name] = kwargs.pop(old_name)

        # Apply translated kwargs as overrides
        time_series = kwargs.pop("time_series", time_series)
        resolution = kwargs.pop("resolution", resolution)
        no_typical_periods = kwargs.pop("no_typical_periods", no_typical_periods)
        no_segments = kwargs.pop("no_segments", no_segments)
        hours_per_period = kwargs.pop("hours_per_period", hours_per_period)
        cluster_method = kwargs.pop("cluster_method", cluster_method)
        eval_sum_periods = kwargs.pop("eval_sum_periods", eval_sum_periods)
        sort_values = kwargs.pop("sort_values", sort_values)
        same_mean = kwargs.pop("same_mean", same_mean)
        rescale_cluster_periods = kwargs.pop(
            "rescale_cluster_periods", rescale_cluster_periods
        )
        rescale_exclude_columns = kwargs.pop(
            "rescale_exclude_columns", rescale_exclude_columns
        )
        weight_dict = kwargs.pop("weight_dict", weight_dict)
        segmentation = kwargs.pop("segmentation", segmentation)
        extreme_period_method = kwargs.pop(
            "extreme_period_method", extreme_period_method
        )
        representation_method = kwargs.pop(
            "representation_method", representation_method
        )
        representation_dict = kwargs.pop("representation_dict", representation_dict)
        distribution_period_wise = kwargs.pop(
            "distribution_period_wise", distribution_period_wise
        )
        segment_representation_method = kwargs.pop(
            "segment_representation_method", segment_representation_method
        )
        predef_cluster_order = kwargs.pop("predef_cluster_order", predef_cluster_order)
        predef_cluster_center_indices = kwargs.pop(
            "predef_cluster_center_indices", predef_cluster_center_indices
        )
        predef_extreme_cluster_idx = kwargs.pop(
            "predef_extreme_cluster_idx", predef_extreme_cluster_idx
        )
        predef_segment_order = kwargs.pop("predef_segment_order", predef_segment_order)
        predef_segment_durations = kwargs.pop(
            "predef_segment_durations", predef_segment_durations
        )
        predef_segment_centers = kwargs.pop(
            "predef_segment_centers", predef_segment_centers
        )
        solver = kwargs.pop("solver", solver)
        numerical_tolerance = kwargs.pop("numerical_tolerance", numerical_tolerance)
        round_output = kwargs.pop("round_output", round_output)
        add_peak_min = kwargs.pop("add_peak_min", add_peak_min)
        add_peak_max = kwargs.pop("add_peak_max", add_peak_max)
        add_mean_min = kwargs.pop("add_mean_min", add_mean_min)
        add_mean_max = kwargs.pop("add_mean_max", add_mean_max)

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {set(kwargs)}")

        warnings.warn(
            "TimeSeriesAggregation is deprecated and will be removed in a future version. "
            "Use tsam.aggregate() instead. See the migration guide in the documentation.",
            LegacyAPIWarning,
            stacklevel=2,
        )
        if add_mean_min is None:
            add_mean_min = []
        if add_mean_max is None:
            add_mean_max = []
        if add_peak_max is None:
            add_peak_max = []
        if add_peak_min is None:
            add_peak_min = []
        if weight_dict is None:
            weight_dict = {}
        self.time_series = time_series

        self.resolution = resolution

        self.hours_per_period = hours_per_period

        self.no_typical_periods = no_typical_periods

        self.no_segments = no_segments

        self.cluster_method = cluster_method

        self.extreme_period_method = extreme_period_method

        self.eval_sum_periods = eval_sum_periods

        self.sort_values = sort_values

        self.same_mean = same_mean

        self.rescale_cluster_periods = rescale_cluster_periods

        self.rescale_exclude_columns = rescale_exclude_columns or []

        self.weight_dict = weight_dict

        self.representation_method = representation_method

        self.representation_dict = representation_dict

        self.distribution_period_wise = distribution_period_wise

        self.segment_representation_method = segment_representation_method

        self.predef_cluster_order = predef_cluster_order

        self.predef_cluster_center_indices = predef_cluster_center_indices

        self.predef_extreme_cluster_idx = predef_extreme_cluster_idx

        self.predef_segment_order = predef_segment_order

        self.predef_segment_durations = predef_segment_durations

        self.predef_segment_centers = predef_segment_centers

        self.solver = solver

        self.numerical_tolerance = numerical_tolerance

        self.segmentation = segmentation

        self.round_output = round_output

        self.add_peak_min = add_peak_min

        self.add_peak_max = add_peak_max

        self.add_mean_min = add_mean_min

        self.add_mean_max = add_mean_max

        self._check_init_args()

        # internal attributes
        self._normalized_mean = None

        return

    def _check_init_args(self):
        # check time_series and set it as pandas DataFrame
        if not isinstance(self.time_series, pd.DataFrame):
            if isinstance(self.time_series, dict) or isinstance(
                self.time_series, np.ndarray
            ):
                self.time_series = pd.DataFrame(self.time_series)
            else:
                raise ValueError(
                    "time_series has to be of type pandas.DataFrame() "
                    + "or of type np.array() "
                    "in initialization of object of class " + type(self).__name__
                )

        # check if extreme periods exist in the dataframe
        for peak in self.add_peak_min:
            if peak not in self.time_series.columns:
                raise ValueError(
                    peak
                    + ' listed in "add_peak_min"'
                    + " does not occur as time_series column"
                )
        for peak in self.add_peak_max:
            if peak not in self.time_series.columns:
                raise ValueError(
                    peak
                    + ' listed in "add_peak_max"'
                    + " does not occur as time_series column"
                )
        for peak in self.add_mean_min:
            if peak not in self.time_series.columns:
                raise ValueError(
                    peak
                    + ' listed in "add_mean_min"'
                    + " does not occur as time_series column"
                )
        for peak in self.add_mean_max:
            if peak not in self.time_series.columns:
                raise ValueError(
                    peak
                    + ' listed in "add_mean_max"'
                    + " does not occur as time_series column"
                )

        # derive resolution from date time index if not provided
        if self.resolution is None:
            try:
                timedelta = self.time_series.index[1] - self.time_series.index[0]
                self.resolution = float(timedelta.total_seconds()) / 3600
            except AttributeError as exc:
                raise ValueError(
                    "'resolution' argument has to be nonnegative float or int"
                    + " or the given timeseries needs a datetime index"
                ) from exc
            except TypeError:
                try:
                    self.time_series.index = pd.to_datetime(self.time_series.index)
                    timedelta = self.time_series.index[1] - self.time_series.index[0]
                    self.resolution = float(timedelta.total_seconds()) / 3600
                except Exception as exc:
                    raise ValueError(
                        "'resolution' argument has to be nonnegative float or int"
                        + " or the given timeseries needs a datetime index"
                    ) from exc

        if not (isinstance(self.resolution, int) or isinstance(self.resolution, float)):
            raise ValueError("resolution has to be nonnegative float or int")

        # check hours_per_period
        if self.hours_per_period is None or self.hours_per_period <= 0:
            raise ValueError("hours_per_period has to be nonnegative float or int")

        # check typical Periods
        if (
            self.no_typical_periods is None
            or self.no_typical_periods <= 0
            or not isinstance(self.no_typical_periods, int)
        ):
            raise ValueError("no_typical_periods has to be nonnegative integer")
        self.time_steps_per_period = int(self.hours_per_period / self.resolution)
        if not self.time_steps_per_period == self.hours_per_period / self.resolution:
            raise ValueError(
                "The combination of hours_per_period and the "
                + "resolution does not result in an integer "
                + "number of time steps per period"
            )
        if self.segmentation:
            if self.no_segments > self.time_steps_per_period:
                warnings.warn(
                    "The number of segments must be less than or equal to the number of time steps per period. "
                    "Segment number is decreased to number of time steps per period."
                )
                self.no_segments = self.time_steps_per_period

        # check cluster_method
        if self.cluster_method not in self.CLUSTER_METHODS:
            raise ValueError(
                "cluster_method needs to be one of "
                + "the following: "
                + f"{self.CLUSTER_METHODS}"
            )

        # check representation_method
        if (
            self.representation_method is not None
            and self.representation_method not in self.REPRESENTATION_METHODS
        ):
            raise ValueError(
                "If specified, representation_method needs to be one of "
                + "the following: "
                + f"{self.REPRESENTATION_METHODS}"
            )

        # check segment_representation_method
        if self.segment_representation_method is None:
            self.segment_representation_method = self.representation_method
        else:
            if self.segment_representation_method not in self.REPRESENTATION_METHODS:
                raise ValueError(
                    "If specified, segment_representation_method needs to be one of "
                    + "the following: "
                    + f"{self.REPRESENTATION_METHODS}"
                )

        # if representation_dict None, represent by maximum time steps in each cluster
        if self.representation_dict is None:
            self.representation_dict = dict.fromkeys(
                list(self.time_series.columns), "mean"
            )
        # sort representation_dict alphabetically to make sure that the min, max or mean function is applied to the right
        # column
        self.representation_dict = (
            pd.Series(self.representation_dict).sort_index(axis=0).to_dict()
        )

        # check extreme_periods
        if self.extreme_period_method not in self.EXTREME_PERIOD_METHODS:
            raise ValueError(
                "extreme_period_method needs to be one of "
                + "the following: "
                + f"{self.EXTREME_PERIOD_METHODS}"
            )

        # check eval_sum_periods
        if not isinstance(self.eval_sum_periods, bool):
            raise ValueError("eval_sum_periods has to be boolean")
        # check sort_values
        if not isinstance(self.sort_values, bool):
            raise ValueError("sort_values has to be boolean")
        # check same_mean
        if not isinstance(self.same_mean, bool):
            raise ValueError("same_mean has to be boolean")
        # check rescale_cluster_periods
        if not isinstance(self.rescale_cluster_periods, bool):
            raise ValueError("rescale_cluster_periods has to be boolean")

        # check predef_cluster_order
        if self.predef_cluster_order is not None:
            if not isinstance(self.predef_cluster_order, (list, np.ndarray)):
                raise ValueError("predef_cluster_order has to be an array or list")
            if self.predef_cluster_center_indices is not None:
                # check predef_cluster_center_indices
                if not isinstance(
                    self.predef_cluster_center_indices, (list, np.ndarray)
                ):
                    raise ValueError(
                        "predef_cluster_center_indices has to be an array or list"
                    )
        elif self.predef_cluster_center_indices is not None:
            raise ValueError(
                'If "predef_cluster_center_indices" is defined, "predef_cluster_order" needs to be defined as well'
            )

        # check predef_segment_order
        if self.predef_segment_order is not None:
            if not isinstance(self.predef_segment_order, (list, tuple)):
                raise ValueError("predef_segment_order has to be a list or tuple")
            if self.predef_segment_durations is None:
                raise ValueError(
                    'If "predef_segment_order" is defined, "predef_segment_durations" '
                    "needs to be defined as well"
                )
            if not isinstance(self.predef_segment_durations, (list, tuple)):
                raise ValueError("predef_segment_durations has to be a list or tuple")
        elif self.predef_segment_durations is not None:
            raise ValueError(
                'If "predef_segment_durations" is defined, "predef_segment_order" '
                "needs to be defined as well"
            )

        if self.predef_segment_centers is not None:
            if self.predef_segment_order is None:
                raise ValueError(
                    'If "predef_segment_centers" is defined, "predef_segment_order" '
                    "needs to be defined as well"
                )
            if not isinstance(self.predef_segment_centers, (list, tuple)):
                raise ValueError("predef_segment_centers has to be a list or tuple")

        return

    def _normalize_time_series(self, same_mean=False):
        """
        Normalizes each time series independently.

        :param same_mean: Decides if the time series should have all the same mean value.
            Relevant for weighting time series. optional (default: False)
        :type same_mean: boolean

        :returns: normalized time series
        """
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_time_series = pd.DataFrame(
            min_max_scaler.fit_transform(self.time_series),
            columns=self.time_series.columns,
            index=self.time_series.index,
        )

        self._normalized_mean = normalized_time_series.mean()
        if same_mean:
            normalized_time_series /= self._normalized_mean

        return normalized_time_series

    def _unnormalize_time_series(self, normalized_time_series, same_mean=False):
        """
        Equivalent to '_normalize_time_series'. Just does the back
        transformation.

        :param normalized_time_series: Time series which should get back transformated. required
        :type normalized_time_series: pandas.DataFrame()

        :param same_mean: Has to have the same value as in _normalize_time_series. optional (default: False)
        :type same_mean: boolean

        :returns: unnormalized time series
        """
        from sklearn import preprocessing

        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit(self.time_series)

        if same_mean:
            normalized_time_series *= self._normalized_mean

        unnormalized_time_series = pd.DataFrame(
            min_max_scaler.inverse_transform(normalized_time_series),
            columns=normalized_time_series.columns,
            index=normalized_time_series.index,
        )

        return unnormalized_time_series

    def _pre_process_time_series(self):
        """
        Normalize the time series, weight them based on the weight dict and
        puts them into the correct matrix format.
        """
        # first sort the time series in order to avoid bug mention in #18
        self.time_series.sort_index(axis=1, inplace=True)

        # convert the dataframe to floats
        self.time_series = self.time_series.astype(float)

        # normalize the time series and group them to periodly profiles
        self.normalized_time_series = self._normalize_time_series(
            same_mean=self.same_mean
        )

        for column in self.weight_dict:
            if self.weight_dict[column] < MIN_WEIGHT:
                print(
                    'weight of "'
                    + str(column)
                    + '" set to the minmal tolerable weighting'
                )
                self.weight_dict[column] = MIN_WEIGHT
            self.normalized_time_series[column] = (
                self.normalized_time_series[column] * self.weight_dict[column]
            )

        self.normalized_periodly_profiles, self.time_index = unstack_to_periods(
            self.normalized_time_series, self.time_steps_per_period
        )

        # check if no NaN is in the resulting profiles
        if self.normalized_periodly_profiles.isnull().values.any():
            raise ValueError(
                "Pre processed data includes NaN. Please check the time_series input data."
            )

    def _post_process_time_series(self, normalized_time_series, apply_weighting=True):
        """
        Neutralizes the weighting the time series back and unnormalizes them.
        """
        if apply_weighting:
            for column in self.weight_dict:
                normalized_time_series[column] = (
                    normalized_time_series[column] / self.weight_dict[column]
                )

        unnormalized_time_series = self._unnormalize_time_series(
            normalized_time_series, same_mean=self.same_mean
        )

        if self.round_output is not None:
            unnormalized_time_series = unnormalized_time_series.round(
                decimals=self.round_output
            )

        return unnormalized_time_series

    def _add_extreme_periods(
        self,
        grouped_series,
        cluster_centers,
        cluster_order,
        extreme_period_method="new_cluster_center",
        add_peak_min=None,
        add_peak_max=None,
        add_mean_min=None,
        add_mean_max=None,
    ):
        """
        Adds different extreme periods based on the to the clustered data.
        """

        # init required dicts and lists
        self.extreme_periods = {}
        extreme_period_no = []

        cc_list = [center.tolist() for center in cluster_centers]

        # check which extreme periods exist in the profile and add them to
        # self.extreme_periods dict
        for column in self.time_series.columns:
            if column in add_peak_max:
                step_no = grouped_series[column].max(axis=1).idxmax()
                if (
                    step_no not in extreme_period_no
                    and grouped_series.loc[step_no, :].values.tolist() not in cc_list
                ):
                    max_col = self._append_col_with(column, " max.")
                    self.extreme_periods[max_col] = {
                        "step_no": step_no,
                        "profile": grouped_series.loc[step_no, :].values,
                        "column": column,
                    }
                    extreme_period_no.append(step_no)

            if column in add_peak_min:
                step_no = grouped_series[column].min(axis=1).idxmin()
                if (
                    step_no not in extreme_period_no
                    and grouped_series.loc[step_no, :].values.tolist() not in cc_list
                ):
                    min_col = self._append_col_with(column, " min.")
                    self.extreme_periods[min_col] = {
                        "step_no": step_no,
                        "profile": grouped_series.loc[step_no, :].values,
                        "column": column,
                    }
                    extreme_period_no.append(step_no)

            if column in add_mean_max:
                step_no = grouped_series[column].mean(axis=1).idxmax()
                if (
                    step_no not in extreme_period_no
                    and grouped_series.loc[step_no, :].values.tolist() not in cc_list
                ):
                    mean_max_col = self._append_col_with(column, " daily max.")
                    self.extreme_periods[mean_max_col] = {
                        "step_no": step_no,
                        "profile": grouped_series.loc[step_no, :].values,
                        "column": column,
                    }
                    extreme_period_no.append(step_no)

            if column in add_mean_min:
                step_no = grouped_series[column].mean(axis=1).idxmin()
                if (
                    step_no not in extreme_period_no
                    and grouped_series.loc[step_no, :].values.tolist() not in cc_list
                ):
                    mean_min_col = self._append_col_with(column, " daily min.")
                    self.extreme_periods[mean_min_col] = {
                        "step_no": step_no,
                        "profile": grouped_series.loc[step_no, :].values,
                        "column": column,
                    }
                    extreme_period_no.append(step_no)

        for period_type in self.extreme_periods:
            # get current related clusters of extreme periods
            self.extreme_periods[period_type]["cluster_no"] = cluster_order[
                self.extreme_periods[period_type]["step_no"]
            ]

            # init new cluster structure
        new_cluster_centers = []
        new_cluster_order = cluster_order
        extreme_cluster_idx = []

        # integrate extreme periods to clusters
        if extreme_period_method == "append":
            # attach extreme periods to cluster centers
            for i, center in enumerate(cluster_centers):
                new_cluster_centers.append(center)
            for i, period_type in enumerate(self.extreme_periods):
                extreme_cluster_idx.append(len(new_cluster_centers))
                new_cluster_centers.append(self.extreme_periods[period_type]["profile"])
                new_cluster_order[self.extreme_periods[period_type]["step_no"]] = (
                    i + len(cluster_centers)
                )

        elif extreme_period_method == "new_cluster_center":
            for i, center in enumerate(cluster_centers):
                new_cluster_centers.append(center)
            # attach extreme periods to cluster centers and consider for all periods
            # if they fit better to the cluster or the extreme period
            for i, period_type in enumerate(self.extreme_periods):
                extreme_cluster_idx.append(len(new_cluster_centers))
                new_cluster_centers.append(self.extreme_periods[period_type]["profile"])
                self.extreme_periods[period_type]["new_cluster_no"] = i + len(
                    cluster_centers
                )

            for i, c_period in enumerate(new_cluster_order):
                # calculate euclidean distance to cluster center
                cluster_dist = sum(
                    (grouped_series.iloc[i].values - cluster_centers[c_period]) ** 2
                )
                for ii, ext_period_type in enumerate(self.extreme_periods):
                    # exclude other extreme periods from adding to the new
                    # cluster center
                    is_other_extreme = False
                    for other_ex_period in self.extreme_periods:
                        if (
                            i == self.extreme_periods[other_ex_period]["step_no"]
                            and other_ex_period != ext_period_type
                        ):
                            is_other_extreme = True
                    # calculate distance to extreme periods
                    extperiod_dist = sum(
                        (
                            grouped_series.iloc[i].values
                            - self.extreme_periods[ext_period_type]["profile"]
                        )
                        ** 2
                    )
                    # choose new cluster relation
                    if extperiod_dist < cluster_dist and not is_other_extreme:
                        new_cluster_order[i] = self.extreme_periods[ext_period_type][
                            "new_cluster_no"
                        ]

        elif extreme_period_method == "replace_cluster_center":
            # Worst Case Clusterperiods
            new_cluster_centers = cluster_centers
            for period_type in self.extreme_periods:
                index = grouped_series.columns.get_loc(
                    self.extreme_periods[period_type]["column"]
                )
                new_cluster_centers[self.extreme_periods[period_type]["cluster_no"]][
                    index
                ] = self.extreme_periods[period_type]["profile"][index]
                if (
                    self.extreme_periods[period_type]["cluster_no"]
                    not in extreme_cluster_idx
                ):
                    extreme_cluster_idx.append(
                        self.extreme_periods[period_type]["cluster_no"]
                    )

        return new_cluster_centers, new_cluster_order, extreme_cluster_idx

    def _append_col_with(self, column, append_with=" max."):
        """Appends a string to the column name. For MultiIndexes, which turn out to be
        tuples when this method is called, only last level is changed"""
        if isinstance(column, str):
            return column + append_with
        elif isinstance(column, tuple):
            col = list(column)
            col[-1] = col[-1] + append_with
            return tuple(col)

    def _rescale_cluster_periods(
        self, cluster_order, cluster_periods, extreme_cluster_idx
    ):
        """
        Rescale the values of the clustered Periods such that mean of each time
        series in the typical Periods fits the mean value of the original time
        series, without changing the values of the extreme_periods.
        """
        # Initialize dict to store rescaling deviations per column
        self._rescale_deviations = {}

        weighting_vec = pd.Series(self._cluster_period_no_occur).values
        columns = list(self.time_series.columns)
        n_clusters = len(self.cluster_periods)
        n_cols = len(columns)
        n_timesteps = self.time_steps_per_period

        # Convert to 3D numpy array for fast operations: (n_clusters, n_cols, n_timesteps)
        arr = np.array(self.cluster_periods).reshape(n_clusters, n_cols, n_timesteps)

        # Indices for non-extreme clusters
        idx_wo_peak = np.delete(np.arange(n_clusters), extreme_cluster_idx)
        extreme_cluster_idx_arr = np.array(extreme_cluster_idx, dtype=int)

        for ci, column in enumerate(columns):
            # Skip columns excluded from rescaling
            if column in self.rescale_exclude_columns:
                continue

            col_data = arr[:, ci, :]  # (n_clusters, n_timesteps)
            sum_raw = self.normalized_periodly_profiles[column].sum().sum()

            # Sum of extreme periods (weighted)
            if len(extreme_cluster_idx_arr) > 0:
                sum_peak = np.sum(
                    weighting_vec[extreme_cluster_idx_arr]
                    * col_data[extreme_cluster_idx_arr, :].sum(axis=1)
                )
            else:
                sum_peak = 0.0

            sum_clu_wo_peak = np.sum(
                weighting_vec[idx_wo_peak] * col_data[idx_wo_peak, :].sum(axis=1)
            )

            # define the upper scale dependent on the weighting of the series
            scale_ub = 1.0
            if self.same_mean:
                scale_ub = (
                    scale_ub
                    * self.time_series[column].max()
                    / self.time_series[column].mean()
                )
            if column in self.weight_dict:
                scale_ub = scale_ub * self.weight_dict[column]

            # difference between predicted and original sum
            diff = abs(sum_raw - (sum_clu_wo_peak + sum_peak))

            # use while loop to rescale cluster periods
            a = 0
            while diff > sum_raw * TOLERANCE and a < MAX_ITERATOR:
                # rescale values (only non-extreme clusters)
                arr[idx_wo_peak, ci, :] *= (sum_raw - sum_peak) / sum_clu_wo_peak

                # reset values higher than the upper scale or less than zero
                arr[:, ci, :] = np.clip(arr[:, ci, :], 0, scale_ub)

                # Handle NaN (replace with 0)
                np.nan_to_num(arr[:, ci, :], copy=False, nan=0.0)

                # calc new sum and new diff to orig data
                col_data = arr[:, ci, :]
                sum_clu_wo_peak = np.sum(
                    weighting_vec[idx_wo_peak] * col_data[idx_wo_peak, :].sum(axis=1)
                )
                diff = abs(sum_raw - (sum_clu_wo_peak + sum_peak))
                a += 1

            # Calculate and store final deviation
            deviation_pct = (diff / sum_raw) * 100 if sum_raw != 0 else 0.0
            converged = a < MAX_ITERATOR
            self._rescale_deviations[column] = {
                "deviation_pct": deviation_pct,
                "converged": converged,
                "iterations": a,
            }

            if not converged and deviation_pct > 0.01:
                warnings.warn(
                    'Max iteration number reached for "'
                    + str(column)
                    + '" while rescaling the cluster periods.'
                    + " The integral of the aggregated time series deviates by: "
                    + str(round(deviation_pct, 2))
                    + "%"
                )

        # Reshape back to 2D: (n_clusters, n_cols * n_timesteps)
        return arr.reshape(n_clusters, -1)

    def _cluster_sorted_periods(self, candidates, n_init=20, n_clusters=None):
        """
        Runs the clustering algorithms for the sorted profiles within the period
        instead of the original profiles. (Duration curve clustering)
        """
        # Vectorized sort: reshape to 3D (periods x columns x timesteps), sort, reshape back
        values = self.normalized_periodly_profiles.values.copy()
        n_periods, n_total = values.shape
        n_cols = len(self.time_series.columns)
        n_timesteps = n_total // n_cols

        # Sort each period's timesteps descending for all columns at once
        # Use stable sort for deterministic tie-breaking across environments
        values_3d = values.reshape(n_periods, n_cols, n_timesteps)
        sorted_cluster_values = (-np.sort(-values_3d, axis=2, kind="stable")).reshape(
            n_periods, -1
        )

        if n_clusters is None:
            n_clusters = self.no_typical_periods

        (
            _alt_cluster_centers,
            self.cluster_center_indices,
            cluster_orders,
        ) = aggregate_periods(
            sorted_cluster_values,
            n_clusters=n_clusters,
            n_iter=30,
            solver=self.solver,
            cluster_method=self.cluster_method,
            representation_method=self.representation_method,
            representation_dict=self.representation_dict,
            distribution_period_wise=self.distribution_period_wise,
            n_timesteps_per_period=self.time_steps_per_period,
        )

        cluster_centers_sorted = []

        # take the clusters and determine the most representative sorted
        # period as cluster center
        for cluster_num in np.unique(cluster_orders):
            indice = np.where(cluster_orders == cluster_num)[0]
            if len(indice) > 1:
                # mean value for each time step for each time series over
                # all Periods in the cluster
                current_mean = sorted_cluster_values[indice].mean(axis=0)
                # index of the period with the lowest distance to the cluster
                # center
                mindist_idx = np.argmin(
                    np.square(sorted_cluster_values[indice] - current_mean).sum(axis=1)
                )
                # append original time series of this period
                medoid = candidates[indice][mindist_idx]

                # append to cluster center
                cluster_centers_sorted.append(medoid)

            else:
                # if only one period is part of the cluster, add this index
                cluster_centers_sorted.append(candidates[indice][0])

        return cluster_centers_sorted, cluster_orders

    def create_typical_periods(self):
        """
        Clusters the Periods.

        :returns: **self.typical_periods** --  All typical Periods in scaled form.
        """
        self._pre_process_time_series()

        # check for additional cluster parameters
        if self.eval_sum_periods:
            evaluation_values = (
                self.normalized_periodly_profiles.stack(future_stack=True, level=0)
                .sum(axis=1)
                .unstack(level=1)
            )
            # how many values have to get deleted later
            del_cluster_params = -len(evaluation_values.columns)
            candidates = np.concatenate(
                (self.normalized_periodly_profiles.values, evaluation_values.values),
                axis=1,
            )
        else:
            del_cluster_params = None
            candidates = self.normalized_periodly_profiles.values

        # skip aggregation procedure for the case of a predefined cluster sequence and get only the correct representation
        if self.predef_cluster_order is not None:
            self._cluster_order = self.predef_cluster_order
            # check if representatives are defined
            if self.predef_cluster_center_indices is not None:
                self.cluster_center_indices = self.predef_cluster_center_indices
                self.cluster_centers = candidates[self.predef_cluster_center_indices]
            else:
                # otherwise take the medoids
                self.cluster_centers, self.cluster_center_indices = representations(
                    candidates,
                    self._cluster_order,
                    default="medoid",
                    representation_method=self.representation_method,
                    representation_dict=self.representation_dict,
                    n_timesteps_per_period=self.time_steps_per_period,
                )
        else:
            cluster_duration = time.time()
            if not self.sort_values:
                # cluster the data
                (
                    self.cluster_centers,
                    self.cluster_center_indices,
                    self._cluster_order,
                ) = aggregate_periods(
                    candidates,
                    n_clusters=self.no_typical_periods,
                    n_iter=100,
                    solver=self.solver,
                    cluster_method=self.cluster_method,
                    representation_method=self.representation_method,
                    representation_dict=self.representation_dict,
                    distribution_period_wise=self.distribution_period_wise,
                    n_timesteps_per_period=self.time_steps_per_period,
                )
            else:
                self.cluster_centers, self._cluster_order = (
                    self._cluster_sorted_periods(
                        candidates, n_clusters=self.no_typical_periods
                    )
                )
            self.clustering_duration = time.time() - cluster_duration

        # get cluster centers without additional evaluation values
        self.cluster_periods = []
        for i, center in enumerate(self.cluster_centers):
            self.cluster_periods.append(center[:del_cluster_params])

        if not self.extreme_period_method == "None":
            (
                self.cluster_periods,
                self._cluster_order,
                self.extreme_cluster_idx,
            ) = self._add_extreme_periods(
                self.normalized_periodly_profiles,
                self.cluster_periods,
                self._cluster_order,
                extreme_period_method=self.extreme_period_method,
                add_peak_min=self.add_peak_min,
                add_peak_max=self.add_peak_max,
                add_mean_min=self.add_mean_min,
                add_mean_max=self.add_mean_max,
            )
        else:
            # Use predefined extreme cluster indices if provided (for transfer/apply)
            if self.predef_extreme_cluster_idx is not None:
                self.extreme_cluster_idx = list(self.predef_extreme_cluster_idx)
            else:
                self.extreme_cluster_idx = []

        # get number of appearance of the the typical periods
        nums, counts = np.unique(self._cluster_order, return_counts=True)
        self._cluster_period_no_occur = {num: counts[ii] for ii, num in enumerate(nums)}

        if self.rescale_cluster_periods:
            self.cluster_periods = self._rescale_cluster_periods(
                self._cluster_order, self.cluster_periods, self.extreme_cluster_idx
            )

        # if additional time steps have been added, reduce the number of occurrence of the typical period
        # which is related to these time steps
        if not len(self.time_series) % self.time_steps_per_period == 0:
            self._cluster_period_no_occur[self._cluster_order[-1]] -= (
                1
                - float(len(self.time_series) % self.time_steps_per_period)
                / self.time_steps_per_period
            )

        # put the clustered data in pandas format and scale back
        self.normalized_typical_periods = (
            pd.concat(
                [
                    pd.Series(s, index=self.normalized_periodly_profiles.columns)
                    for s in self.cluster_periods
                ],
                axis=1,
            )
            .unstack("TimeStep")
            .T
        )

        if self.segmentation:
            from tsam.utils.segmentation import segmentation as run_segmentation

            (
                self.segmented_normalized_typical_periods,
                self.predicted_segmented_normalized_typical_periods,
                self.segment_center_indices,
            ) = run_segmentation(
                self.normalized_typical_periods,
                self.no_segments,
                self.time_steps_per_period,
                representation_method=self.segment_representation_method,
                representation_dict=self.representation_dict,
                distribution_period_wise=self.distribution_period_wise,
                predef_segment_order=self.predef_segment_order,
                predef_segment_durations=self.predef_segment_durations,
                predef_segment_centers=self.predef_segment_centers,
            )
            self.normalized_typical_periods = (
                self.segmented_normalized_typical_periods.reset_index(
                    level=3, drop=True
                )
            )

        self.typical_periods = self._post_process_time_series(
            self.normalized_typical_periods
        )

        # check if original time series boundaries are not exceeded
        exceeds_max = self.typical_periods.max(axis=0) > self.time_series.max(axis=0)
        if exceeds_max.any():
            diff = self.typical_periods.max(axis=0) - self.time_series.max(axis=0)
            exceeding_diff = diff[exceeds_max]
            if exceeding_diff.max() > self.numerical_tolerance:
                warnings.warn(
                    "At least one maximal value of the "
                    + "aggregated time series exceeds the maximal value "
                    + "the input time series for: "
                    + f"{exceeding_diff.to_dict()}"
                    + ". To silence the warning set the 'numerical_tolerance' to a higher value."
                )
        below_min = self.typical_periods.min(axis=0) < self.time_series.min(axis=0)
        if below_min.any():
            diff = self.time_series.min(axis=0) - self.typical_periods.min(axis=0)
            exceeding_diff = diff[below_min]
            if exceeding_diff.max() > self.numerical_tolerance:
                warnings.warn(
                    "Something went wrong... At least one minimal value of the "
                    + "aggregated time series exceeds the minimal value "
                    + "the input time series for: "
                    + f"{exceeding_diff.to_dict()}"
                    + ". To silence the warning set the 'numerical_tolerance' to a higher value."
                )
        return self.typical_periods

    def prepare_enersys_input(self):
        """
        Creates all dictionaries and lists which are required for the energy system
        optimization input.
        """
        warnings.warn(
            '"prepare_enersys_input" is deprecated, since the created attributes can be directly accessed as properties',
            DeprecationWarning,
        )
        return

    @property
    def step_idx(self):
        """
        Index inside a single cluster
        """
        if self.segmentation:
            return [ix for ix in range(0, self.no_segments)]
        else:
            return [ix for ix in range(0, self.time_steps_per_period)]

    @property
    def cluster_period_idx(self):
        """
        Index of the clustered periods
        """
        if not hasattr(self, "_cluster_order"):
            self.create_typical_periods()
        return np.sort(np.unique(self._cluster_order))

    @property
    def cluster_order(self):
        """
        The sequence/order of the typical period to represent
        the original time series
        """
        if not hasattr(self, "_cluster_order"):
            self.create_typical_periods()
        return self._cluster_order

    @property
    def cluster_period_no_occur(self):
        """
        How often does a typical period occur in the original time series
        """
        if not hasattr(self, "_cluster_order"):
            self.create_typical_periods()
        return self._cluster_period_no_occur

    @property
    def cluster_period_dict(self):
        """
        Time series data for each period index as dictionary
        """
        if not hasattr(self, "_cluster_order"):
            self.create_typical_periods()
        if not hasattr(self, "_cluster_period_dict"):
            self._cluster_period_dict = {}
            for column in self.typical_periods:
                self._cluster_period_dict[column] = self.typical_periods[
                    column
                ].to_dict()
        return self._cluster_period_dict

    @property
    def segment_duration_dict(self):
        """
        Segment duration in time steps for each period index as dictionary
        """
        if not hasattr(self, "_cluster_order"):
            self.create_typical_periods()
        if not hasattr(self, "_segment_duration_dict"):
            if self.segmentation:
                self._segment_duration_dict = (
                    self.segmented_normalized_typical_periods.drop(
                        self.segmented_normalized_typical_periods.columns, axis=1
                    )
                    .reset_index(level=3, drop=True)
                    .reset_index(2)
                    .to_dict()
                )
            else:
                self._segment_duration_dict = self.typical_periods.drop(
                    self.typical_periods.columns, axis=1
                )
                self._segment_duration_dict["Segment Duration"] = 1
                self._segment_duration_dict = self._segment_duration_dict.to_dict()
                warnings.warn(
                    "Segmentation is turned off. All segments are consistent the time steps."
                )
        return self._segment_duration_dict

    def predict_original_data(self):
        """
        Predicts the overall time series if every period would be placed in the
        related cluster center

        :returns: **predicted_data** (pandas.DataFrame) -- DataFrame which has the same shape as the original one.
        """
        if not hasattr(self, "_cluster_order"):
            self.create_typical_periods()

        # Select typical periods source based on segmentation
        if self.segmentation:
            typical = self.predicted_segmented_normalized_typical_periods
        else:
            typical = self.normalized_typical_periods

        # Unstack once, then use vectorized indexing to select periods by cluster order
        typical_unstacked = typical.unstack()
        reconstructed = typical_unstacked.loc[list(self._cluster_order)].values

        # Back in matrix form
        clustered_data_df = pd.DataFrame(
            reconstructed,
            columns=self.normalized_periodly_profiles.columns,
            index=self.normalized_periodly_profiles.index,
        )
        clustered_data_df = clustered_data_df.stack(future_stack=True, level="TimeStep")

        # back in form
        self.normalized_predicted_data = pd.DataFrame(
            clustered_data_df.values[: len(self.time_series)],
            index=self.time_series.index,
            columns=self.time_series.columns,
        )
        # Normalize again if same_mean=True to undo in-place modification from create_typical_periods.
        # But NOT for segmentation - predicted_segmented_normalized_typical_periods wasn't modified in-place.
        if self.same_mean and not self.segmentation:
            self.normalized_predicted_data /= self._normalized_mean
        self.predicted_data = self._post_process_time_series(
            self.normalized_predicted_data, apply_weighting=False
        )

        return self.predicted_data

    def index_matching(self):
        """
        Relates the index of the original time series with the indices
        represented by the clusters

        :returns: **time_step_matching** (pandas.DataFrame) -- DataFrame which has the same shape as the original one.
        """
        if not hasattr(self, "_cluster_order"):
            self.create_typical_periods()

        # create aggregated period and time step index lists
        period_index = []
        step_index = []
        for label in self._cluster_order:
            for step in range(self.time_steps_per_period):
                period_index.append(label)
                step_index.append(step)

        # create a dataframe
        time_step_matching = pd.DataFrame(
            [period_index, step_index],
            index=["PeriodNum", "TimeStep"],
            columns=self.time_index,
        ).T

        # if segmentation is chosen, append another column stating which
        if self.segmentation:
            segment_index = []
            for label in self._cluster_order:
                segment_index.extend(
                    np.repeat(
                        self.segmented_normalized_typical_periods.loc[
                            label, :
                        ].index.get_level_values(0),
                        self.segmented_normalized_typical_periods.loc[
                            label, :
                        ].index.get_level_values(1),
                    ).values
                )
            time_step_matching = pd.DataFrame(
                [period_index, step_index, segment_index],
                index=["PeriodNum", "TimeStep", "SegmentIndex"],
                columns=self.time_index,
            ).T

        return time_step_matching

    def accuracy_indicators(self):
        """
        Compares the predicted data with the original time series.

        :returns: **pd.DataFrame(indicator_raw)** (pandas.DataFrame) -- Dataframe containing indicators evaluating the
                    accuracy of the aggregation
        """
        if not hasattr(self, "predicted_data"):
            self.predict_original_data()

        indicator_raw = {
            "RMSE": {},
            "RMSE_duration": {},
            "MAE": {},
        }

        for column in self.normalized_time_series.columns:
            if self.weight_dict:
                orig_ts = self.normalized_time_series[column] / self.weight_dict[column]
            else:
                orig_ts = self.normalized_time_series[column]
            pred_ts = self.normalized_predicted_data[column]
            indicator_raw["RMSE"][column] = np.sqrt(
                mean_squared_error(orig_ts, pred_ts)
            )
            indicator_raw["RMSE_duration"][column] = np.sqrt(
                mean_squared_error(
                    orig_ts.sort_values(ascending=False).reset_index(drop=True),
                    pred_ts.sort_values(ascending=False).reset_index(drop=True),
                )
            )
            indicator_raw["MAE"][column] = mean_absolute_error(orig_ts, pred_ts)

        return pd.DataFrame(indicator_raw)

    def total_accuracy_indicators(self):
        """
        Derives the accuracy indicators over all time series
        """
        return np.sqrt(
            self.accuracy_indicators().pow(2).sum()
            / len(self.normalized_time_series.columns)
        )

    # Backward-compatible method aliases (deprecated)
    createTypicalPeriods = create_typical_periods
    predictOriginalData = predict_original_data
    accuracyIndicators = accuracy_indicators
    totalAccuracyIndicators = total_accuracy_indicators
    prepareEnersysInput = prepare_enersys_input
    indexMatching = index_matching

    # Backward-compatible property aliases (deprecated)
    stepIdx = step_idx
    clusterPeriodIdx = cluster_period_idx
    clusterOrder = cluster_order
    clusterPeriodNoOccur = cluster_period_no_occur
    clusterPeriodDict = cluster_period_dict
    segmentDurationDict = segment_duration_dict
