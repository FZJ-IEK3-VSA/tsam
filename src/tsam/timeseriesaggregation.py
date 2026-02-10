import copy
import warnings

import numpy as np
import pandas as pd

from tsam.config import (
    ClusterConfig,
    Distribution,
    ExtremeConfig,
    MinMaxMean,
    SegmentConfig,
)
from tsam.exceptions import LegacyAPIWarning
from tsam.period_aggregation import aggregate_periods  # noqa: F401 (re-exported)
from tsam.pipeline import run_pipeline
from tsam.pipeline.types import PredefParams
from tsam.representations import representations  # noqa: F401 (re-exported)
from tsam.weights import validate_weights

pd.set_option("mode.chained_assignment", None)


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


# Translation maps from old API names to new API names
_CLUSTER_METHOD_MAP = {
    "k_means": "kmeans",
    "k_medoids": "kmedoids",
    "k_maxoids": "kmaxoids",
    "adjacent_periods": "contiguous",
    "averaging": "averaging",
    "hierarchical": "hierarchical",
}

_REPR_METHOD_MAP = {
    "meanRepresentation": "mean",
    "medoidRepresentation": "medoid",
    "maxoidRepresentation": "maxoid",
}

_EXTREME_METHOD_MAP = {
    "None": None,
    "append": "append",
    "new_cluster_center": "new_cluster",
    "replace_cluster_center": "replace",
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

    def _translate_representation(self, method=None):
        """Map old representation_method to new API representation."""
        if method is None:
            method = self.representation_method
        if method is None:
            return None
        if method in ("distributionRepresentation", "durationRepresentation"):
            return Distribution(
                scope="cluster" if self.distribution_period_wise else "global"
            )
        if method == "distributionAndMinMaxRepresentation":
            return Distribution(
                scope="cluster" if self.distribution_period_wise else "global",
                preserve_minmax=True,
            )
        if method == "minmaxmeanRepresentation":
            max_cols = [c for c, r in self.representation_dict.items() if r == "max"]
            min_cols = [c for c, r in self.representation_dict.items() if r == "min"]
            return MinMaxMean(max_columns=max_cols, min_columns=min_cols)
        return _REPR_METHOD_MAP.get(method)

    def _build_pipeline_args(self):
        """Build kwargs for run_pipeline() from old-API parameters."""
        cluster = ClusterConfig(
            method=_CLUSTER_METHOD_MAP[self.cluster_method],
            representation=self._translate_representation(),
            weights=self.weight_dict if self.weight_dict else None,
            normalize_column_means=self.same_mean,
            use_duration_curves=self.sort_values,
            include_period_sums=self.eval_sum_periods,
            solver=self.solver,
        )

        extremes = None
        if self.extreme_period_method != "None":
            extremes = ExtremeConfig(
                method=_EXTREME_METHOD_MAP[self.extreme_period_method],
                max_value=list(self.add_peak_max),
                min_value=list(self.add_peak_min),
                max_period=list(self.add_mean_max),
                min_period=list(self.add_mean_min),
            )
            if not extremes.has_extremes():
                extremes = None

        segments = None
        if self.segmentation:
            seg_repr = self._translate_representation(
                self.segment_representation_method
            )
            segments = SegmentConfig(
                n_segments=self.no_segments,
                representation=seg_repr if seg_repr is not None else "mean",
            )

        predef = None
        if self.predef_cluster_order is not None:
            predef = PredefParams(
                cluster_order=list(self.predef_cluster_order),
                cluster_center_indices=(
                    list(self.predef_cluster_center_indices)
                    if self.predef_cluster_center_indices is not None
                    else None
                ),
                extreme_cluster_idx=(
                    list(self.predef_extreme_cluster_idx)
                    if self.predef_extreme_cluster_idx is not None
                    else None
                ),
                segment_order=(
                    [list(s) for s in self.predef_segment_order]
                    if self.predef_segment_order is not None
                    else None
                ),
                segment_durations=(
                    [list(s) for s in self.predef_segment_durations]
                    if self.predef_segment_durations is not None
                    else None
                ),
                segment_centers=(
                    [list(s) for s in self.predef_segment_centers]
                    if self.predef_segment_centers is not None
                    else None
                ),
            )

        return {
            "data": self.time_series,
            "n_clusters": self.no_typical_periods,
            "n_timesteps_per_period": self.time_steps_per_period,
            "cluster": cluster,
            "extremes": extremes,
            "segments": segments,
            "rescale_cluster_periods": self.rescale_cluster_periods,
            "rescale_exclude_columns": self.rescale_exclude_columns or None,
            "round_decimals": self.round_output,
            "numerical_tolerance": self.numerical_tolerance,
            "temporal_resolution": self.resolution,
            "predef": predef,
        }

    def create_typical_periods(self):
        """
        Clusters the Periods.

        :returns: **self.typical_periods** --  All typical Periods in scaled form.
        """
        # Sort + cast (matches old _pre_process_time_series)
        self.time_series.sort_index(axis=1, inplace=True)
        self.time_series = self.time_series.astype(float)

        # NaN check (must happen before pipeline, same error message)
        if self.time_series.isnull().values.any():
            raise ValueError(
                "Pre processed data includes NaN. Please check the time_series input data."
            )

        # Validate weights before pipeline
        validated = validate_weights(self.time_series.columns, self.weight_dict or None)
        if validated is not None:
            self.weight_dict = validated

        # Run pipeline
        result = run_pipeline(**self._build_pipeline_args())

        # Extract state for properties and other methods
        self._pipeline_result = result
        self._cluster_order = np.array(result.clustering_result.cluster_assignments)
        self._cluster_period_no_occur = result.cluster_counts
        self.cluster_center_indices = (
            list(result.clustering_result.cluster_centers)
            if result.clustering_result.cluster_centers is not None
            else None
        )
        self.extreme_cluster_idx = (
            list(result.clustering_result.extreme_cluster_indices)
            if result.clustering_result.extreme_cluster_indices is not None
            else []
        )
        self.clustering_duration = result.clustering_duration
        self.time_index = result.time_index

        # Segmentation data
        if self.segmentation and result.segmented_df is not None:
            self.segmented_normalized_typical_periods = result.segmented_df

        # typical_periods: alphabetically sorted columns (old API contract)
        self.typical_periods = result.typical_periods.sort_index(axis=1)

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
        if not hasattr(self, "_pipeline_result"):
            self.create_typical_periods()
        self.predicted_data = self._pipeline_result.reconstructed_data.sort_index(
            axis=1
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
        if not hasattr(self, "_pipeline_result"):
            self.create_typical_periods()
        return self._pipeline_result.accuracy_indicators

    def total_accuracy_indicators(self):
        """
        Derives the accuracy indicators over all time series
        """
        return np.sqrt(
            self.accuracy_indicators().pow(2).sum() / len(self.time_series.columns)
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
