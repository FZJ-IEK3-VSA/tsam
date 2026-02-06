import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from tsam.representations import representations


def segmentation(
    normalized_typical_periods,
    n_segments,
    n_timesteps_per_period,
    representation_method=None,
    representation_dict=None,
    distribution_period_wise=True,
    predef_segment_order=None,
    predef_segment_durations=None,
    predef_segment_centers=None,
):
    """
    Agglomerative clustering of adjacent time steps within a set of typical periods in order to further reduce the
    temporal resolution within typical periods and to further reduce complexity of input data.

    :param normalized_typical_periods: MultiIndex DataFrame containing the typical periods as first index, the time steps
        within the periods as second index and the attributes as columns.
    :type normalized_typical_periods: pandas DataFrame

    :param n_segments: Number of segments in which the typical periods should be subdivided - equivalent to the number of
        inner-period clusters.
    :type n_segments: integer

    :param n_timesteps_per_period: Number of time steps per period
    :type n_timesteps_per_period: integer

    :param predef_segment_order: Predefined segment assignments per timestep, per typical period.
        If provided, skips clustering and uses these assignments directly.
        List of lists/arrays, one per typical period.
    :type predef_segment_order: list or None

    :param predef_segment_durations: Predefined durations per segment, per typical period.
        Required if predef_segment_order is provided.
        List of lists/arrays, one per typical period.
    :type predef_segment_durations: list or None

    :param predef_segment_centers: Predefined center indices per segment, per typical period.
        If provided with predef_segment_order, uses these as segment centers
        instead of calculating representations.
        List of lists/arrays, one per typical period.
    :type predef_segment_centers: list or None

    :returns:     - **segmented_typical** (pandas DataFrame) --  MultiIndex DataFrame similar to
                    normalized_typical_periods but with segments instead of time steps. Moreover, two additional index
                    levels define the length of each segment and the time step index at which each segment starts.
                  - **predicted_segmented** (pandas DataFrame) -- MultiIndex DataFrame with the same
                    shape of normalized_typical_periods, but with overwritten values derived from segmentation used for
                    prediction of the original periods and accuracy indicators.
                  - **segment_center_indices_list** (list) -- List of segment center indices per typical period.
                    Each entry is a list of indices indicating which timestep is the representative for each segment.
    """
    # Initialize lists for predicted and segmented DataFrame
    segmented_list = []
    predicted_list = []
    segment_center_indices_list = []

    # Get unique period indices
    period_indices = normalized_typical_periods.index.get_level_values(0).unique()
    n_clusters = len(period_indices)

    # Validate predefined segment array lengths
    if predef_segment_order is not None:
        if len(predef_segment_order) != n_clusters:
            raise ValueError(
                f"predef_segment_order has {len(predef_segment_order)} entries "
                f"but data has {n_clusters} periods"
            )
        if (
            predef_segment_durations is not None
            and len(predef_segment_durations) != n_clusters
        ):
            raise ValueError(
                f"predef_segment_durations has {len(predef_segment_durations)} entries "
                f"but data has {n_clusters} periods"
            )
        if (
            predef_segment_centers is not None
            and len(predef_segment_centers) != n_clusters
        ):
            raise ValueError(
                f"predef_segment_centers has {len(predef_segment_centers)} entries "
                f"but data has {n_clusters} periods"
            )

        # Validate segment durations sum to timesteps per period
        if predef_segment_durations is not None:
            for i, durations in enumerate(predef_segment_durations):
                duration_sum = sum(durations)
                if duration_sum != n_timesteps_per_period:
                    raise ValueError(
                        f"predef_segment_durations for period {i} sum to {duration_sum} "
                        f"but n_timesteps_per_period is {n_timesteps_per_period}"
                    )

        # Validate segment center indices are within bounds
        if predef_segment_centers is not None:
            for i, centers in enumerate(predef_segment_centers):
                for idx in centers:
                    if idx < 0 or idx >= n_timesteps_per_period:
                        raise ValueError(
                            f"predef_segment_centers index {idx} for period {i} "
                            f"is out of bounds [0, {n_timesteps_per_period})"
                        )

    # do for each typical period
    for period_i, period_label in enumerate(period_indices):
        # make numpy array with rows containing the segmentation candidates (time steps)
        # and columns as dimensions of the
        segmentation_candidates = np.asarray(
            normalized_typical_periods.loc[period_label, :]
        )

        # Check if using predefined segments for this period
        if predef_segment_order is not None:
            # Use predefined segment order
            cluster_order = np.asarray(predef_segment_order[period_i])

            # Get predefined durations
            segment_no_occur = np.asarray(predef_segment_durations[period_i])

            # Calculate segment numbers and start indices from durations
            seg_no = np.arange(n_segments)
            indices = np.concatenate([[0], np.cumsum(segment_no_occur)[:-1]])

            # The unique cluster order is just 0, 1, 2, ..., n_segments-1 in order
            cluster_order_unique = list(range(n_segments))

            # Determine segment values
            if predef_segment_centers is not None:
                # Use predefined centers directly
                segment_center_indices = list(predef_segment_centers[period_i])
                cluster_centers = segmentation_candidates[segment_center_indices]
            else:
                # Calculate representations from predefined order
                cluster_centers, segment_center_indices = representations(
                    segmentation_candidates,
                    cluster_order,
                    default="mean",
                    representation_method=representation_method,
                    representation_dict=representation_dict,
                    distribution_period_wise=distribution_period_wise,
                    n_timesteps_per_period=1,
                )
        else:
            # Original clustering logic
            # produce adjacency matrix: Each time step is only connected to its preceding and succeeding one
            adjacency_matrix = np.eye(n_timesteps_per_period, k=1) + np.eye(
                n_timesteps_per_period, k=-1
            )
            # execute clustering of adjacent time steps
            if n_segments == 1:
                cluster_order = np.asarray([0] * len(segmentation_candidates))
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=n_segments, linkage="ward", connectivity=adjacency_matrix
                )
                cluster_order = clustering.fit_predict(segmentation_candidates)
            # determine the indices where the segments change and the number of time steps in each segment
            seg_no, indices, segment_no_occur = np.unique(
                cluster_order, return_index=True, return_counts=True
            )
            cluster_order_unique = [cluster_order[index] for index in sorted(indices)]
            # determine the segments' values
            cluster_centers, segment_center_indices = representations(
                segmentation_candidates,
                cluster_order,
                default="mean",
                representation_method=representation_method,
                representation_dict=representation_dict,
                distribution_period_wise=distribution_period_wise,
                n_timesteps_per_period=1,
            )
            # Reorder segment center indices to match temporal order (cluster_order_unique)
            if segment_center_indices is not None:
                segment_center_indices = [
                    segment_center_indices[c] for c in cluster_order_unique
                ]

        # predict each time step of the period by representing it with the corresponding segment's values
        predicted_segmented = (
            pd.DataFrame(cluster_centers, columns=normalized_typical_periods.columns)
            .reindex(cluster_order)
            .reset_index(drop=True)
        )
        # represent the period by the segments in the right order only instead of each time step
        segmented_typical = (
            pd.DataFrame(cluster_centers, columns=normalized_typical_periods.columns)
            .reindex(cluster_order_unique)
            .set_index(np.sort(indices))
        )
        # keep additional information on the lengths of the segments in the right order
        segment_duration = (
            pd.DataFrame(segment_no_occur, columns=["Segment Duration"])
            .reindex(cluster_order_unique)
            .set_index(np.sort(indices))
        )
        # create DataFrame with reduced number of segments together with three indices per period:
        # 1. The segment number
        # 2. The segment duration
        # 3. The index of the original time step, at which the segment starts
        result = segmented_typical.set_index(
            [
                pd.Index(seg_no, name="Segment Step"),
                segment_duration["Segment Duration"],
                pd.Index(np.sort(indices), name="Original Start Step"),
            ]
        )
        # append predicted and segmented DataFrame to list to create a big DataFrame for all periods
        predicted_list.append(predicted_segmented)
        segmented_list.append(result)
        segment_center_indices_list.append(segment_center_indices)

    # create a big DataFrame for all periods for predicted segmented time steps and segments and return
    predicted_segmented = pd.concat(
        predicted_list,
        keys=period_indices,
    ).rename_axis(["", "TimeStep"])
    segmented_typical = pd.concat(
        segmented_list,
        keys=period_indices,
    )
    return (
        segmented_typical,
        predicted_segmented,
        segment_center_indices_list,
    )
