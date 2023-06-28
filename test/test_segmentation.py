import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam


def test_segmentation():

    raw = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "examples", "testdata.csv"),
        index_col=0,
    )

    orig_raw = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "examples",
            "results",
            "testperiods_segmentation.csv",
        ),
        index_col=[0, 1, 2],
    )

    starttime = time.time()

    aggregation = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=20,
        hoursPerPeriod=24,
        clusterMethod="hierarchical",
        representationMethod="meanRepresentation",
        segmentation=True,
        noSegments=12,
    )

    typPeriods = aggregation.createTypicalPeriods()

    print("Clustering took " + str(time.time() - starttime))

    # sort the typical days in order to avoid error assertion due to different order
    sortedDaysOrig = orig_raw.groupby(level=0).sum().sort_values("GHI").index
    sortedDaysTest = typPeriods.groupby(level=0).sum().sort_values("GHI").index

    # rearange their order
    orig = orig_raw[typPeriods.columns].unstack().loc[sortedDaysOrig, :].stack()
    test = typPeriods.unstack().loc[sortedDaysTest, :].stack()

    np.testing.assert_array_almost_equal(orig.values, test.values, decimal=4)




def test_representation_in_segmentation():

    segmentationCandidates = np.array([[0.        , 0.38936961, 0.27539063, 0.25      ],
        [0.        , 0.35591778, 0.26841518, 0.25      ],
        [0.        , 0.35045773, 0.265625  , 0.25      ],
        [0.        , 0.36418749, 0.25372024, 0.25      ],
        [0.        , 0.38386857, 0.25167411, 0.25      ],
        [0.        , 0.42710529, 0.24237351, 0.16666667],
        [0.        , 0.5798638 , 0.23707217, 0.1922619 ],
        [0.        , 0.70166596, 0.24507068, 0.16666667],
        [0.01838546, 0.74661296, 0.24739583, 0.18363095],
        [0.06893491, 0.75398663, 0.26041667, 0.16666667],
        [0.0942519 , 0.77160913, 0.28385417, 0.16666667],
        [0.14374472, 0.80191153, 0.3046875 , 0.25      ],
        [0.11999155, 0.79502066, 0.3125    , 0.22678571],
        [0.10016906, 0.77613611, 0.31845238, 0.16666667],
        [0.07438715, 0.76489634, 0.3203125 , 0.16666667],
        [0.0101437 , 0.75082659, 0.31538318, 0.16666667],
        [0.        , 0.74856422, 0.3077567 , 0.16666667],
        [0.        , 0.76062049, 0.29678199, 0.08333333],
        [0.        , 0.78148316, 0.29427083, 0.16666667],
        [0.        , 0.75668439, 0.28738839, 0.16666667],
        [0.        , 0.67461737, 0.2859933 , 0.16666667],
        [0.        , 0.624061  , 0.28041295, 0.16666667],
        [0.        , 0.56076035, 0.2734375 , 0.16666667],
        [0.        , 0.4734255 , 0.27092634, 0.16666667]])

    clusterOrder = np.array([5, 5, 5, 5, 5, 7, 3, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 6, 6,
       4, 4])

    clusterCenters_mean, clusterCenterIndices = tsam.representations(
        segmentationCandidates,
        clusterOrder,
        default="meanRepresentation",
        representationMethod="meanRepresentation",
        distributionPeriodWise=False,
        timeStepsPerPeriod=1,
    )

    clusterCenters_dist, clusterCenterIndices = tsam.representations(
        segmentationCandidates,
        clusterOrder,
        default="meanRepresentation",
        representationMethod="distributionRepresentation",
        distributionPeriodWise=True,
        timeStepsPerPeriod=1,
    )

    assert np.isclose(clusterCenters_mean, clusterCenters_dist).all()


if __name__ == "__main__":
    test_segmentation()
