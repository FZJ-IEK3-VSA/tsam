# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 23:25:37 2016

@author: Kotzur
"""

import copy
import time
import warnings

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('mode.chained_assignment', None)

# max iterator while resacling cluster profiles
MAX_ITERATOR = 20

# tolerance while rescaling cluster periods to meet the annual sum of the original profile
TOLERANCE = 1e-6


# minimal weight that overwrites a weighting of zero in order to carry the profile through the aggregation process
MIN_WEIGHT = 1e-6

def unstackToPeriods(timeSeries, timeStepsPerPeriod):
    """
    Extend the timeseries to an integer multiple of the period length and
    groups the time series to the periods.

    Parameters
    -----------
    timeSeries
        pandas.DataFrame()
    timeStepsPerPeriod: integer, required
        The number of discrete timesteps which describe one period.

    Returns
    -------
    unstackedTimeSeries
        pandas.DataFrame() which is stacked such that each row represents a
        candidate period
    timeIndex
        pandas.Series.index which is the modification of the original
        timeseriesindex in case an integer multiple was created
    """
    # init new grouped timeindex
    unstackedTimeSeries = timeSeries.copy()

    # initalize new indices
    periodIndex = []
    stepIndex = []

    # extend to inger multiple of period length
    if len(timeSeries) % timeStepsPerPeriod == 0:
        attached_timesteps = 0
    else:
        # calculate number of timesteps which get attached
        attached_timesteps = timeStepsPerPeriod - \
                             len(timeSeries) % timeStepsPerPeriod

        # take these from the head of the original time series
        rep_data = unstackedTimeSeries.head(attached_timesteps)

        # append them at the end of the time series
        unstackedTimeSeries = unstackedTimeSeries.append(rep_data,
                                                         ignore_index=False)

    # create period and step index
    for ii in range(0, len(unstackedTimeSeries)):
        periodIndex.append(int(ii / timeStepsPerPeriod))
        stepIndex.append(
            ii - int(ii / timeStepsPerPeriod) * timeStepsPerPeriod)

    # save old index
    timeIndex = copy.deepcopy(unstackedTimeSeries.index)

    # create new double index and unstack the time series
    unstackedTimeSeries.index = pd.MultiIndex.from_arrays([stepIndex,
                                                           periodIndex],
                                                          names=['TimeStep',
                                                                 'PeriodNum'])
    unstackedTimeSeries = unstackedTimeSeries.unstack(level='TimeStep')

    return unstackedTimeSeries, timeIndex


def aggregatePeriods(candidates, n_clusters=8,
                     n_iter=100, clusterMethod='k_means', solver='glpk'):
    '''
    Clusters the data based on one of the cluster methods:
        'averaging','k_means','exact k_medoid' or 'hierarchical'

    Parameters
    ----------
    candidates: np.ndarray, required
        Dissimilarity matrix where each row represents a candidate
    n_clusters: int, optional (default: 8)
        Number of aggregated cluster.
    n_iter: int, optional (default: 10)
        Only required for the number of starts of the k-mean algorithm.
    clusterMethod: str, optional (default: 'k_means')
        Chosen clustering algorithm. Possible values are
        'averaging','k_means','exact k_medoid' or 'hierarchical'
    '''

    if clusterMethod == 'hierarchical':
        clusterCenterIndices = []
    else:
        clusterCenterIndices = None

    # cluster the data
    if clusterMethod == 'averaging':
        n_sets = len(candidates)
        if n_sets % n_clusters == 0:
            cluster_size = int(n_sets / n_clusters)
            clusterOrder = [
                [n_cluster] *
                cluster_size for n_cluster in range(n_clusters)]
        else:
            cluster_size = int(n_sets / n_clusters)
            clusterOrder = [
                [n_cluster] *
                cluster_size for n_cluster in range(n_clusters)]
            clusterOrder.append([n_clusters - 1] *
                                int(n_sets - cluster_size * n_clusters))
        clusterOrder = np.hstack(np.array(clusterOrder))
        clusterCenters = []
        for clusterNum in np.unique(clusterOrder):
            indice = np.where(clusterOrder == clusterNum)
            currentMean = candidates[indice].mean(axis=0)
            clusterCenters.append(currentMean)

    if clusterMethod == 'k_means':
        from sklearn.cluster import KMeans
        k_means = KMeans(
            n_clusters=n_clusters,
            max_iter=1000,
            n_init=n_iter,
            tol=1e-4)

        clusterOrder = k_means.fit_predict(candidates)
        clusterCenters = k_means.cluster_centers_

    elif clusterMethod == 'k_medoids':
        from tsam.utils.k_medoids_exact import KMedoids
        k_medoid = KMedoids(n_clusters=n_clusters, solver=solver)

        clusterOrder = k_medoid.fit_predict(candidates)
        clusterCenters = k_medoid.cluster_centers_
    #

    elif clusterMethod == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, linkage='ward')

        clusterOrder = clustering.fit_predict(candidates)

        from sklearn.metrics.pairwise import euclidean_distances
        # set cluster center as medoid
        clusterCenters = []
        for clusterNum in np.unique(clusterOrder):
            indice = np.where(clusterOrder == clusterNum)
            innerDistMatrix = euclidean_distances(candidates[indice])
            mindistIdx = np.argmin(innerDistMatrix.sum(axis=0))
            clusterCenters.append(candidates[indice][mindistIdx])
            clusterCenterIndices.append(indice[0][mindistIdx])

    return clusterCenters, clusterCenterIndices, clusterOrder


class TimeSeriesAggregation(object):
    '''
    Clusters time series data to typical periods.
    '''
    CLUSTER_METHODS = ['averaging', 'k_medoids', 'k_means', 'hierarchical']

    EXTREME_PERIOD_METHODS = [
        'None',
        'append',
        'new_cluster_center',
        'replace_cluster_center']

    def __init__(self, timeSeries, resolution=None, noTypicalPeriods=10,
                 hoursPerPeriod=24, clusterMethod='hierarchical',
                 evalSumPeriods=False, sortValues=False, sameMean=False,
                 rescaleClusterPeriods=True, weightDict=None,
                 extremePeriodMethod='None', solver='glpk',
                 roundOutput = None,
                 addPeakMin=None,
                 addPeakMax=None,
                 addMeanMin=None,
                 addMeanMax=None):
        '''
        Initialize the periodly clusters.

        Parameters
        -----------
        timeSeries: pandas.DataFrame() or dict, required
            DataFrame with the datetime as index and the relevant
            time series parameters as columns.
        resolution: float, optional, default: delta_T in timeSeries
            Resolution of the time series in hours [h]. If timeSeries is a
            pandas.DataFrame() the resolution is derived from the datetime
            index.
        hoursPerPeriod: int, optional, default: 24
            Value which defines the length of a cluster period.
        noTypicalPeriods: int, optional, default: 10
            Number of typical Periods - equivalent to the number of clusters.
        clusterMethod: {'averaging','k_means','k_medoids','hierarchical'},
                        optional, default: 'hierarchical'
            Chosen clustering method.
        evalSumPeriods: boolean, optional, default: False
            Boolean if in the clustering process also the averaged periodly values
            shall be integrated additional to the periodly profiles as parameters.
        sameMean: boolean, optional, default: False
            Boolean which is used in the normalization procedure. If true,
            all time series get normalized such that they have the same mean value.
        sortValues: boolean, optional (default: False)
            Boolean if the clustering should be done by the periodly duration
            curves (true) or the original shape of the data.
        rescaleClusterPeriods: boolean, optional (default: True)
            Decides if the cluster Periods shall get rescaled such that their
            weighted mean value fits the mean value of the original time
            series.
        weightDict: dict, optional (default: None )
            Dictionary which weights the profiles. It is done by scaling
            the time series while the normalization process. Normally all time
            series have a scale from 0 to 1. By scaling them, the values get
            different distances to each other and with this, they are
            differently evaluated while the clustering process.
        extremePeriodMethod: {'None','append','new_cluster_center',
                           'replace_cluster_center'}, optional, default: 'None'
            Method how to integrate extreme Periods (peak demand,
                                                  lowest temperature etc.)
            into to the typical period profiles.
                None: No integration at all.
                'append': append typical Periods to cluster centers
                'new_cluster_center': add the extreme period as additional cluster
                    center. It is checked then for all Periods if they fit better
                    to the this new center or their original cluster center.
                'replace_cluster_center': replaces the cluster center of the
                    cluster where the extreme period belongs to with the periodly
                    profile of the extreme period. (Worst case system design)
        solver: string, optional (default: 'glpk' )
            Solver that is used for k_medoids clustering.
        roundOutput: int, optional (default: None )
            Decimals to what the output time series get round.
        addPeakMin: list, optional, default: []
            List of column names which's minimal value shall be added to the
            typical periods. E.g.: ['Temperature']
        addPeakMax: list, optional, default: []
            List of column names which's maximal value shall be added to the
            typical periods. E.g. ['EDemand', 'HDemand']
        addMeanMin: list, optional, default: []
            List of column names where the period with the cumulative minimal value
            shall be added to the typical periods. E.g. ['Photovoltaic']
        addMeanMax: list, optional, default: []
            List of column names where the period with the cumulative maximal value
            shall be added to the typical periods.
        '''
        if addMeanMin is None:
            addMeanMin = []
        if addMeanMax is None:
            addMeanMax = []
        if addPeakMax is None:
            addPeakMax = []
        if addPeakMin is None:
            addPeakMin = []
        if weightDict is None:
            weightDict = {}
        self.timeSeries = timeSeries

        self.resolution = resolution

        self.hoursPerPeriod = hoursPerPeriod

        self.noTypicalPeriods = noTypicalPeriods

        self.clusterMethod = clusterMethod

        self.extremePeriodMethod = extremePeriodMethod

        self.evalSumPeriods = evalSumPeriods

        self.sortValues = sortValues

        self.sameMean = sameMean

        self.rescaleClusterPeriods = rescaleClusterPeriods

        self.weightDict = weightDict

        self.solver = solver

        self.roundOutput = roundOutput

        self.addPeakMin = addPeakMin

        self.addPeakMax = addPeakMax

        self.addMeanMin = addMeanMin

        self.addMeanMax = addMeanMax

        self._check_init_args()

        return

    def _check_init_args(self):

        # check timeSeries and set it as pandas DataFrame
        if not isinstance(self.timeSeries, pd.DataFrame):
            if isinstance(self.timeSeries, dict):
                self.timeSeries = pd.DataFrame(self.timeSeries)
            elif isinstance(self.timeSeries, np.ndarray):
                self.timeSeries = pd.DataFrame(self.timeSeries)
            else:
                raise ValueError('timeSeries has to be of type pandas.DataFrame() ' +
                                 'or of type np.array() '
                                 'in initialization of object of class ' +
                                 type(self).__name__)

        # check if extreme periods exist in the dataframe
        for peak in self.addPeakMin:
            if peak not in self.timeSeries.columns:
                raise ValueError(peak + ' listed in "addPeakMin"' +
                                 ' does not occure as timeSeries column')
        for peak in self.addPeakMax:
            if peak not in self.timeSeries.columns:
                raise ValueError(peak + ' listed in "addPeakMax"' +
                                 ' does not occure as timeSeries column')
        for peak in self.addMeanMin:
            if peak not in self.timeSeries.columns:
                raise ValueError(peak + ' listed in "addMeanMin"' +
                                 ' does not occure as timeSeries column')
        for peak in self.addMeanMax:
            if peak not in self.timeSeries.columns:
                raise ValueError(peak + ' listed in "addMeanMax"' +
                                 ' does not occure as timeSeries column')

        # derive resolution from date time index if not provided
        if self.resolution is None:
            try:
                timedelta = self.timeSeries.index[1] - self.timeSeries.index[0]
                self.resolution = float(timedelta.total_seconds()) / 3600
            except TypeError:
                try:
                    self.timeSeries.index = pd.to_datetime(self.timeSeries.index)
                    timedelta = self.timeSeries.index[1] - self.timeSeries.index[0]
                    self.resolution = float(timedelta.total_seconds()) / 3600
                except:
                    ValueError("'resolution' argument has to be nonnegative float or int" +
                               " or the given timeseries needs a datetime index")

        if not (isinstance(self.resolution, int) or isinstance(self.resolution, float)):
            raise ValueError("resolution has to be nonnegative float or int")

        # check hoursPerPeriod
        if self.hoursPerPeriod is None or self.hoursPerPeriod <= 0 or \
                not isinstance(self.hoursPerPeriod, int):
            raise ValueError("hoursPerPeriod has to be nonnegative integer")

        # check typical Periods
        if self.noTypicalPeriods is None or self.noTypicalPeriods <= 0 or \
                not isinstance(self.noTypicalPeriods, int):
            raise ValueError("noTypicalPeriods has to be nonnegative integer")
        self.timeStepsPerPeriod = int(self.hoursPerPeriod / self.resolution)
        if not self.timeStepsPerPeriod == self.hoursPerPeriod / self.resolution:
            raise ValueError('The combination of hoursPerPeriod and the '
                             + 'resulution does not result in an integer '
                             + 'number of time steps per period')

        # check clusterMethod
        if self.clusterMethod not in self.CLUSTER_METHODS:
            raise ValueError("clusterMethod needs to be one of " +
                             "the following: " +
                             "{}".format(self.CLUSTER_METHODS))

        # check extremePeriods
        if self.extremePeriodMethod not in self.EXTREME_PERIOD_METHODS:
            raise ValueError("extremePeriodMethod needs to be one of " +
                             "the following: " +
                             "{}".format(self.EXTREME_PERIOD_METHODS))

        # check evalSumPeriods
        if not isinstance(self.evalSumPeriods, bool):
            raise ValueError("evalSumPeriods has to be boolean")
        # check sortValues
        if not isinstance(self.sortValues, bool):
            raise ValueError("sortValues has to be boolean")
        # check sameMean
        if not isinstance(self.sameMean, bool):
            raise ValueError("sameMean has to be boolean")
        # check rescaleClusterPeriods
        if not isinstance(self.rescaleClusterPeriods, bool):
            raise ValueError("rescaleClusterPeriods has to be boolean")

    def _normalizeTimeSeries(self, sameMean=False):
        '''
        Normalizes each time series independently.

        Parameters
        ----------
        sameMean: boolean, optional (default: False)
            Decides if the time series should have all the same mean value.
            Relevant for weighting time series.

        Returns
        ---------
        normalized time series
        '''
        normalizedTimeSeries = pd.DataFrame()
        for column in self.timeSeries:
            if not self.timeSeries[column].max() == self.timeSeries[column].min():  # ignore constant timeseries
                normalizedTimeSeries[column] = (
                                                   self.timeSeries[column] -
                                                   self.timeSeries[column].min()) / \
                                               (self.timeSeries[column].max() -
                                                self.timeSeries[column].min())
                if sameMean:
                    normalizedTimeSeries[column] = normalizedTimeSeries[
                                                       column] / normalizedTimeSeries[column].mean()
            else:
                normalizedTimeSeries[column] = self.timeSeries[column]
        return normalizedTimeSeries

    def _unnormalizeTimeSeries(self, normalizedTimeSeries, sameMean=False):
        '''
        Equivalent to '_normalizeTimeSeries'. Just does the back
        transformation.

        Parameters
        ----------
        normalizedTimeSeries: pandas.DataFrame(), required
            Time series which should get back transformated.
        sameMean: boolean, optional (default: False)
            Has to have the same value as in _normalizeTimeSeries.
        '''
        unnormalizedTimeSeries = pd.DataFrame()
        for column in self.timeSeries:
            if not self.timeSeries[column].max() == self.timeSeries[column].min():  # ignore constant timeseries
                if sameMean:
                    unnormalizedTimeSeries[column] = \
                        normalizedTimeSeries[column] * \
                        (self.timeSeries[column].mean() -
                         self.timeSeries[column].min()) / \
                        (self.timeSeries[column].max() -
                         self.timeSeries[column].min())
                else:
                    unnormalizedTimeSeries[
                        column] = normalizedTimeSeries[column]
                unnormalizedTimeSeries[column] = \
                    unnormalizedTimeSeries[column] * \
                    (self.timeSeries[column].max() -
                     self.timeSeries[column].min()) + \
                    self.timeSeries[column].min()
            else:
                unnormalizedTimeSeries[column] = normalizedTimeSeries[column]
        return unnormalizedTimeSeries

    def _preProcessTimeSeries(self):
        '''
        Normalize the time series, weight them based on the weight dict and
        puts them into the correct matrix format.
        '''
        # first sort the time series in order to avoid bug mention in #18
        self.timeSeries = self.timeSeries.reindex(sorted(self.timeSeries.columns), axis=1)

        # normalize the time series and group them to periodly profiles
        self.normalizedTimeSeries = self._normalizeTimeSeries(
            sameMean=self.sameMean)

        for column in self.weightDict:
            if self.weightDict[column] < MIN_WEIGHT:
                print('weight of "'+ str(column) + '" set to the minmal tolerable weighting')
                self.weightDict[column] = MIN_WEIGHT
            self.normalizedTimeSeries[column] = self.normalizedTimeSeries[
                                                    column] * self.weightDict[column]

        self.normalizedPeriodlyProfiles, self.timeIndex = unstackToPeriods(
            self.normalizedTimeSeries, self.timeStepsPerPeriod)

        # check if no NaN is in the resulting profiles
        if self.normalizedPeriodlyProfiles.isnull().values.any():
            raise ValueError(
                'Pre processed data includes NaN. Please check the timeSeries input data.')

    def _postProcessTimeSeries(self, normalizedTimeSeries):
        '''
        Neutralizes the weighting the time series back and unnormalizes them.
        '''
        for column in self.weightDict:
            normalizedTimeSeries[column] = normalizedTimeSeries[
                                                column] / self.weightDict[column]

        unnormalizedTimeSeries = self._unnormalizeTimeSeries(
            normalizedTimeSeries, sameMean=self.sameMean)

        if self.roundOutput is not None:
            unnormalizedTimeSeries = unnormalizedTimeSeries.round(decimals = self.roundOutput)

        return unnormalizedTimeSeries

    def _addExtremePeriods(self, groupedSeries, clusterCenters,
                           clusterOrder,
                           extremePeriodMethod='new_cluster_center',
                           addPeakMin=None,
                           addPeakMax=None,
                           addMeanMin=None,
                           addMeanMax=None):
        '''
        Adds different extreme periods based on the to the clustered data,
        decribed by the clusterCenters and clusterOrder.

        Parameters
        ----------
        groupedSeries: pandas.DataFrame(), required
            periodly grouped groupedSeries on which basis it should be decided,
            which period is an extreme period.
        clusterCenters: dict, required
            Output from clustering with sklearn.
        clusterOrder: dict, required
            Output from clsutering with sklearn.
        extremePeriodMethod: str, optional(default: 'new_cluster_center' )
            Chosen extremePeriodMethod. The method

        Returns
        -------
        newClusterCenters
            The new cluster centers extended with the extreme periods.
        newClusterOrder
            The new cluster order including the extreme periods.
        extremeClusterIdx
            A list of indices where in the newClusterCenters are the extreme
            periods located.
        '''

        # init required dicts and lists
        if addPeakMin is None:
            addPeakMin = []
        if addPeakMax is None:
            addPeakMax = []
        if addMeanMin is None:
            addMeanMin = []
        if addMeanMax is None:
            addMeanMax = []
        self.extremePeriods = {}
        extremePeriodNo = []

        ccList = [center.tolist() for center in clusterCenters]

        # check which extreme periods exist in the profile and add them to
        # self.extremePeriods dict
        for column in self.timeSeries.columns:

            if column in addPeakMax:
                stepNo = groupedSeries[column].max(axis=1).idxmax()
                # add only if stepNo is not already in extremePeriods
                # if it is not already a cluster center
                if stepNo not in extremePeriodNo and groupedSeries.loc[stepNo,:].values.tolist() not in ccList:
                    self.extremePeriods[column + ' max.'] = \
                        {'stepNo': stepNo,
                         'profile': groupedSeries.loc[stepNo,:].values,
                         'column': column}
                    extremePeriodNo.append(stepNo)

            if column in addPeakMin:
                stepNo = groupedSeries[column].min(axis=1).idxmin()
                # add only if stepNo is not already in extremePeriods
                # if it is not already a cluster center
                if stepNo not in extremePeriodNo and groupedSeries.loc[stepNo,:].values.tolist() not in ccList:
                    self.extremePeriods[column + ' min.'] = \
                        {'stepNo': stepNo,
                         'profile': groupedSeries.loc[stepNo,:].values,
                         'column': column}
                    extremePeriodNo.append(stepNo)

            if column in addMeanMax:
                stepNo = groupedSeries[column].mean(axis=1).idxmax()
                # add only if stepNo is not already in extremePeriods
                # if it is not already a cluster center
                if stepNo not in extremePeriodNo and groupedSeries.loc[stepNo,:].values.tolist() not in ccList:
                    self.extremePeriods[column + ' daily min.'] = \
                        {'stepNo': stepNo,
                         'profile': groupedSeries.loc[stepNo,:].values,
                         'column': column}
                    extremePeriodNo.append(stepNo)

            if column in addMeanMin:
                stepNo = groupedSeries[column].mean(axis=1).idxmin()
                # add only if stepNo is not already in extremePeriods and
                # if it is not already a cluster center
                if stepNo not in extremePeriodNo and groupedSeries.loc[stepNo,:].values.tolist() not in ccList:
                    self.extremePeriods[column + ' daily min.'] = \
                        {'stepNo': stepNo,
                         'profile': groupedSeries.loc[stepNo,:].values,
                         'column': column}
                    extremePeriodNo.append(stepNo)

        for periodType in self.extremePeriods:
            # get current related clusters of extreme periods
            self.extremePeriods[periodType]['clusterNo'] = clusterOrder[
                self.extremePeriods[periodType]['stepNo']]

            # init new cluster structure
        newClusterCenters = []
        newClusterOrder = clusterOrder
        extremeClusterIdx = []

        # integrate extreme periods to clusters
        if extremePeriodMethod == 'append':
            # attach extreme periods to cluster centers
            for i, cluster_center in enumerate(clusterCenters):
                newClusterCenters.append(cluster_center)
            for i, periodType in enumerate(self.extremePeriods):
                extremeClusterIdx.append(len(newClusterCenters))
                newClusterCenters.append(
                    self.extremePeriods[periodType]['profile'])
                newClusterOrder[self.extremePeriods[periodType]['stepNo']] = i + len(clusterCenters)

        elif extremePeriodMethod == 'new_cluster_center':
            for i, cluster_center in enumerate(clusterCenters):
                newClusterCenters.append(cluster_center)
            # attach extrem periods to cluster centers and consider for all periods
            # if the fit better to the cluster or the extrem period
            for i, periodType in enumerate(self.extremePeriods):
                extremeClusterIdx.append(len(newClusterCenters))
                newClusterCenters.append(
                    self.extremePeriods[periodType]['profile'])
                self.extremePeriods[periodType][
                    'newClusterNo'] = i + len(clusterCenters)

            for i, cPeriod in enumerate(newClusterOrder):
                # caclulate euclidean distance to cluster center
                cluster_dist = sum(
                    (groupedSeries.iloc[i].values - clusterCenters[cPeriod]) ** 2)
                for ii, extremPeriodType in enumerate(self.extremePeriods):
                    # exclude other extreme periods from adding to the new
                    # cluster center
                    isOtherExtreme = False
                    for otherExPeriod in self.extremePeriods:
                        if (i == self.extremePeriods[otherExPeriod]['stepNo']
                                and otherExPeriod != extremPeriodType):
                            isOtherExtreme = True
                    # calculate distance to extreme periods
                    extperiod_dist = sum(
                        (groupedSeries.iloc[i].values -
                         self.extremePeriods[extremPeriodType]['profile']) ** 2)
                    # choose new cluster relation
                    if extperiod_dist < cluster_dist and not isOtherExtreme:
                        newClusterOrder[i] = self.extremePeriods[
                            extremPeriodType]['newClusterNo']

        elif extremePeriodMethod == 'replace_cluster_center':
            # Worst Case Clusterperiods
            newClusterCenters = clusterCenters
            for periodType in self.extremePeriods:
                index = groupedSeries.columns.get_loc(
                    self.extremePeriods[periodType]['column'])
                newClusterCenters[self.extremePeriods[periodType]['clusterNo']][index] = \
                    self.extremePeriods[periodType]['profile'][index]
                if not self.extremePeriods[periodType]['clusterNo'] in extremeClusterIdx:
                    extremeClusterIdx.append(
                        self.extremePeriods[periodType]['clusterNo'])

        else:
            raise NotImplementedError('Chosen "extremePeriodMethod": ' +
                                      str(extremePeriodMethod) + ' is ' +
                                      'not implemented.')

        return newClusterCenters, newClusterOrder, extremeClusterIdx

    def _rescaleClusterPeriods(
            self, clusterOrder, clusterPeriods, extremeClusterIdx):
        '''
        Rescale the values of the clustered Periods such that mean of each time
        series in the typical Periods fits the mean value of the original time
        series, without changing the values of the extremePeriods.
        '''
        weightingVec = pd.Series(self._clusterPeriodNoOccur).values
        typicalPeriods = pd.DataFrame(
            clusterPeriods, columns=self.normalizedPeriodlyProfiles.columns)
        idx_wo_peak = np.delete(typicalPeriods.index, extremeClusterIdx)
        for column in self.timeSeries.columns:
            diff = 1
            sum_raw = self.normalizedPeriodlyProfiles[column].sum().sum()
            sum_peak = sum(
                weightingVec[extremeClusterIdx] *
                typicalPeriods[column].loc[extremeClusterIdx,:].sum(
                    axis=1))
            sum_clu_wo_peak = sum(
                weightingVec[idx_wo_peak] *
                typicalPeriods[column].loc[idx_wo_peak,:].sum(
                    axis=1))

            # define the upper scale dependent on the weighting of the series
            scale_ub = 1.0
            if self.sameMean:
                scale_ub = scale_ub * self.timeSeries[column].max(
                ) / self.timeSeries[column].mean()
            if column in self.weightDict:
                scale_ub = scale_ub * self.weightDict[column]

            # difference between predicted and original sum
            diff = abs(sum_raw - (sum_clu_wo_peak + sum_peak))

            # use while loop to rescale cluster periods
            a = 0
            while diff > sum_raw * TOLERANCE and a < MAX_ITERATOR:
                # rescale values
                typicalPeriods.loc[idx_wo_peak, column] = \
                    (typicalPeriods[column].loc[idx_wo_peak,:].values *
                     (sum_raw - sum_peak) / sum_clu_wo_peak)

                # reset values higher than the upper sacle or less than zero
                typicalPeriods[column][
                    typicalPeriods[column] > scale_ub] = scale_ub
                typicalPeriods[column][
                    typicalPeriods[column] < 0.0] = 0.0

                typicalPeriods[column] = typicalPeriods[
                    column].fillna(0.0)

                # calc new sum and new diff to orig data
                sum_clu_wo_peak = sum(
                    weightingVec[idx_wo_peak] *
                    typicalPeriods[column].loc[idx_wo_peak,:].sum(
                        axis=1))
                diff = abs(sum_raw - (sum_clu_wo_peak + sum_peak))
                a += 1
            if a == MAX_ITERATOR:
                deviation = str(round((diff/sum_raw)*100,2))
                warnings.warn(
                    'Max iteration number reached for "'+ str(column)
                    +'" while rescaling the cluster periods.' + 
                    ' The integral of the aggregated time series deviates by: ' 
                    + deviation + '%')
        return typicalPeriods.values

    def _clusterSortedPeriods(self, candidates, n_init=20):
        '''
        Runs the clustering algorithms for the sorted profiles within the period
        instead of the original profiles. (Duration curve clustering)
        '''
        # initialize
        normalizedSortedPeriodlyProfiles = copy.deepcopy(
            self.normalizedPeriodlyProfiles)
        for column in self.timeSeries.columns:
            # sort each period individually
            df = normalizedSortedPeriodlyProfiles[column]
            values = df.values
            values.sort(axis=1)
            values = values[:, ::-1]
            normalizedSortedPeriodlyProfiles[column] = pd.DataFrame(
                values, df.index, df.columns)
        sortedClusterValues = normalizedSortedPeriodlyProfiles.values

        clusterOrders_iter = []
        clusterCenters_iter = []
        distanceMedoid_iter = []

        for i in range(n_init):
            altClusterCenters, clusterCenterIndices, clusterOrders_C = (
                aggregatePeriods(
                    sortedClusterValues, n_clusters=self.noTypicalPeriods,
                    n_iter=30, solver=self.solver,
                    clusterMethod=self.clusterMethod))

            clusterCenters_C = []
            distanceMedoid_C = []

            # take the clusters and determine the most representative sorted
            # period as cluster center
            for clusterNum in np.unique(clusterOrders_C):
                indice = np.where(clusterOrders_C == clusterNum)[0]
                if len(indice) > 1:
                    # mean value for each time step for each time series over
                    # all Periods in the cluster
                    currentMean_C = sortedClusterValues[indice].mean(axis=0)
                    # index of the period with the lowest distance to the cluster
                    # center
                    mindistIdx_C = np.argmin(
                        np.square(
                            sortedClusterValues[indice] -
                            currentMean_C).sum(
                            axis=1))
                    # append original time series of this period
                    medoid_C = candidates[indice][mindistIdx_C]

                    # append to cluster center
                    clusterCenters_C.append(medoid_C)

                    # calculate metrix for evaluation
                    distanceMedoid_C.append(
                        abs(candidates[indice] - medoid_C).sum())

                else:
                    # if only on period is part of the cluster, add this index
                    clusterCenters_C.append(candidates[indice][0])
                    distanceMedoid_C.append(0)

            # collect matrix
            distanceMedoid_iter.append(abs(sum(distanceMedoid_C)))
            clusterCenters_iter.append(clusterCenters_C)
            clusterOrders_iter.append(clusterOrders_C)

        bestFit = np.argmin(distanceMedoid_iter)

        return clusterCenters_iter[bestFit], clusterOrders_iter[bestFit]

    def createTypicalPeriods(self):
        '''
        Clusters the Periods.

        Returns
        -------
        self.clusterPeriods
            All typical Periods in scaled form.
        '''
        self._preProcessTimeSeries()

        # check for additional cluster parameters
        if self.evalSumPeriods:
            evaluationValues = self.normalizedPeriodlyProfiles.stack(
                level=0).sum(
                axis=1).unstack(
                level=1)
            # how many values have to get deleted later
            delClusterParams = -len(evaluationValues.columns)
            candidates = np.concatenate(
                (self.normalizedPeriodlyProfiles.values, evaluationValues.values),
                axis=1)
        else:
            delClusterParams = None
            candidates = self.normalizedPeriodlyProfiles.values

        cluster_duration = time.time()
        if not self.sortValues:
            # cluster the data
            self.clusterCenters, self.clusterCenterIndices, \
                self._clusterOrder = aggregatePeriods(
                candidates, n_clusters=self.noTypicalPeriods, n_iter=100,
                solver=self.solver, clusterMethod=self.clusterMethod)
        else:
            self.clusterCenters, self._clusterOrder = self._clusterSortedPeriods(
                candidates)
        self.clusteringDuration = time.time() - cluster_duration

        # get cluster centers without additional evaluation values
        self.clusterPeriods = []
        for i, cluster_center in enumerate(self.clusterCenters):
            self.clusterPeriods.append(cluster_center[:delClusterParams])

        if not self.extremePeriodMethod == 'None':
            # overwrite clusterPeriods and clusterOrder
            self.clusterPeriods, self._clusterOrder, self.extremeClusterIdx = \
                self._addExtremePeriods(self.normalizedPeriodlyProfiles,
                                        self.clusterPeriods,
                                        self._clusterOrder,
                                        extremePeriodMethod=self.extremePeriodMethod,
                                        addPeakMin=self.addPeakMin,
                                        addPeakMax=self.addPeakMax,
                                        addMeanMin=self.addMeanMin,
                                        addMeanMax=self.addMeanMax)
        else:
            self.extremeClusterIdx = []

        # get number of appearance of the the typical periods
        nums, counts = np.unique(self._clusterOrder, return_counts=True)
        self._clusterPeriodNoOccur = {num: counts[ii] for ii, num in enumerate(nums)}

        if self.rescaleClusterPeriods:
            self.clusterPeriods = self._rescaleClusterPeriods(
                self._clusterOrder, self.clusterPeriods, self.extremeClusterIdx)

        # if additional time steps have been added, reduce the number of occurance of the typical period
        # which is related to these time steps
        if not len(self.timeSeries) % self.timeStepsPerPeriod == 0:
            self._clusterPeriodNoOccur[self._clusterOrder[-1]] -= (
                1 - float(len(self.timeSeries) % self.timeStepsPerPeriod) / self.timeStepsPerPeriod)

        # put the clustered data in pandas format and scale back
        clustered_data_raw = pd.DataFrame(
            self.clusterPeriods,
            columns=self.normalizedPeriodlyProfiles.columns).stack(
            level='TimeStep')

        self.typicalPeriods = self._postProcessTimeSeries(clustered_data_raw)

        # check if original time series boundaries are not exceeded
        if np.array(self.typicalPeriods.max(axis=0) > self.timeSeries.max(axis=0)).any():
            warnings.warn("Something went wrong: At least one maximal value of the aggregated time series exceeds the maximal value the input time series")
        if np.array(self.typicalPeriods.min(axis=0) < self.timeSeries.min(axis=0)).any():
            warnings.warn("Something went wrong: At least one minimal value of the aggregated time series exceeds the minimal value the input time series")
        return self.typicalPeriods

    def prepareEnersysInput(self):
        '''
        Creates all dictionaries and lists which are required for the energysystem
        optimization input.
        '''
        warnings.warn(
            '"prepareEnersysInput" is deprecated, since the created attributes can be directly accessed as properties',
            DeprecationWarning)
        return

    @property
    def stepIdx(self):
        '''
        Index inside a single cluster
        '''
        return [ix for ix in range(0, self.timeStepsPerPeriod)]

    @property
    def clusterPeriodIdx(self):
        '''
        Index of the clustered periods
        '''
        if not hasattr(self, 'clusterOrder'):
            self.createTypicalPeriods()
        return np.sort(np.unique(self._clusterOrder))

    @property
    def clusterOrder(self):
        '''
        How often does an typical period occure in the original time series
        '''
        if not hasattr(self, '_clusterOrder'):
            self.createTypicalPeriods()
        return self._clusterOrder

    @property
    def clusterPeriodNoOccur(self):
        '''
        How often does an typical period occure in the original time series
        '''
        if not hasattr(self, 'clusterOrder'):
            self.createTypicalPeriods()
        return self._clusterPeriodNoOccur

    @property
    def clusterPeriodDict(self):
        '''
        Time series data for each period index as dictionary
        '''
        if not hasattr(self, '_clusterOrder'):
            self.createTypicalPeriods()
        if not hasattr(self, '_clusterPeriodDict'):
            self._clusterPeriodDict = {}
            for column in self.typicalPeriods:
                self._clusterPeriodDict[column] = self.typicalPeriods[column].to_dict()
        return self._clusterPeriodDict

    def predictOriginalData(self):
        '''
        Predicts the overall time series if every period would be placed in the
        related cluster center

        Returns
        -------
        pandas.DataFrame
            DataFrame which has the same shape as the original one.
        '''
        if not hasattr(self, '_clusterOrder'):
            self.createTypicalPeriods()

        new_data = []
        for label in self._clusterOrder:
            new_data.append(self.clusterPeriods[label])

        # back in matrix
        clustered_data_df = \
            pd.DataFrame(new_data,
                         columns=self.normalizedPeriodlyProfiles.columns,
                         index=self.normalizedPeriodlyProfiles.index)
        clustered_data_df = clustered_data_df.stack(level='TimeStep')

        # back in form
        self.normalizedPredictedData = \
            pd.DataFrame(clustered_data_df.values[:len(self.timeSeries)],
                         index=self.timeSeries.index,
                         columns=self.timeSeries.columns)
        self.predictedData = self._postProcessTimeSeries(
            self.normalizedPredictedData)

        return self.predictedData

    def indexMatching(self):
        '''
        Relates the index of the original time series with the indices
        represented by the clusters

        Returns
        -------
        pandas.DataFrame
            DataFrame which has the same shape as the original one.
        '''
        if not hasattr(self, '_clusterOrder'):
            self.createTypicalPeriods()

        # create aggregated period and timestep index lists
        periodIndex = []
        stepIndex = []
        for label in self._clusterOrder:
            for step in range(self.timeStepsPerPeriod):
                periodIndex.append(label)
                stepIndex.append(step)

        # create a dataframe
        timeStepMatching = \
            pd.DataFrame([periodIndex, stepIndex],
                         index=['PeriodNum', 'TimeStep'],
                         columns=self.timeIndex).T

        return timeStepMatching

    def accuracyIndicators(self):
        '''
        Compares the predicted data with the orginal time series.

        Returns
        -------
        pandas.DataFrame
            Dataframe containing indicators evaluating the accuracy of the
            aggregation
        '''
        if not hasattr(self, 'predictedData'):
            self.predictOriginalData()

        indicatorRaw = {
            'RMSE': {},
            'RMSE_duration': {},
            'MAE': {}}  # 'Silhouette score':{},

        for column in self.normalizedTimeSeries.columns:
            origTS = self.normalizedTimeSeries[column]
            predTS = self.normalizedPredictedData[column]
            indicatorRaw['RMSE'][column] = np.sqrt(
                mean_squared_error(origTS, predTS))
            indicatorRaw['RMSE_duration'][column] = np.sqrt(mean_squared_error(
                origTS.sort_values(ascending=False).reset_index(drop=True),
                predTS.sort_values(ascending=False).reset_index(drop=True)))
            indicatorRaw['MAE'][column] = mean_absolute_error(origTS, predTS)

        return pd.DataFrame(indicatorRaw)
