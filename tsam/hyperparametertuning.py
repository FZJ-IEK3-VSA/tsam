# -*- coding: utf-8 -*-

import copy

import numpy as np

import tqdm

from tsam.timeseriesaggregation import TimeSeriesAggregation
        
def getNoPeriodsForDataReduction(noRawTimeSteps, segmentsPerPeriod, dataReduction):
    """        
    Identifies the maximum number of periods which can be set to achieve the required data reduction.

    :param noRawTimeSteps: Number of original time steps. required
    :type noRawTimeSteps: int

    :param segmentsPerPeriod: Segments per period. required
    :type segmentsPerPeriod: int

    :param dataReduction: Factor by which the resulting dataset should be reduced. required
    :type dataReduction: float

    :returns: **noTypicalPeriods** --  Number of typical periods that can be set.
    """
    return int(np.floor(dataReduction * float(noRawTimeSteps)/segmentsPerPeriod))

def getNoSegmentsForDataReduction(noRawTimeSteps, typicalPeriods, dataReduction):
    """        
    Identifies the maximum number of segments which can be set to achieve the required data reduction.

    :param noRawTimeSteps: Number of original time steps. required
    :type noRawTimeSteps: int

    :param typicalPeriods: Number of typical periods. required
    :type typicalPeriods: int

    :param dataReduction: Factor by which the resulting dataset should be reduced. required
    :type dataReduction: float

    :returns: **segmentsPerPeriod** --  Number of segments per period that can be set.
    """
    return int(np.floor(dataReduction * float(noRawTimeSteps)/typicalPeriods))




class HyperTunedAggregations(object):

    def __init__(self, base_aggregation, saveAggregationHistory=True):
        """
        A class that does a parameter variation and tuning of the aggregation itself.

        :param base_aggregation: TimeSeriesAggregation object which is used as basis for tuning the hyper parameters. required
        :type base_aggregation: TimeSeriesAggregation

        :param saveAggregationHistory: Defines if all aggregations that are created during the tuning and iterations shall be saved under self.aggregationHistory.
        :type saveAggregationHistory: boolean
        """
        self.base_aggregation = base_aggregation

        if not isinstance(self.base_aggregation, TimeSeriesAggregation):
            raise ValueError(
                "base_aggregation has to be an TimeSeriesAggregation object"
            )

        self._alterableAggregation=copy.deepcopy(self.base_aggregation)

        self.saveAggregationHistory=saveAggregationHistory

        self._segmentHistory=[]

        self._periodHistory=[]

        self._RMSEHistory=[]

        if self.saveAggregationHistory:
            self.aggregationHistory=[]
            

        
        
    def _testAggregation(self, noTypicalPeriods, noSegments):
        """
        Tests the aggregation for a set of typical periods and segments and returns the RMSE
        """
        self._segmentHistory.append(noSegments)

        self._periodHistory.append(noTypicalPeriods)

        self._alterableAggregation.noTypicalPeriods=noTypicalPeriods

        self._alterableAggregation.noSegments=noSegments

        self._alterableAggregation.createTypicalPeriods()

        self._alterableAggregation.predictOriginalData()

        RMSE=self._alterableAggregation.totalAccuracyIndicators()["RMSE"]

        self._RMSEHistory.append(RMSE)

        if self.saveAggregationHistory:
            self.aggregationHistory.append(copy.copy(self._alterableAggregation))

        return RMSE
	
    def _deleteTestHistory(self, index):
        """
        Delelets the defined index from the test history
        """
        del self._segmentHistory[index]
        del self._periodHistory[index]
        del self._RMSEHistory[index]
        
        if self.saveAggregationHistory:
            del self.aggregationHistory[index]

				
    def identifyOptimalSegmentPeriodCombination(self, dataReduction):
        """        
        Identifies the optimal combination of number of typical periods and number of segments for a given data reduction set.

        :param dataReduction: Factor by which the resulting dataset should be reduced. required
        :type dataReduction: float

        :returns: **noSegments, noTypicalperiods** --  The optimal combination of segments and typical periods for the given optimization set.
        """
        if not self.base_aggregation.segmentation:
            raise ValueError("This function does only make sense in combination with 'segmentation' activated.")

        noRawTimeSteps=len(self.base_aggregation.timeSeries.index)
        # derive the minimum of periods allowed for this data reduction as starting point
        _minPeriods = getNoPeriodsForDataReduction(noRawTimeSteps, self.base_aggregation.timeStepsPerPeriod, dataReduction)
        # get the maximum number of periods as limit for the convergence
        _maxPeriods = min(getNoPeriodsForDataReduction(noRawTimeSteps, 1, dataReduction), int(float(noRawTimeSteps)/self.base_aggregation.timeStepsPerPeriod))
        
        # starting point
        noTypicalPeriods=_minPeriods
        noSegments=self.base_aggregation.timeStepsPerPeriod
        RMSE_old=self._testAggregation(noTypicalPeriods, noSegments)
        
        # start the iteration
        convergence=False
        while not convergence and noTypicalPeriods<=_maxPeriods and noSegments>0:
            # increase the number of periods until we get a reduced set of segments
            while self._segmentHistory[-1]==noSegments and noTypicalPeriods<_maxPeriods:
                noTypicalPeriods+=1
                noSegments=getNoSegmentsForDataReduction(noRawTimeSteps, noTypicalPeriods, dataReduction)
            
            # derive new typical periods
            RMSE_n=self._testAggregation(noTypicalPeriods, noSegments)
                    
            # check if the RMSE could be reduced
            if RMSE_n < RMSE_old:
                RMSE_old=RMSE_n
                convergence=False
            # in case it cannot be reduced anymore stop
            else:
                convergence=True
        
        # take the previous set, since the latest did not have a reduced error
        noTypicalPeriods=self._periodHistory[-2]
        noSegments=self._segmentHistory[-2]

        # and return the segment and typical period pair
        return noSegments, noTypicalPeriods


    def identifyParetoOptimalAggregation(self, untilTotalTimeSteps=None):
        """        
        Identifies the pareto-optimal combination of number of typical periods and number of segments along with a steepest decent approach, starting from the aggregation to a single period and a single segment up to the representation of the full time series.

        :param untilTotalTimeSteps: Number of timesteps until which the pareto-front should be determined. If None, the maximum number of timesteps is chosen.
        :type untilTotalTimeSteps: int


        :returns: **** -- Nothing. Check aggregation history for results. All typical Periods in scaled form.
        """
        if not self.base_aggregation.segmentation:
            raise ValueError("This function does only make sense in combination with 'segmentation' activated.")

        noRawTimeSteps=len(self.base_aggregation.timeSeries.index)
        
        _maxPeriods = int(float(noRawTimeSteps)/self.base_aggregation.timeStepsPerPeriod)
        _maxSegments = self.base_aggregation.timeStepsPerPeriod
        
        if untilTotalTimeSteps is None:
            untilTotalTimeSteps=noRawTimeSteps


        progressBar = tqdm.tqdm(total=untilTotalTimeSteps)

        # starting point
        noTypicalPeriods=1
        noSegments=1
        _RMSE_0=self._testAggregation(noTypicalPeriods, noSegments)

        # loop until either segments or periods have reached their maximum
        while noTypicalPeriods<_maxPeriods and noSegments<_maxSegments and noSegments*noTypicalPeriods<=untilTotalTimeSteps:
            # test for more segments
            RMSE_segments = self._testAggregation(noTypicalPeriods, noSegments+1)
            # test for more periods
            RMSE_periods = self._testAggregation(noTypicalPeriods+1, noSegments)
            # go along the better RMSE reduction
            if RMSE_periods<RMSE_segments:
                noTypicalPeriods+=1
                # and delete the search direction which was not persued
                self._deleteTestHistory(-2)
            else:
                noSegments+=1
                self._deleteTestHistory(-1)
            progressBar.update(noSegments*noTypicalPeriods-progressBar.n)
            
        # afterwards loop over periods and segments exclusively until maximum is reached
        while noTypicalPeriods<_maxPeriods and noSegments*noTypicalPeriods<=untilTotalTimeSteps:
            noTypicalPeriods+=1
            RMSE = self._testAggregation(noTypicalPeriods, noSegments)
            progressBar.update(noSegments*noTypicalPeriods-progressBar.n)

        while noSegments<_maxSegments and noSegments*noTypicalPeriods<=untilTotalTimeSteps:
            noSegments+=1
            RMSE = self._testAggregation(noTypicalPeriods, noSegments)
            progressBar.update(noSegments*noTypicalPeriods-progressBar.n)
        return 
