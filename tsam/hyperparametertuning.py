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
        
        _maxPeriods = int(float(noRawTimeSteps)/self.base_aggregation.timeStepsPerPeriod)
        _maxSegments = self.base_aggregation.timeStepsPerPeriod

        # save RMSE
        RMSE_history = []

        # correct 0 index of python
        possibleSegments = np.arange(_maxSegments)+1
        possiblePeriods = np.arange(_maxPeriods)+1
        
        # number of time steps of all combinations of segments and periods
        combinedTimeSteps = np.outer(possibleSegments, possiblePeriods)
        # reduce to valid combinations for targeted data reduction
        reductionValidCombinations = combinedTimeSteps <= noRawTimeSteps * dataReduction

        # number of time steps for all feasible combinations
        reductionValidTimsteps = combinedTimeSteps * reductionValidCombinations
        
        # identify max segments and max period combination
        optimalPeriods = np.zeros_like(reductionValidTimsteps)
        optimalPeriods[np.arange(reductionValidTimsteps.shape[0]), reductionValidTimsteps.argmax(axis=1)] = 1
        optimalSegments = np.zeros_like(reductionValidTimsteps)
        optimalSegments[reductionValidTimsteps.argmax(axis=0), np.arange(reductionValidTimsteps.shape[1])] = 1
        
        optimalIndexCombo = np.nonzero(optimalPeriods*optimalSegments)        

        
        for segmentIx, periodIx in tqdm.tqdm(zip(optimalIndexCombo[0],optimalIndexCombo[1])):

            # derive new typical periods and derive rmse
            RMSE_history.append(self._testAggregation(possiblePeriods[periodIx], possibleSegments[segmentIx]))
                                                    
        # take the negative backwards index with the minimal RMSE 
        min_index = - list(reversed(RMSE_history)).index(min(RMSE_history)) - 1
        RMSE_min = RMSE_history[min_index]
        
                
        noTypicalPeriods=self._periodHistory[min_index]
        noSegments=self._segmentHistory[min_index]

        # and return the segment and typical period pair
        return noSegments, noTypicalPeriods, RMSE_min


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
        while (noTypicalPeriods<_maxPeriods and noSegments<_maxSegments 
            and (noSegments+1)*noTypicalPeriods<=untilTotalTimeSteps
            and noSegments*(noTypicalPeriods+1)<=untilTotalTimeSteps):
            # test for more segments
            RMSE_segments = self._testAggregation(noTypicalPeriods, noSegments+1)
            # test for more periods
            RMSE_periods = self._testAggregation(noTypicalPeriods+1, noSegments)
            
            # RMSE old
            RMSE_old = self._RMSEHistory[-3]
            
            # segment gradient (RMSE improvement per increased time step number)
            # for segments: for each period on segment added
            RMSE_segment_gradient = (RMSE_old - RMSE_segments) / noTypicalPeriods
            # for periods: one period with no of segments
            RMSE_periods_gradient = (RMSE_old - RMSE_periods) / noSegments

            # go along the steeper gradient
            if RMSE_periods_gradient>RMSE_segment_gradient:
                noTypicalPeriods+=1
                # and delete the search direction which was not persued
                self._deleteTestHistory(-2)
            else:
                noSegments+=1
                self._deleteTestHistory(-1)
            progressBar.update(noSegments*noTypicalPeriods-progressBar.n)
            
        # afterwards loop over periods and segments exclusively until maximum is reached
        while noTypicalPeriods<_maxPeriods and noSegments*(noTypicalPeriods+1)<=untilTotalTimeSteps:
            noTypicalPeriods+=1
            RMSE = self._testAggregation(noTypicalPeriods, noSegments)
            progressBar.update(noSegments*noTypicalPeriods-progressBar.n)

        while noSegments<_maxSegments and (noSegments+1)*noTypicalPeriods<=untilTotalTimeSteps:
            noSegments+=1
            RMSE = self._testAggregation(noTypicalPeriods, noSegments)
            progressBar.update(noSegments*noTypicalPeriods-progressBar.n)
        return 
