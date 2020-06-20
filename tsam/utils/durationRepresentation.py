# -*- coding: utf-8 -*-
"""Orders a set of representation values to fit several candidate value sets"""

import pyomo.environ as pyomo
import pyomo.opt as opt
import numpy as np
import pandas as pd


def durationRepresentation(candidates, clusterOrder, timeStepsPerPeriod, representMinMax=False, solver='glpk'):
    '''
    Represents the candidates of a given cluster group (clusterOrder)
    such that for every attribute the number of time steps is best fit.

    :param candidates: Dissimilarity matrix where each row represents a candidate
    :type candidates: np.ndarray

    :param clusterOrder: Integer array where the index refers to the candidate and the Integer entry to the group
    :type clusterOrder: np.array

    :param representMinMax: If in every cluster the minimum and the maximum of the attribute should be represented
    :type representMinMax: bool

    :param solver: Specifies the solver
    :type solver: string
    '''

    # make pd.DataFrame each row represents a candidate, and the columns are defined by two levels: the attributes and
    # the time steps inside the candidates.
    columnTuples = []
    for i in range(int(candidates.shape[1] / 24)):
        for j in range(timeStepsPerPeriod):
            columnTuples.append((i, j))
    candidates = pd.DataFrame(candidates, columns=pd.MultiIndex.from_tuples(columnTuples))

    clusterCenters = []
    for clusterNum in np.unique(clusterOrder):
        print('Representing cluster ' + str(clusterNum) + ' .')
        indice = np.where(clusterOrder == clusterNum)
        noCandidates = len(indice[0])
        clean_index = []

        clustercenter = []
        # get a clean index depending on the size
        for y in candidates.columns.levels[1]:
            for x in range(noCandidates):
                clean_index.append((x, y))
        for a in candidates.columns.levels[0]:
            # get all the values of a certain attribute and cluster
            candidateValues = candidates.loc[indice[0], a]
            # sort all values
            sortedAttr = candidateValues.stack().sort_values()
            # reindex and arange such that every sorted segment gets represented by its mean
            sortedAttr.index = pd.MultiIndex.from_tuples(clean_index)
            representationValues = sortedAttr.unstack(level=0).mean(axis=1)
            # respect max and min of the attributes
            if representMinMax:
                representationValues.loc[0] = sortedAttr.values[0]
                representationValues.loc[representationValues.index[-1]] = sortedAttr.values[-1]
            # get the order of the representation values such that euclidian distance to the candidates is minimized
            order = get_min_euclid_order(candidateValues, representationValues, solver=solver)
            # arange
            representationValues.index = order
            representationValues.sort_index(inplace=True)

            # add to cluster center
            clustercenter = np.append(clustercenter, representationValues.values)

        clusterCenters.append(clustercenter)
    return clusterCenters


def get_min_euclid_order(candidate_values, representation_values, solver='glpk'):
    '''
    Aranges the a set of representation values to fit the candidate values with the help of a MIP.

    Parameters
    ----------
    :param candidate_values: A set of candidate values. Every row is a candidate and every column a time step.
    :type candidate_values: pd.DataFrame

    :param representation_values: A set of values representing the candidates. Length should meet the number of columns
        of the candidates.
    :param representation_values: pd.Series

    :param solver: Specifies the solver. optional, default: 'glpk'
    :type solver: string
    '''
    distances = pd.DataFrame(0, columns=representation_values.index, index=representation_values.index)
    for j in representation_values.index:
        distances.loc[j, :] = candidate_values.sub(representation_values.loc[j]).apply(np.square).sum(axis=0)

    # Create model
    M = pyomo.ConcreteModel()

    # get distance matrix
    M.d = distances.values

    # Distances is a symmetrical matrix, extract its length
    length = distances.shape[0]

    # get indices
    M.i = [j for j in range(length)]
    M.j = [j for j in range(length)]

    # initialize vars
    M.z = pyomo.Var(M.i, M.j, within=pyomo.Binary)

    # get objective
    def objRule(M):
        return sum(sum(M.d[i, j] * M.z[i, j] for j in M.j) for i in M.i)

    M.obj = pyomo.Objective(rule=objRule)

    # s.t.
    # Assign every representation to a single place
    def singlePlaceRule(M, j):
        return sum(M.z[i, j] for i in M.i) == 1

    M.singlePlaceCon = pyomo.Constraint(M.j, rule=singlePlaceRule)

    # s.t.
    # Every time step can only have a single representation
    def singleRepresentationRule(M, i):
        return sum(M.z[i, j] for j in M.j) == 1

    M.singleRepresentationCon = pyomo.Constraint(M.j, rule=singleRepresentationRule)

    optprob = opt.SolverFactory(solver)

    results = optprob.solve(M, tee=False)

    r_x = np.array([[round(M.z[i, j].value) for i in range(length)]
                    for j in range(length)])
    order = r_x.argmax(axis=0)

    return order
