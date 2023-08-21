.. _structure_of_tsam:

#################
Structure of tsam
#################

Tsam consists of a main class containing the basic functionailities to aggregate time series to typical periods based
on averaging, k-means clustering of hierarchical clustering. The utils-folder contains additional clustering algorithms
including a mixed-interger linear programming based formulation of the k-medoids algorithm and constrained agglomerative
clustering of adjacent time steps called segmentation.

**Submodule with the basic functionalities**

.. toctree::
   :maxdepth: 1

   timeseriesaggregationDoc
   hypertunedaggregationDoc
   periodAggregationDoc
   representationsDoc

**Sub functions such as k-medoids and segmentation**

.. toctree::
   :maxdepth: 1

   exactKmedoidsDoc
   kmaxoidsDoc
   durationRepresentationDoc
   segmentationDoc