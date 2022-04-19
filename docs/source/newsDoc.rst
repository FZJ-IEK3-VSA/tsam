#################
tsam's Change Log
#################

*********************
Release version 2.0.1
*********************

tsam release (1.1.2) includes the following new functionalities:
* Changed dependency of scikit-learn to make tsam conda-forge runnable.


*********************
Release version 2.0.0
*********************

In tsam release 2.0.0 the following functionalities were included:
* A new comprehensive structure that allows for free cross-combination of clustering algorithms and cluster representations, e.g. centroids or medoids.
* A novel cluster representation method that precisely replicates the original time series value distribution in the aggregated time series based on “Hoffmann, Kotzur and Stolten (2021): The Pareto-Optimal Temporal Aggregation of Energy System Models (https://arxiv.org/abs/2111.12072)”
* Maxoids as representation algorithm which represents time series by outliers only based on “Sifa and Bauckhage (2017): Online k-Maxoids clustering”
* K-medoids contiguity: An algorithm based on “Oehrlein and Hauner (2017): A cutting-plane method for adjacency-constrained spatial aggregation” that accounts for contiguity constraints to e.g. cluster only time series in neighboring regions


*********************
Release version 1.1.2
*********************

tsam release (1.1.2) includes the following new functionalities:

* Added first version of the k-medoid contiguity algorithm.

*********************
Release version 1.1.1
*********************

tsam release (1.1.1) includes the following new functionalities:

* Significantly increased test coverage 
* Separation between clustering and representation, i.e. for clustering algorithms like Ward’s hierarchical clustering algorithm the representation by medoids or centroids can now freely be chosen.

*********************
Release version 1.1.0
*********************

tsam release (1.1.0) includes the following new functionalities:

* Segmentation - the clustering of adjacent time steps - according to Pineda et al. (2018)
* k-MILP - an extension of the MILP-based k-medoids clustering that allows automatic identification of extreme periods according to Zatti et al. (2019)
* The option to dynamically choose whether to clusters found should be represented by their centroid or medoid.
