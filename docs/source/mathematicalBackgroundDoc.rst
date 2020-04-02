************************
Mathematical Description
************************

The description of tsam presented in the following is based on the review on time series aggregation methods by
`Hoffmann et al. (2020) <https://www.mdpi.com/1996-1073/13/3/641>`_.
tsam is aggregating time series by reducing the number of time steps. Generally, time series can also be aggregated by
grouping similar time series as illustrated in the upper right part of the figure below. Instead, tsam is decreasing
the amount of time series data by merging adjacent time steps based on their similarity (segmentation) or forming time
periods along the time axis and clustering those based on their similarity. This is shown in the middle right part and the
lower part of the figure below. The number of attributes to be clustered is thus not changed and accordingly, tsam is also
capable of clustering multi-dimensional time series without changing their dimensionality.

.. image:: https://www.mdpi.com/energies/energies-13-00641/article_deploy/html/images/energies-13-00641-g004.png
    :target: https://www.mdpi.com/energies/energies-13-00641/article_deploy/html/images/energies-13-00641-g004.png
    :alt: Review Figure 1
    :align: center

The process of clustering applied in tsam includes for different steps: Preprocessing, clustering, adding extreme periods and
backscaling. This is shown in the figure below.

.. image:: https://www.mdpi.com/energies/energies-13-00641/article_deploy/html/images/energies-13-00641-g009.png
    :target: ../../source/https://www.mdpi.com/energies/energies-13-00641/article_deploy/html/images/energies-13-00641-g009.png
    :alt: Review Figure 2
    :align: center

The preprocessing mainly consists of an attribute-wise normalization of all time series in order to avoid overweighting of attributes
with larger scales during the clustering process:

.. math::
   x_{a,s}=\frac{x'_{a,s}-\min{x'_a}}{\max{x'_a}-\min{x'_a}}

Then, all time steps within the chosen periods (e.g. hourly time steps within daily periods) are realigned in such a way that each
period becomes an own row-vector or hyperdimensional point whose dimensions are formed by the number of time steps within the periods
for each attribute.

Then, clustering is applied to these hyperdimensional points. Clustering generally strives to group data points in such a way that
points within a cluster are more similar to each other than data points from different clusters. An example for this is the k-means
clustering algorithm with the objective function to minimize the sum of all distances of all data points to their cluster centers as
given by:

.. math::
   \min{\sum_{k=1}^{N_k}\sum_{p\in\mathbb{C}_k}}\text{dist}(x_p,c_k)^2

With:

.. math::
   \text{dist}(x_p,c_k)=\sqrt{\sum_{a=1}^{N_a}\sum_{t=1}^{N_t}(x_{p,a,t}-c_k)^2}

And:

.. math::
   c_k=\frac{1}{\left | \mathbb{C}_k \right |}\sum_{p\in\mathbb{C}_k}x_{p,a,t}

After that and since some of the clustering methods in tsam are not preserving the average value of each time series, an optional
attribute-wise rescaling step according to the following equation can be performed:

.. math::
   c^*_{k,a,t}=c_{k,a,t}\frac{\sum_{p=1}^{N_p}\sum_{t=1}^{N_t}x_{p,a,t}}{\sum_{k=1}^{N_k}\left ( \left | \mathbb{C}_k \right |\sum_{t=1}^{N_t}c_{p,a,t}  \right )} \qquad \forall \qquad k,a,t

In an additional intermediate step the temporal resolution of the periods can also be decreased using segmentation.
In the end all time series are scaled back to their original scale:

.. math::
   c'^*_{k,a,t}=c^*_{k,a,t}\left ( \max{x'_a}-\min{x'_a} \right ) + \min{x'_a} \qquad \forall \qquad a

The output of tsam are thus clustered periods with different numbers of occurences consisting of time segments with different lenghts.