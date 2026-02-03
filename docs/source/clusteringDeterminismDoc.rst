.. _clustering_determinism:

Clustering Determinism
======================

Some clustering methods in tsam are non-deterministic when no random seed is set.
This page documents which methods are deterministic and which require seeding for reproducible results.

Summary Table
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 15 30

   * - Method
     - Implementation
     - Deterministic
     - Notes
   * - ``averaging``
     - Sequential division
     - Yes
     - Divides periods sequentially by order
   * - ``hierarchical``
     - ``sklearn.AgglomerativeClustering``
     - Yes
     - Ward linkage, no randomness
   * - ``contiguous``
     - ``sklearn.AgglomerativeClustering`` + connectivity
     - Yes
     - Same as hierarchical with adjacency constraint
   * - ``kmedoids``
     - Pyomo MILP (exact optimization)
     - Yes
     - Solves optimization problem exactly
   * - ``kmeans``
     - ``sklearn.KMeans``
     - **No**
     - Uses ``k-means++`` initialization with ``random_state=None``
   * - ``kmaxoids``
     - Custom implementation
     - **No**
     - Uses ``numpy.random.permutation()``


Deterministic Methods
---------------------

**averaging**
    Simply divides periods into equal sequential groups. No randomness involved.

**hierarchical**
    Uses sklearn's ``AgglomerativeClustering`` with Ward linkage. This is a deterministic
    algorithm that produces the same dendrogram given the same input.

**contiguous** (also known as ``adjacent_periods``)
    Same as hierarchical but with an adjacency constraint matrix that ensures only
    consecutive periods can be grouped together. Still deterministic.

**kmedoids** (also known as ``k_medoids``)
    Uses Mixed Integer Linear Programming (MILP) via Pyomo with the HiGHS solver.
    This is an exact optimization that always finds the same optimal solution.


Non-Deterministic Methods
-------------------------

These methods require seeding for reproducible results.

**kmeans** (also known as ``k_means``)
    Uses sklearn's ``KMeans`` with ``k-means++`` initialization. The initialization
    is random and controlled by sklearn's internal random state.

    To ensure reproducibility, set the random seed before calling::

        import numpy as np
        np.random.seed(42)

        result = tsam.aggregate(data, n_clusters=10, cluster=ClusterConfig(method="kmeans"))

**kmaxoids** (also known as ``k_maxoids``)
    Custom implementation in ``tsam/utils/k_maxoids.py`` that uses
    ``numpy.random.permutation()`` for random restarts.

    To ensure reproducibility, set the random seed before calling::

        import numpy as np
        np.random.seed(42)

        result = tsam.aggregate(data, n_clusters=10, cluster=ClusterConfig(method="kmaxoids"))


Best Practices
--------------

For reproducible results with non-deterministic methods:

1. **Set the random seed** at the start of your script::

       import numpy as np
       np.random.seed(42)

2. **Use deterministic methods** when reproducibility is critical and performance allows
   (e.g., ``hierarchical`` or ``kmedoids``)

3. **Document the seed** used in your experiments for reproducibility
