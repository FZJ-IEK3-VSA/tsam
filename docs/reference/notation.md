# Notation and equations

Every symbol and formula tsam's pipeline uses, on one page, following the notation of
[Hoffmann et al. (2020)](https://www.mdpi.com/1996-1073/13/3/641).

This page is a lookup, not a lesson: each equation links to the notebook that derives it,
works it on real data, and explains *why* it is there. For plain-language definitions of
terms like *period* or *medoid*, see the [Glossary](glossary.md); for where tsam sits among
aggregation methods generally, see
[Methodological positioning](../explanation/background/architecture/context.md#methodological-positioning).

## Symbols

### Indices and sizes

| Symbol | Meaning |
|---|---|
| $a = 1 \dots N_a$ | **attribute** — one input column (e.g. `solar`, `load`) |
| $t = 1 \dots N_t$ | **timestep within a period** (e.g. the 24 hours of a day) |
| $s$ | timestep in the original, flat series (before periods are formed) |
| $p = 1 \dots N_p$ | **period** — one candidate row of the period matrix $D$ |
| $k = 1 \dots N_k$ | **cluster** — $N_k$ = `n_clusters` |
| $i, j$ | period indices used as *candidate center* and *assigned period* |

### Values

| Symbol | Meaning |
|---|---|
| $x'_{a,s}$ | **raw** value of attribute $a$ at step $s$, in physical units |
| $x_{a,s}$ | **normalized** value, in $[0, 1]$ |
| $x_{p,a,t}$ | normalized value of attribute $a$ at timestep $t$ of period $p$ |
| $x_p$ | period $p$ as one row-vector of $D$ — a point in $N_a \cdot N_t$ dimensions |
| $\mathbb{C}_k$ | **cluster $k$**: the set of periods assigned to group $k$ |
| $\lvert\mathbb{C}_k\rvert$ | **occurrences** of cluster $k$ — how many periods it stands for |
| $c_k$, $c_{k,a,t}$ | the **center** (representative) of cluster $k$ |
| $c^*_{k,a,t}$ | representative after **rescaling** |
| $c'^*_{k,a,t}$ | representative after **denormalization** — back in physical units |
| $d_{i,j}$ | distance between periods $i$ and $j$ |
| $z_{i,j}$ | k-medoids binary: 1 if period $j$ is assigned to center $i$ |
| $J$ | total within-cluster distance — the clustering objective |

## Preprocessing

Attribute-wise min–max normalization, so attributes with larger physical ranges do not
dominate the distance:

$$
x_{a,s} = \frac{x'_{a,s} - \min x'_a}{\max x'_a - \min x'_a}
$$

The normalized series is then **unstacked**: each period becomes one row-vector whose
dimensions are $N_a \cdot N_t$. The number of attributes never changes — tsam reduces
timesteps, not columns.

*Derived in:* [Preprocessing](../explanation/how-aggregation-works/01_preprocessing.ipynb)

## Clustering

Distance between a period and a center, over every $(a, t)$ coordinate:

$$
\text{dist}(x_p, c_k) = \sqrt{\sum_{a=1}^{N_a} \sum_{t=1}^{N_t} (x_{p,a,t} - c_{k,a,t})^2}
$$

The objective both k-means and k-medoids minimize — every period measured against the center
of the cluster it lands in:

$$
J = \sum_{k=1}^{N_k} \sum_{p \in \mathbb{C}_k} \text{dist}(x_p, c_k)^2
$$

k-means (Lloyd's algorithm) reaches a *local* optimum by repeatedly moving each center to the
mean of its members:

$$
c_k = \frac{1}{\lvert\mathbb{C}_k\rvert} \sum_{p \in \mathbb{C}_k} x_p
$$

k-medoids minimizes the *same* $J$ exactly, restricting centers to real periods, which makes
it a $p$-median MILP over the distance matrix alone:

$$
\min_{z}\ \sum_{i}\sum_{j} d_{i,j}\, z_{i,j} \;=\; J
$$

*Derived in:* [Partitional clustering](../explanation/how-aggregation-works/02_clustering/01_partitional_clustering.ipynb)
— including the MILP's three constraints.
Other methods carry their own objectives: Ward's merge cost $\Delta(A,B)$ in
[Agglomerative clustering](../explanation/how-aggregation-works/02_clustering/02_agglomerative_clustering.ipynb)
and the spread objective $E(M)$ in
[Extremal-prototype selection](../explanation/how-aggregation-works/02_clustering/03_extremal_prototype_selection.ipynb).

## Rescaling

Some representations do not preserve each attribute's mean, so the occurrence-weighted totals
drift from the original. An optional multiplicative factor per attribute restores them
(`preserve_column_means`, on by default):

$$
c^*_{k,a,t} = c_{k,a,t} \cdot
\frac{\sum_{p=1}^{N_p}\sum_{t=1}^{N_t} x_{p,a,t}}
{\sum_{k=1}^{N_k} \left(\lvert\mathbb{C}_k\rvert \sum_{t=1}^{N_t} c_{k,a,t}\right)}
\qquad \forall \qquad k, a, t
$$

The numerator is the original total; the denominator is what the occurrence-weighted
representatives sum to before correction. If they already match, the factor is 1.

*Derived in:* [Rescaling and denormalization](../explanation/how-aggregation-works/05_rescaling.ipynb)

## Denormalization

The exact inverse of the min–max step, returning representatives to physical units:

$$
c'^*_{k,a,t} = c^*_{k,a,t} \left( \max x'_a - \min x'_a \right) + \min x'_a
\qquad \forall \qquad a
$$

*Derived in:* [Rescaling and denormalization](../explanation/how-aggregation-works/05_rescaling.ipynb)

## Output

The result is a set of typical periods, each with an occurrence count
$\lvert\mathbb{C}_k\rvert$, optionally made of segments of differing length. Segmentation
applies the same Ward criterion *within* periods rather than across them — see
[Segmentation](../explanation/how-aggregation-works/06_segmentation.ipynb).
