# Dimensionality Reduction

:::{admonition} How to work through these notes (accessible version)
:class: important
On the standard site this page is a Jupyter notebook. Here, the code is written
to be run as a **Python script** in VS Code — see [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Put the code into a file such as `dimreduction.py`, adding the sections below one
at a time, and run it with `python dimreduction.py`. Read printed output with the
terminal's Accessible View.

**Plots.** Each figure has a **"What the plot shows"** description. Where a plot
carries an argument, the code also **prints the numbers** behind it, so you never
have to rely on the picture. To explore a plot yourself with sound and a
text/braille readout, use **MAIDR** — see [Accessible plots with
MAIDR](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/sci_python/trying_maidr.html).
:::

So far in this course we have worked on **supervised** machine learning: we have a
labeled data set, and we try to learn a mapping between our features and our known
labels. **Unsupervised** machine learning is a different paradigm. Each sample
still has features associated with it, but we have **no labels at all**. What we
are looking for instead are groupings, patterns, or simplifications that we can
find in the data using only the features themselves.

This page covers the first half of the unsupervised lecture:

- why we need to reduce the dimensionality of large data sets (the **curse of
  dimensionality**)
- the main families of **dimensionality reduction** methods
- **Principal Component Analysis (PCA)** — how it works, and how it is applied
- when PCA **doesn't** work well

Clustering — the other major branch of classical unsupervised learning — is covered
in the [Clustering](clustering.md) notes. Later in the course we will also revisit
unsupervised learning from a deep learning perspective.

Once you have worked through these notes, the [Tutorial on Dimensionality
Reduction](PCA.md) applies PCA hands-on to image data and to tropical Pacific sea
surface temperatures.

## The curse of dimensionality

The **curse of dimensionality** refers to the challenges that arise as the
dimensionality of a data set increases: increased sparsity, computational
complexity, and decreased efficiency in machine learning and data analysis tasks.

The intuition is a counting argument. Suppose each feature can take one of $n$
distinct values:

- with **1** feature, there are $n$ possible states a sample could occupy
- with **2** features, there are $n$ squared
- with **3** features, there are $n$ cubed
- with $m$ features, there are $n$ to the power $m$

Every dimension you add multiplies the number of places your data could live. But
the number of samples you actually have stays the same. So as dimensionality grows,
your data set becomes **very sparse**: training instances end up far away from one
another, with a lot of empty space between them. Two samples that each have only
two features are likely to be reasonably close to each other; two samples in very
high dimensions are likely to be very far apart.

This matters because most machine learning relies on some notion of samples being
"near" or "similar to" one another. When every point is far from every other point,
that notion breaks down, and the risk of **overfitting** increases.

We can measure this directly. For a fixed number of random points, we track how the
distance to the nearest neighbor compares with the distance to a typical point, as
we add dimensions:

```python
import numpy as np

rng = np.random.default_rng(0)

n_samples = 500
print(" dims | mean distance | nearest distance | ratio nearest/mean")
for d in [1, 2, 3, 5, 10, 25, 50, 100]:
    P = rng.random((n_samples, d))
    diff = P[:100, None, :] - P[None, :100, :]
    D = np.sqrt((diff ** 2).sum(-1))
    D = D[np.triu_indices(100, k=1)]
    print(f" {d:4d} | {D.mean():13.3f} | {D.min():16.3f} | {D.min()/D.mean():18.3f}")
```

:::{admonition} What the plot shows
:class: note
The notebook draws two line plots side by side. The left shows the mean distance
between points and the distance to the nearest neighbor both **rising steadily**
as dimensions are added — everything gets further from everything else.

The right-hand panel is the important one: the **ratio** of nearest distance to
mean distance. It starts near 0 in 1D and climbs steadily toward 1.

The printed table above gives you the same quantities directly, and is the more
precise version of that panel. Read down the last column: it goes from about 0.00
in 1D — where your nearest neighbor is essentially on top of you while a typical
point is far away, so "who is nearest?" is a very meaningful question — to about
0.58 at 25 dimensions and **0.78 at 100**, still climbing. Once the nearest point is
three-quarters as far as an average point, everything is starting to look
equidistant, and "nearest neighbor" stops meaning much.
:::

### Data usually lives on a lower-dimensional manifold

Here is the saving grace. Real data sets may have very many features, but the states
those variables *actually* take are far fewer than every possible combination. In
practice, the **degrees of freedom are limited**.

Climate is a good example. A climate model has many, many variables and parameters —
but only certain parameter values are physically reasonable, and only certain climate
states would ever be expected to occur in the real world. So despite the enormous
nominal dimensionality of the data set, the number of real states it can occupy is
far smaller.

Data that nominally lives in a high-dimensional space but really occupies a
lower-dimensional surface within it is said to live on a **manifold**. Unsupervised
machine learning helps us discover, and then exploit, these lower-dimensional
relationships.

## Dimensionality reduction

**Dimensionality reduction** refers to techniques that reduce the number of features
in a data set while retaining the essential information. It is used for:

- **Data visualization** — we can only really look at 2 or 3 dimensions at a time
- **Data analysis** — finding interpretable patterns
- **Feature engineering** — reducing many variables to a smaller, more informative
  set before feeding them to a model
- **Improved model performance** — focusing only on the features that matter most,
  and reducing computational expense

The main families of methods are:

- **Projection methods** linearly transform data to lower dimensions. **Principal
  Component Analysis (PCA)** is by far the most common, and is the focus of these
  notes.
- **Manifold learning methods** capture the underlying data structure for
  *non-linear* relationships. Examples are **t-distributed Stochastic Neighbor
  Embedding (t-SNE)** and **Uniform Manifold Approximation and Projection (UMAP)**.
  These draw on ideas from topology and Riemannian manifolds that are beyond the
  scope of this course. They are used mostly for **visualizing** high-dimensional
  data, and less for quantitative analysis.
- **Deep learning methods** non-linearly transform data to lower dimensions.
  **Auto-encoders** and **variational auto-encoders** can be thought of as a
  non-linear generalization of PCA. We will return to these later in the course.

## Why do we need dimensionality reduction in climate and environmental science?

In this field we very often have **high-dimensional spatio-temporal data sets**. A
climate model, a weather model, or a hydrology model gives you data at every single
point in space and time, for a number of different variables. These are extremely
high-dimensional data sets.

But what we are usually interested in is *not* the exact value of temperature or
wind at every single point in space and time. It is the **patterns** that emerge and
that tell us something about how the real world works. Dimensionality reduction
helps us to:

- **Make sense of high-dimensional data** — for example, Lee et al. (2023, *Journal
  of Climate*) applied unsupervised dimensionality reduction to weather data over
  North America, looking at the 500 mb pressure level to find recurring patterns in
  jet stream behavior.
- **Discover inherent patterns and correlations** — Nakamura et al. (2009, *Journal
  of Climate*) looked at hurricanes and tropical cyclones in the Atlantic and found
  subsets of repeated cyclone track patterns that emerged naturally from historic
  storm data. Uncovering those patterns is a route to learning about the physical
  drivers behind them.
- **Feature engineering** and **reducing computational expense**.

## Principal Component Analysis (PCA)

The basic idea behind PCA is that we find a transformation of our features into a
new coordinate system, where the data is projected along the **directions of
greatest variance**. The principal components are ordered by how much variance they
explain: the first principal component is the direction of greatest variance, the
second is the direction of the next greatest variance (constrained to be orthogonal
to the first), and so on.

When PCA is applied to spatio-temporal data, as it often is in climate and
meteorology, it is also known as **Empirical Orthogonal Function (EOF) analysis**.
The two are effectively equivalent — the terminology just differs by field.

```python
from sklearn.decomposition import PCA

# A simple 2D data set with a clear direction of greatest variance.
rng = np.random.default_rng(4)
n = 200
base = rng.normal(0, 1, n)
X = np.column_stack([base * 2.0 + rng.normal(0, 0.35, n),
                     base * 0.9 + rng.normal(0, 0.35, n)])

pca = PCA(n_components=2).fit(X)
print("variance explained by each component:", pca.explained_variance_ratio_.round(3))
```

This prints `[0.975 0.025]`.

:::{admonition} What the plot shows
:class: note
The notebook draws a scatter of 200 red points forming a long, narrow, tilted cloud
running diagonally up and to the right. Two double-headed black arrows are drawn
through the center of the cloud: a long one labeled $c_1$ pointing **along** the
cloud's long axis, and a much shorter one labeled $c_2$ pointing **across** it, at
right angles. The arrows are scaled by how much variance each explains, which is why
$c_1$ is so much longer.
:::

$x_1$ and $x_2$ are the original variables. The direction of greatest variance is
along $c_1$ — the **first principal component**. The next direction is $c_2$, the
**second principal component**. If we transform our coordinates so the data lies
along $c_1$ and $c_2$, we get a much simpler description of the data set.

Notice how lopsided the explained variance is: **97.5%** along $c_1$ and only
**2.5%** along $c_2$. Almost all the variability is along a single direction, which
means we could describe this data set with one number per sample and lose very
little.

### How to proceed, step by step

Our data is in a data matrix $\mathbf{X}$ of shape $n_{samples}$ by $n_{features}$.
Done by hand, PCA works like this:

1. **Mean-center the data**, so it is centered on the origin:
   $\mathbf{X}_c = \mathbf{X} - \mathbf{X}_{mean}$

2. **Apply Singular Value Decomposition (SVD)** to the centered matrix:
   $\mathbf{X}_c = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{T}$.
   The matrix $\mathbf{V}$ contains all of the principal components — the rows of
   $\mathbf{V}^T$ are the principal directions, ordered from most to least variance
   explained.

3. **Choose the number of dimensions** $d$ to keep, to retain as much variance as you
   need.

4. **Project** the data onto the first $d$ principal directions. With numpy's
   `U, s, Vt = np.linalg.svd(X_centered)`, that is `X_centered @ Vt[:d].T`, which gives
   an $n_{samples}$ by $d$ matrix — one row per sample, one column per component kept.
   (Watch the transposes: `Vt[:d]` picks the first $d$ principal directions as *rows*,
   so it needs transposing back before the multiply. Equivalently, you are multiplying
   by the first $d$ *columns* of $\mathbf{V}$.)

In practice you would use `scikit-learn`'s `PCA`, which has already implemented this
factorization, and which follows the usual pattern: initialize with the number of
components, then `fit_transform` on your feature matrix.

```python
# Doing it by hand with SVD...
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
print("principal directions from SVD:")
print(Vt[:2])

# ...gives the same answer as scikit-learn.
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
print("principal directions from scikit-learn:")
print(pca.components_)
```

The two agree, up to an overall sign — a principal direction and its negative
describe the same axis, so the sign PCA hands back is arbitrary.

### How many dimensions should we keep?

Once we have the principal components, we project the data onto them. If we keep only
the **leading** components and their amplitudes, we can use that smaller subset of
variables to effectively **compress** the data while still keeping the majority of
the variance.

To decide how many to keep, we look at the **cumulative explained variance** against
the number of dimensions retained, and pick the point where we have as much of the
original variance as we need (95% is a common choice).

```python
from sklearn.datasets import load_digits

# Handwritten digit images, 8x8 pixels flattened into 64 features per sample.
digits = load_digits()
Xd = digits.data
print("original shape:", Xd.shape)

pca_full = PCA().fit(Xd)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

print(f"{d} components retain {cumsum[d-1]*100:.1f}% of the variance "
      f"(down from {Xd.shape[1]} features)")

# Read the curve as numbers rather than as a picture:
for k in [1, 5, 10, 20, 29, 40, 64]:
    print(f"  {k:3d} components -> {cumsum[k-1]*100:5.1f}% of variance retained")
```

:::{admonition} What the plot shows
:class: note
A single line rising from near 0 and flattening out: cumulative explained variance
(vertical) against dimensions kept (horizontal, 0 to 64). It climbs very steeply over
the first ~10 components, bends sharply — an "elbow" is annotated around 10 — and then
creeps up slowly. Dotted guide lines mark where the curve crosses 0.95, at **29** of
the 64 dimensions.

The message: most of the information is concentrated in a small number of components,
and past the elbow you pay many extra dimensions for very little extra variance. The
printed table above gives the same curve as numbers.
:::

The same argument scales up. In the [tutorial](PCA.md) you will apply PCA to the
MNIST data set, where each image has **784** features. There, retaining about 95% of
the original variance takes only slightly more than **150** dimensions rather than
the original 784.

### PCA can transform variables to separate different classes

If what we care about is telling classes apart, PCA can find a transformation into a
space where the classes are more easily separable. The classic demonstration is the
**Iris** data set: three species of iris, each described by four features.

```python
from sklearn.datasets import load_iris

iris = load_iris()
Xi, yi = iris.data, iris.target

Xi_pca = PCA(n_components=2).fit_transform(Xi)

# How much do the classes overlap, before and after PCA?
# Compare each class's spread along an axis with the gaps between class centers.
for name, data, axis_label in [("original feature 1", Xi[:, 0], iris.feature_names[0]),
                               ("first principal component", Xi_pca[:, 0], "PC1")]:
    print(f"\n{name} ({axis_label}) — class mean +/- std:")
    for k, cname in enumerate(iris.target_names):
        vals = data[yi == k]
        print(f"   {cname:12s}: {vals.mean():6.2f} +/- {vals.std():.2f}")
```

:::{admonition} What the plot shows
:class: note
Two scatter plots side by side, each with the three iris species in different colors.

**Left — two of the four original features** (sepal length vs sepal width): one
species sits clearly apart, but the other two form a single merged blob. You could
not draw a line between them.

**Right — the first two principal components**: the three species now form three
distinguishable groups strung out along the horizontal axis. The two that were merged
on the left are now largely separated, with only slight overlap at their boundary —
you could plausibly draw decision boundaries between all three.

The printed class means and standard deviations above show this numerically. Along
the first principal component the class means are far apart relative to the spread
within each class; along an original feature they are not.
:::

After PCA we have new variables that are **linear combinations of the four original
features**, and the classes are considerably easier to separate.

This is why dimensionality reduction is often used **in combination with** the
clustering methods in the [next set of notes](clustering.md): together they let us do
**unsupervised classification**, where we don't start with labeled data, but instead
find a transformation into a space where the classes separate on their own.

## When does PCA not work well?

PCA has trouble with:

- **Non-linear relationships** — high-dimensional data oriented on a complex manifold
- **Outliers**

Be careful with the first one: *high dimensionality on its own is not the problem*. We
just used PCA to take 64-dimensional digits down to 29, and in the tutorial it takes
MNIST from 784 down to 154. What defeats PCA is high-dimensional data whose structure is
**curved**. PCA can only rotate and project, so if the information lives on a bent
surface, no flat projection will recover it.

**Outliers** are a separate weakness, and follow directly from what PCA optimizes. It
picks the directions of greatest *variance*, and a far-away point contributes a very
large squared distance — so a handful of outliers can drag a principal axis toward
themselves and away from the structure of the bulk of the data.

The standard illustration of the non-linear case is the **Swiss roll**. It is a 3D
data set, but the data is really rolled up on what would be two dimensions if we could
find a transformation that flattened it out. PCA — being a *linear* transformation —
cannot do that. It can only project, so it squashes the roll flat and smears together
points that are far apart along the roll.

```python
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
from scipy.stats import spearmanr

X_sr, color = make_swiss_roll(n_samples=1500, noise=0.05, random_state=42)

X_pca = PCA(n_components=2).fit_transform(X_sr)
X_lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10,
                               random_state=42).fit_transform(X_sr)

# `color` is each point's true position ALONG the roll. A method that successfully
# unrolls the data should produce one coordinate that tracks it monotonically.
for name, emb in [("PCA", X_pca), ("LLE (manifold learning)", X_lle)]:
    best = max(abs(spearmanr(emb[:, i], color).correlation) for i in range(2))
    print(f"{name:24s}: best |rank correlation| with true position along roll = {best:.3f}")
```

:::{admonition} What the plot shows
:class: note
Three panels. The color of every point encodes its true position **along** the roll,
running smoothly through a rainbow from one end to the other.

**Left — the Swiss roll in 3D**: points spiral around in a curled sheet, like a rolled
carpet seen end-on, with the color winding around the spiral.

**Middle — PCA**: the colors **fold back over each other**. Points from opposite ends
of the roll land on top of one another, because a flat projection of a curled sheet
necessarily overlaps. The rainbow is jumbled rather than ordered.

**Right — manifold learning (LLE)**: the color varies **smoothly from one side to the
other**, in order. The 2D structure has been recovered — the roll has been unrolled.

The printed rank correlations above capture this without the picture: for LLE, one
embedding coordinate tracks the true position along the roll closely (correlation near
1), while for PCA no coordinate does nearly as well, because the fold-over destroys the
ordering.
:::

None of this makes PCA a bad method. It remains extremely useful precisely **because**
it is a linear transformation: it is fast, and it is fairly easy to interpret, which is
why it is so widely used. It is just worth knowing what it can and cannot see.

There are many other dimensionality reduction methods available; it is worth reading the
[scikit-learn documentation](https://scikit-learn.org/stable/modules/manifold.html) and
exploring, depending on the data set you are working with.

## Where next

The [Tutorial on Dimensionality Reduction](PCA.md) puts PCA to work on real data —
compressing handwritten digit images, and then applying EOF analysis to tropical Pacific
sea surface temperatures to recover the El Niño/Southern Oscillation pattern without ever
labeling it.

The [Clustering](clustering.md) notes cover the other half of classical unsupervised
learning.