# Tutorial on Dimensionality Reduction

:::{admonition} How to work through this tutorial (accessible version)
:class: important
On the standard site this is a Jupyter notebook. Here, work through it as a
**Python script** in VS Code — see [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Put the code into a file such as `pca.py`, adding the pieces below one section at
a time, and run it in the terminal with `python pca.py` (or run the sections as
`# %%` cells if you set that up). Read printed output with the terminal's
Accessible View.

**Plots.** Each `plt.show()` below has a **"What the plot shows"** description so
you know what a sighted reader would see. To explore a plot yourself with sound
and a text/braille readout, render it through **MAIDR** instead of `plt.show()` —
see [Accessible plots with
MAIDR](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/sci_python/trying_maidr.html).
The maps of spatial patterns in the ENSO section are the hardest plots here to
read by ear, so for those we also **print the numbers** that carry the argument.
:::

In this tutorial, we will explore dimensionality reduction using a very common
approach, Principal Component Analysis (PCA), and how it can be applied to
spatio-temporal data to discover patterns in climate data.

The concepts behind PCA are introduced in the [Dimensionality
Reduction](dimensionalityreduction.md) lecture notes; this page is the hands-on
companion to those notes.

## Principal Component Analysis

PCA is a linear dimensionality reduction technique. It is used to linearly
transform data to a new coordinate system, based on the principal components of
the data. The principal components capture the directions of greatest variance in
the data, in decreasing order: the first principal component explains the greatest
variance in the data, the second explains the next most, and so on. Because the
components are ordered this way, keeping only the first few of them keeps most of
the information while throwing away most of the columns.

We'll start with a small synthetic 3D data set, so that "reducing the
dimensionality" is something we can reason about concretely: the data is a noisy,
tilted oval that lives in three dimensions but is essentially flat.

```python
import numpy as np
from scipy.spatial.transform import Rotation

m = 60
X = np.zeros((m, 3))  # initialize 3D dataset
np.random.seed(42)
angles = (np.random.rand(m) ** 3 + 0.5) * 2 * np.pi  # uneven distribution
X[:, 0], X[:, 1] = np.cos(angles), np.sin(angles) * 0.5  # oval
X += 0.28 * np.random.randn(m, 3)  # add more noise
X = Rotation.from_rotvec([np.pi / 29, -np.pi / 20, np.pi / 4]).apply(X)
X += [0.2, 0, 0.2]  # shift a bit

print(X.shape)
```

This prints `(60, 3)`: 60 samples, each with 3 features.

If we take the first 2 principal components of this data set, we are effectively
learning a projection from 3D down to 2D.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)          # dataset reduced to 2D
X3D_inv = pca.inverse_transform(X2D)  # 3D position of the projected samples

# How far is each original point from the plane it was projected onto?
offsets = np.linalg.norm(X - X3D_inv, axis=1)
print("distance from each point to the projection plane:")
print("  mean:", offsets.mean().round(3), " max:", offsets.max().round(3))
print("  spread of the data itself (std per axis):", X.std(axis=0).round(3))
```

:::{admonition} What the plot shows
:class: note
The notebook draws a 3D scatter plot: 60 red points forming a noisy, tilted oval
ring, with a translucent blue plane slicing through them. Each red point is joined
by a short dashed line to a blue point lying on the plane — the blue points are
where PCA "flattens" each sample onto the 2D plane. The visual message is that the
dashed lines are **short**: the cloud is nearly flat already, so the plane passes
close to every point and squashing 3D down to 2D loses very little.

The numbers printed above make the same point without the picture: the mean
distance from a point to the plane is small compared to the overall spread of the
data along its axes.
:::

Our data is in a data matrix $\mathbf{X}$, which has the dimensions $n_{samples}$
by $n_{features}$. To perform PCA analysis by hand, you would follow these steps:

1. **Mean-center the data.** Subtract the mean of each feature from the data set.
   This centers the data around the origin:

   $$\mathbf{X}_{c} = \mathbf{X} - \mathbf{X}_{mean}$$

2. **Apply Singular Value Decomposition (SVD)** to the centered matrix. SVD
   factors the matrix as

   $$\mathbf{X}_{c} = \mathbf{U}\,\mathbf{\Sigma}\,\mathbf{V}^{T}$$

   where the rows of $\mathbf{V}^{T}$ are the principal directions, ordered from
   most to least variance explained.

```python
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt[0]
c2 = Vt[1]

print("first principal direction: ", c1)
print("second principal direction:", c2)
```

Each of these is a 3-element unit vector — a direction in the original 3D space.
The first is the direction along which the data varies most.

We can also use `scikit-learn` to do this instead, where it is very easy to
implement PCA. In this case, it is not necessary to subtract the mean, since this
is done automatically.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)

print("principal directions found by scikit-learn:")
print(pca.components_)
```

These should match `c1` and `c2` from the SVD above, up to an overall sign (a
principal direction and its negative describe the same axis).

The **explained variance ratio** tells us how much variance in our original 3D
data each component explains.

```python
print("explained variance ratio:", pca.explained_variance_ratio_.round(4))
print("variance lost by dropping to 2D:", round(1 - pca.explained_variance_ratio_.sum(), 4))
```

The first dimension explains about 76% of the variance, while the second explains
about 15%. By projecting from 3D down to 2D we have lost only about 9% of the
variance in our original data set.

## How many dimensions should we keep?

This next section uses the data set MNIST to explore how we can figure out how
many dimensions we should keep in our data set. MNIST is a data set of small
images of (labeled) handwritten digits (0 through 9), and is used to benchmark
many machine learning models.

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False, parser="auto")
X_train, y_train = mnist.data[:60_000], mnist.target[:60_000]
X_test, y_test = mnist.data[60_000:], mnist.target[60_000:]

print(X_train.shape)
```

This prints `(60000, 784)`. The data set contains 28x28 pixel images, but here the
images are flattened into a single vector of 28*28 = 784 points.

The notebook visualizes one example image with `plt.imshow(X_train[20].reshape(28, 28))`.

:::{admonition} What the plot shows
:class: note
A single 28x28 grayscale image of a handwritten digit, displayed as a small square
picture. It is one of the 60,000 training digits; its label is in `y_train[20]`.
You can check what digit it is by printing `y_train[20]` rather than looking at
the image.
:::

```python
print("this example is the digit:", y_train[20])
```

For the sake of this example, we are using MNIST as a high dimensional data set
that we might want to apply dimensionality reduction to. We'll start by applying
PCA to the training data set.

```python
pca = PCA()
pca.fit(X_train)
print("number of components kept by default:", pca.n_components_)
```

If we don't specify how many components to keep, PCA will return a number that is
the same size as our original number of features (784).

We can determine how many principal components we need in order to retain 95% of
the original variance in the MNIST dataset:

```python
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print("components needed to retain 95% of the variance:", d)
```

This indicates that we need to keep **154** principal components — down from 784.

We can tell PCA to choose that number automatically, and then transform the data
into this new, lower dimensional space:

```python
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)

print("components chosen:", pca.n_components_)
print("variance retained:", pca.explained_variance_ratio_.sum().round(4))
```

The notebook then plots the cumulative sum of the explained variance against the
number of components kept.

:::{admonition} What the plot shows
:class: note
A line rising steeply from 0 and then flattening out: explained variance
(vertical, 0 to 1) against number of dimensions kept (horizontal, 0 to 400). The
curve climbs very fast over the first ~50 components, then bends sharply — an
"elbow" is annotated around 60–70 dimensions — and afterwards creeps up slowly.
Dotted guide lines mark the point where the curve crosses 0.95, at 154 dimensions.

The message: most of the information is concentrated in a small number of
components, and past the elbow you pay many extra dimensions for very little extra
variance.

To read the curve as numbers instead, print a few checkpoints:
:::

```python
for k in [10, 25, 50, 100, 154, 200, 300]:
    print(f"{k:4d} components -> {cumsum[k-1]*100:.1f}% of variance retained")
```

PCA can be used to **compress** data, because we can keep the majority of the
information in our original data set but with many fewer variables.

```python
print("reduced shape: ", X_reduced.shape)   # 154 features

X_recovered = pca.inverse_transform(X_reduced)
print("recovered shape:", X_recovered.shape)  # back to 784

# How much did we actually lose, per pixel?
err = np.abs(X_train - X_recovered)
print("mean absolute reconstruction error per pixel (0-255 scale):", err.mean().round(2))
```

The notebook shows a 5x5 grid of original digits beside a 5x5 grid of the same
digits reconstructed from only 154 features.

:::{admonition} What the plot shows
:class: note
Two panels side by side, titled "Original" and "Compressed", each a 5x5 grid of
handwritten digits in black on white. The compressed digits are clearly still the
same, readable digits, just very slightly softer and blurrier at the edges — the
fine speckle of the originals is smoothed away. The point is that discarding 630
of the 784 dimensions costs almost nothing you would notice.

The printed mean absolute reconstruction error above quantifies this: it is small
relative to the 0–255 pixel range.
:::

This type of compression is one way that we can do feature engineering for machine
learning. If we have an original high dimensional data set, we might first apply
PCA (or some other dimensionality reduction method) to our data, and then use the
first few principal components as input to our machine learning algorithm. This
approach is useful because it ensures that we both focus on the most significant
sources of variance in the data set and reduce the computational expense of
training the model.

## Using PCA to discover the El Niño/Southern Oscillation (ENSO) pattern

As an example of how PCA can be applied to climate data, we will look at an
analysis of the tropical Pacific sea surface temperature (SST) to reveal the
signature of the El Niño/Southern Oscillation (ENSO) phenomenon. This follows the
analysis described
[here](https://kls2177.github.io/Climate-and-Geophysical-Data-Analysis/chapters/Week7/Intro_to_PCA.html).
ENSO is an important pattern of climate variability in the tropical Pacific, with
worldwide consequences for weather and ecosystems.

When PCA is applied to spatio-temporal data like this, it is also known as
**Empirical Orthogonal Function (EOF) analysis**. The principal directions
(`pca.components_`) become spatial maps — the EOFs — and the transformed data
(the PCs) become time series describing how strongly each spatial pattern is
expressed at each time step.

```python
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
```

We will use a data set from NOAA of monthly SSTs from 1854 until 2025, which can
be accessed by running this code:

```python
sst = xr.open_dataset("http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc")
print(sst)
```

Printing the dataset gives a text summary of its dimensions, coordinates and
variables — this is already screen-reader friendly, and is the best way to
understand the shape of the data.

The notebook plots one month of SST with `sst['sst'][0].plot()`.

:::{admonition} What the plot shows
:class: note
A world map of sea surface temperature for the first month in the record
(January 1854), drawn as a colored grid: warm reds and yellows in a broad band
across the tropics, cooling through greens to blues toward both poles, with the
continents blank (missing data). Nothing surprising — it is simply confirming the
data loaded and looks like Earth.
:::

### The Niño3.4 index

We'll first calculate the **Niño3.4 index**, which is a commonly used metric of
ENSO variability. To do this we will:

1. Compute the area-averaged total SST from (5N–5S, 190E–240E).
2. Compute the monthly climatology from 1950–1979 for that region, and subtract it
   from the area-averaged SST time series to obtain anomalies.
3. Smooth the anomalies with a 5-month running mean.
4. Standardize by the standard deviation over the climatological period.

**Step 1** — compute the area-averaged total SST from (5N–5S, 190E–240E):

```python
sst_n34 = sst['sst'].sel(lat=slice(5, -5), lon=slice(190, 240)).mean(dim=["lat", "lon"])
print(sst_n34)
```

:::{admonition} What the plot shows
:class: note
`sst_n34.plot()` draws a dense, very spiky time series from 1854 to 2025 of
absolute SST in degrees C, oscillating rapidly between roughly 25 and 29 °C. The
spikiness is the **seasonal cycle** — it repeats every year and dominates the
picture, which is exactly why we remove it in the next step.
:::

**Step 2** — compute the monthly climatology from 1950–1979 and subtract it:

```python
sst_n34_1950_1979 = sst_n34.isel(time=sst_n34.time.dt.year.isin(range(1950, 1980)))

sst_n34_clim = sst_n34_1950_1979.groupby('time.month').mean()
print("monthly climatology (Jan..Dec), degrees C:")
print(sst_n34_clim.values.round(2))

sst_n34_anom = sst_n34.groupby('time.month') - sst_n34_clim
print("anomalies: mean", float(sst_n34_anom.mean()).__round__(3),
      " std", float(sst_n34_anom.std()).__round__(3))
```

:::{admonition} What the plot shows
:class: note
Plotting `sst_n34_anom` gives a time series now centered on zero, ranging roughly
from −3 to +3 °C. The regular yearly sawtooth of the previous plot is gone;
what's left is irregular, multi-year wobbling — big positive excursions are El
Niño events, big negative ones are La Niña.
:::

**Step 3** — smooth the anomalies with a 5-month running mean, and **step 4** —
standardize by the standard deviation over 1950–1979:

```python
sst_n34_smoothed = sst_n34_anom.rolling(time=5, center=True).mean()
n34_index = sst_n34_smoothed.groupby('time.month') / sst_n34_1950_1979.groupby('time.month').std()

print("Niño3.4 index: min", float(n34_index.min()).__round__(2),
      " max", float(n34_index.max()).__round__(2))

# The strongest El Niño events in the record:
top = n34_index.sortby(n34_index, ascending=False)[:5]
print("strongest El Niño months:")
for t, v in zip(top.time.values, top.values):
    print("  ", str(t)[:7], round(float(v), 2))
```

:::{admonition} What the plot shows
:class: note
`n34_index.plot()` draws the finished Niño3.4 index over the full record: a
smooth wandering line centered on zero, typically within ±2, with occasional large
positive spikes. The famous strong El Niño events of 1982–83, 1997–98 and 2015–16
appear as the tallest peaks — the printed list above names them.
:::

### PCA (EOF) analysis of tropical Pacific SST

Now we will perform a PCA analysis of the tropical Pacific SST, and see if we can
recreate the Niño3.4 index time series using the time series of PCs.

We'll start by selecting a slightly larger region of the original data set
(30S–30N, 120E–300E). This is so that we can better understand the spatial modes
of variability that are learned by the PCA analysis.

```python
sst_pca = sst['sst'].sel(lat=slice(30, -30), lon=slice(120, 300), time=slice('1950', '2013'))
print(sst_pca)
```

To apply PCA to the SST data set, we will first fill in missing values, since PCA
will not work if we have missing values. Where the continents are located is NaN
in this data set, so we will fill any of these values with 0.0.

```python
ssta_pca_masked = sst_pca.fillna(0)
```

First, we create a numpy array `X` that contains the SST in the region we
selected, where the sample number is each time step (1 month), and the spatial
pattern is our feature matrix. In order to use the PCA function in `scikit-learn`,
we have to reshape the array so that it is $n_{samples}$ by $n_{features}$:

```python
X = ssta_pca_masked.values
print("shape as (time, lat, lon):", X.shape)

nt = X.shape[0]
nlat = X.shape[1]
nlon = X.shape[2]

X = X.reshape(nt, nlat * nlon)
print("reshaped to (samples, features):", X.shape)
```

Note what this reshape means: **every grid cell is now a feature**. This is a
genuinely high-dimensional data set — thousands of features — and each "sample" is
one month's map.

Next, we standardize the data by subtracting the mean and dividing by the standard
deviation, then apply PCA:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
PCs = pca.fit_transform(X_scaled)

# Standardize the PCs too, so we can compare them against the Niño3.4 index.
scaler_PCs = StandardScaler()
PCs_std = scaler_PCs.fit_transform(PCs)

print("variance explained by the first 10 EOFs (%):")
print((pca.explained_variance_ratio_[:10] * 100).round(2))
```

:::{admonition} What the plot shows
:class: note
The notebook plots the percentage of variance explained against EOF number for
the first 10 EOFs, as a line with red circular markers. It falls off very steeply:
EOF 1 towers over the rest, EOF 2 is much smaller, and from about EOF 4 onward the
curve is nearly flat and close to zero. The printed array above gives the same
numbers directly.

The takeaway is that just the first two spatial patterns account for most of the
variability in tropical Pacific SST.
:::

To access the spatial modes of variability that were learned by PCA, we use
`.components_`. We have to reshape each one back to the original region size to
interpret it as a map.

```python
EOFs = pca.components_
print("EOFs shape:", EOFs.shape)

EOF1 = EOFs[0, :].reshape(nlat, nlon)
```

:::{admonition} What the plot shows
:class: note
`plt.imshow(EOF1, cmap='RdBu_r')` shows EOF 1 as a map of the tropical Pacific
using a red–white–blue color scale (red positive, blue negative, white near
zero), with the color limits set to ±0.05. EOF 1 appears as broad bands of one
sign in the northern subtropics and the opposite sign near the equator — a
large-scale, roughly **north–south banded** pattern rather than an east–west one.

This first mode corresponds to the **seasonal cycle**, which becomes obvious from
its time series rather than its map.
:::

We can look at how the time series of this spatial pattern changes over time.
We'll just look at the first 120 months (10 years):

```python
import pandas as pd

# The first 10 years of PC1, month by month — the seasonal cycle should repeat.
pc1 = pd.Series(PCs_std[0:120, 0], index=pd.to_datetime(ssta_pca_masked.time[0:120].values))
print("PC1 averaged by calendar month (Jan..Dec):")
print(pc1.groupby(pc1.index.month).mean().round(2).to_string())
```

:::{admonition} What the plot shows
:class: note
Plotting PC1 over the first 120 months gives a strikingly **regular wave** — a
clean oscillation that repeats once per year, ten times over the decade shown.
That regularity is the signature of the seasonal cycle: this dominant mode of
variability is simply summer-versus-winter.

The printed month-by-month averages above show the same thing numerically: the
values swing smoothly from strongly negative in one part of the year to strongly
positive in the opposite part, and they are consistent across years.
:::

Now we'll look at the second EOF, which explains the next most important mode of
variability in the data set.

```python
EOF2 = EOFs[1, :].reshape(nlat, nlon)
```

:::{admonition} What the plot shows
:class: note
EOF 2, drawn the same way (red–white–blue, limits ±0.05), looks quite different
from EOF 1: it shows a strong tongue of one sign spread along the **equator in the
central and eastern Pacific**, flanked by weaker opposite-signed regions to the
west and in the off-equatorial subtropics. This east–west equatorial seesaw is the
classic spatial fingerprint of **ENSO**.

To check this numerically rather than visually, compare the average loading in the
central-eastern equatorial Pacific against the western Pacific:
:::

```python
lats = ssta_pca_masked.lat.values
lons = ssta_pca_masked.lon.values

eq = np.abs(lats) <= 5                      # equatorial band
east = (lons >= 190) & (lons <= 240)        # Niño3.4 longitudes
west = (lons >= 120) & (lons <= 160)        # western Pacific

print("EOF2 mean loading, equatorial central/east Pacific:",
      EOF2[np.ix_(eq, east)].mean().round(4))
print("EOF2 mean loading, equatorial west Pacific:        ",
      EOF2[np.ix_(eq, west)].mean().round(4))
```

The two numbers have **opposite signs** — that is the east–west seesaw, stated as
numbers instead of colors.

This second spatial mode of variability corresponds to ENSO. We can compare its
time series against the time series of Niño3.4 indices that we calculated earlier:

```python
n34index = n34_index.sel(time=slice('1950', '2013')).values

# Correlate PC2 against the Niño3.4 index (ignoring the NaNs from the rolling mean).
pc2 = -1 * PCs_std[:, 1]
ok = ~np.isnan(n34index)
print("correlation between PC2 and the Niño3.4 index:",
      np.corrcoef(pc2[ok], n34index[ok])[0, 1].round(3))
```

:::{admonition} What the plot shows
:class: note
The notebook overlays two time series from 1950 to 2013: PC2 (multiplied by −1)
and the Niño3.4 index. The two lines **track each other closely** — they rise and
fall together, peaking at the same El Niño events (1972–73, 1982–83, 1997–98) and
dipping together during La Niñas. The PC2 line is noticeably jumpier; the Niño3.4
line is smoother.

The correlation printed above puts a number on how well they agree.
:::

Note the `-1 *` on PC2. The sign of a principal component is arbitrary — PCA can
return a pattern and its time series both flipped, and it describes exactly the
same physical mode. We flip PC2 here purely so it lines up with the sign
convention of the Niño3.4 index.

The Niño3.4 index is smoother (because we applied a 5-month moving average), but
overall the PCA analysis has captured the same time scale and mode of variability
as the ENSO metric — **without ever being told that ENSO exists**. We did not give
the algorithm the Niño3.4 region, or any labels. PCA found this pattern purely
from the structure of the variance in the data. That is what makes it unsupervised
learning, and it is why EOF analysis is such a workhorse in climate science.