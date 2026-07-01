# Tutorial on Data Preprocessing (US Wildfires)

:::{admonition} How to work through this tutorial (accessible version)
:class: important
On the standard site this is a Jupyter notebook. Here, work through it as a
**Python script** in VS Code — see [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Put the code into a file such as `datapreparation.py`, adding the pieces below
one section at a time, and run it in the terminal with `python datapreparation.py`
(or run the sections as `# %%` cells if you set that up). Read printed output with
the terminal's Accessible View.

**Plots.** Each `plt.show()` below has a **"What the plot shows"** description so
you know what a sighted reader would see. To explore a plot yourself with sound
and a text/braille readout, render it through **MAIDR** instead of `plt.show()` —
see [Accessible plots with
MAIDR](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/sci_python/trying_maidr.html).
MAIDR handles histograms, bar charts, scatter plots, heatmaps, and boxplots,
which covers every plot in this tutorial.
:::

In this lesson, we will go through an example of how to do initial exploratory
data analysis and data preprocessing for machine learning. To do this, we will
use a data set of US Wildfires from 1990 - 2016. This data set includes the
location and time of 50,000 recent wildfires, as well as information about the
type of vegetation and co-located meteorological data during the time when the
fires occurred. It is a subset of a much larger data set of 1.8 million US
Wildfires.

The data set is available on Kaggle: [US Wildfires and other
attributes](https://www.kaggle.com/datasets/capcloudcoder/us-wildfire-data-plus-other-attributes?select=Wildfire_att_description.txt)

We'll start by downloading the data set and then we will explore the variables in
the data set and pre-process the data for use with machine learning using
`pandas` and the `sci-kit learn` python packages.

## Download the data set

```python
# To facilitate downloading data from Kaggle, we can install this python package.
# In the accessible workflow, run this once in the terminal (not in the script):
#     pip install kagglehub
```

```python
import kagglehub
import os

datapath = kagglehub.dataset_download("capcloudcoder/us-wildfire-data-plus-other-attributes")

print("Path to dataset files:", datapath)
```

```python
os.listdir(datapath)
```

The data set that we downloaded contains two files. The first is a csv that
includes the variables and data, and the txt file provides metadata that
described the variables that are contained in the csv file. We will start by
loading and printing the metadata from the txt file.

```python
with open(os.path.join(datapath, 'Wildfire_att_description.txt'), 'r') as file:
    content = file.read()
print(content)
```

We will use the `pandas` package to read in the csv file.

```python
import pandas as pd

wildfiresdb = pd.read_csv(os.path.join(datapath, 'FW_Veg_Rem_Combined.csv'))
```

## Explore the .csv file

These four calls print text tables, which the terminal's Accessible View reads
line by line. `head()` shows the first few rows; `columns` lists all 43 column
names; `info()` lists each column's type and how many non-null values it has; and
`describe()` prints summary statistics (count, mean, std, min, quartiles, max) for
the numeric columns.

```python
wildfiresdb.head()
```

```python
wildfiresdb.columns
```

```python
wildfiresdb.info()
```

```python
wildfiresdb.describe()
```

## Explore the distributions of the variables in our data set

Since we have 43 variables, we will focus on a subset of these variables for now.
We want to understand how the variables are distributed. We'll focus on the size
of the fire, its location, how long it took to put out, the year it was
discovered, the type of vegetation, and historical meteorological variables
(Temperature, Wind, Humidity, and Precipitation) averaged over the time it took
to put out the fire.

```python
variables = ["fire_size", "fire_size_class", "latitude", "longitude", "putout_time",
             "disc_pre_year", "Vegetation", "Temp_cont", "Wind_cont", "Hum_cont", "Prec_cont"]
```

First we will put this subsection of the data into a `DataFrame` object by using
the variable names we have chosen.

```python
df = wildfiresdb[variables].copy()
df.head()
```

You can explore the data in various ways. First, how many samples are in our data
set?

```python
print("Number of rows: " + str(len(df)))
```

We can use `matplotlib` to visualize the marginal distributions of our variables
using the `hist` function:

```python
import matplotlib.pyplot as plt

# extra code – the next 5 lines define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

df.hist(bins=50, figsize=(12, 8))
plt.show()
```

:::{admonition} What the plot shows
:class: note
A grid of histograms, one per numeric column. Most are **highly right-skewed**:
`fire_size`, for example, has almost all its mass at very small values with a long
thin tail toward a few huge fires. `latitude` and `longitude` show the spatial
spread of US fires (several clustered peaks rather than one). `disc_pre_year` is
roughly spread across 1990–2016. The meteorological columns (`Temp_cont`,
`Wind_cont`, `Hum_cont`, `Prec_cont`) each show a spike at 0 / −1 — those are
missing-value codes we fix later. To hear each distribution, render one column at
a time through MAIDR.
:::

Another useful python package that we can use is `seaborn`.

*Warning! Seaborn can typically be quite slow, especially if you have a large data
set or many variables. One simple way to deal with this is to randomly subsample
the data, and visualize only a fraction of the data set.*

```python
import seaborn as sns
```

We can look at how the variables are correlated with one another using the
`PairGrid` function. We need both of the lines below to create the plot. The first
line just constructs a blank grid of subplots with each row and column
corresponding to a numeric value in the data set. The second line draws a
bivariate plot on every axis.

```python
g = sns.PairGrid(df)
g.map(sns.scatterplot)
```

:::{admonition} What the plot shows
:class: note
A square matrix of scatter plots (a "pair plot"): every numeric variable plotted
against every other. The diagonal compares a variable with itself. This is a
visual way to eyeball which pairs of variables move together. It is dense and hard
to read even for sighted users; the correlation heatmap and the `corr()` table
below give the same information as **numbers**, which are more useful with a screen
reader.
:::

We can also look for correlations between the variables in our data set, using the
`pandas` DataFrame `corr` method. The `numeric_only` argument needs to be set to
True to avoid an error. This makes sure it only uses the variables in our
dataframe that have numeric values.

```python
corr_matrix = df.corr(numeric_only=True)
```

```python
# Print the correlation matrix as a text table — this is the screen-reader-
# friendly version of the heatmap below.
print(corr_matrix.round(2))

# Create a heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
```

:::{admonition} What the plot shows
:class: note
The heatmap colors each cell of `corr_matrix` from −1 (strong negative) through 0
(none) to +1 (strong positive). The printed table just above shows the same
numbers — read it instead. Correlations near +1 or −1 (ignoring the 1.0 diagonal)
are the pairs of variables that carry redundant information; correlations near 0
are effectively independent.
:::

## Visualize the data spatially

Since we have latitude and longitude, we can also visualize the spatial
distribution of the points in our data set. Looking at fire size, it's clear that
the largest fires occur for the most part in the Western US.

```python
import numpy as np

df.plot(kind='scatter', x='longitude', y='latitude', grid=True, alpha=0.2,
        s=np.log(df["fire_size"]), label="Fire Size",
        c=np.log(df["fire_size"]), cmap="jet", colorbar=True, legend=True,
        sharex=False, figsize=(10, 7))
plt.show()
```

:::{admonition} What the plot shows
:class: note
A map-shaped scatter plot: each fire is a dot at its (longitude, latitude), with
the dot's size and color both set by `log(fire_size)`. The dots trace the outline
of the continental US. The **largest, hottest-colored dots cluster in the Western
US** (California, the Northwest, the Rockies), while the East has many small fires.
Note: this is a 2-D map-style scatter, which MAIDR does not sonify — rely on this
description, and on the numeric `latitude`/`longitude`/`fire_size` summaries from
`describe()`.
:::

## Handling missing/irregular data

The column `putout_time` contains many NaN values, because data on how long it
took to put out the fire is not available for every fire. First, we can try to
understand why data might be missing.

```python
no_putout_time = pd.isna(df['putout_time'])
```

Let's compare the mean size of the fires which have no data for `putout_time` to
the mean size of the fires which do have data:

```python
print("Mean size of fires with NaN for putout_time:")
(df['fire_size'][no_putout_time]).mean()
```

```python
print("Mean size of fires with values for putout_time:")
(df['fire_size'][~no_putout_time]).mean()
```

This suggests that smaller fires typically don't have data about `putout_time`
associated with them.

`putout_time` is currently formatted as string timestamps, rather than floating
point numbers, and they are unfortunately not all in the same format. We can see
this by printing out the unique values from this column.

```python
unique = df['putout_time'].unique()
unique
```

Since the information that we are interested in is the first "word" in each
string, we can use the string split method to get the number of days it took to
put out a fire, and put this into a new column in our data frame. We also want the
variable to be a floating point number (not a string) so that we can use it in our
(future) machine learning model.

```python
df['putout_time_float'] = df['putout_time'].str.split().str[0].astype(float)
df['putout_time_float']
```

```python
df['putout_time_float'].describe()
```

```python
df
```

We'll drop the original column now, since we have fixed the irregular data.

```python
df = df.drop(['putout_time'], axis=1)
df.head()
```

If we look at the meteorological variables (Prec_cont, Wind_cont, Temp_cont,
Hum_cont), we also notice that the distributions look strange. This is because
they have 0 and -1 for missing values. We can guess this because it's not
physically reasonable that these values are negative, or that the Wind_cont,
Temp_cont, Hum_cont are exactly 0.0000, especially during fire season.

```python
fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
axs[0].hist(df["Prec_cont"], bins=50)
axs[1].hist(df["Wind_cont"], bins=50)
axs[2].hist(df["Temp_cont"], bins=50)
axs[3].hist(df["Hum_cont"], bins=50)
axs[0].set_xlabel("Prec_cont")
axs[1].set_xlabel("Wind_cont")
axs[2].set_xlabel("Temp_cont")
axs[3].set_xlabel("Hum_cont")
axs[0].set_ylabel("Number of Fires")
plt.show()
```

:::{admonition} What the plot shows
:class: note
Four side-by-side histograms of the four meteorological columns. Each has a
**tall, isolated spike at 0 and/or −1** (the missing-value codes) sitting apart
from the physically-realistic bulk of the distribution. That gap is the visual
tell that 0 and −1 are placeholders, not real measurements.
:::

```python
df["Temp_cont"]
```

We can replace these suspicious values with NaN's using the `replace` method.

```python
df["Temp_cont"] = df["Temp_cont"].replace([0.0000, -1.0000], np.nan)
df["Hum_cont"]  = df["Hum_cont"].replace([0.0000, -1.0000], np.nan)
df["Wind_cont"] = df["Wind_cont"].replace([0.0000, -1.0000], np.nan)
df["Prec_cont"] = df["Prec_cont"].replace([-1.0000], np.nan)
```

```python
fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
axs[0].hist(df["Prec_cont"], bins=50)
axs[1].hist(df["Wind_cont"], bins=50)
axs[2].hist(df["Temp_cont"], bins=50)
axs[3].hist(df["Hum_cont"], bins=50)
axs[0].set_xlabel("Prec_cont")
axs[1].set_xlabel("Wind_cont")
axs[2].set_xlabel("Temp_cont")
axs[3].set_xlabel("Hum_cont")
axs[0].set_ylabel("Number of Fires")
plt.show()
```

:::{admonition} What the plot shows
:class: note
The same four histograms after replacing 0 and −1 with NaN. The isolated spikes
are **gone**, leaving smoother, more physically-reasonable distributions (e.g.
`Temp_cont` now looks roughly bell-shaped rather than having a spike at zero).
:::

Now that we have fixed the irregular data, we need to figure out what to do with
the NaN values. One very simple way to deal with these missing data is to just
drop all of the rows where there are no values for `putout_time`, using the
`dropna()` method:

```python
df_cleaned = df.dropna().copy()
```

A disadvantage of dropping rows it that we end up with less data, so other
approaches such as imputation can be preferable if our data set is small. After
removing all of the bad rows, we are down to 10% of our original data set!

```python
print("Number of rows (before removing NaN's): " + str(len(df)))
print("Number of rows (after removing NaN's): " + str(len(df_cleaned)))
```

We can also remove the very high outliers for `putout_time_float` (>99.995%):

```python
q_hi = df_cleaned['putout_time_float'].quantile(0.99995)
print(q_hi)

df_filtered = df_cleaned[(df_cleaned['putout_time_float'] < q_hi)].dropna().copy()
```

```python
print("Number of rows (before removing outliers): " + str(len(df_cleaned)))
print("Number of rows (after removing outliers): " + str(len(df_filtered)))
```

We can also fill the NaN's with a value instead, using the `fillna()` method:

```python
df_filled = df.fillna(0).copy()
df_filled
```

Rather than filling with a specific value, it's also possible to fill with the
`median` or `mean` of the column.

```python
df_fill_mean = df.copy()

meantime = df['putout_time_float'].mean()
df_fill_mean['putout_time_float'].fillna(meantime, inplace=True)
df_fill_mean
```

Finally, we can interpolate, using the `interpolate()` method. Interpolate refers
to filling in using the values based on estimating from surrounding non-NaN data
in a column.

```python
df_interp = df.interpolate().copy()
```

Interpolation doesn't make a lot of sense for this data set since the fires are
independent of one another. Interpolation makes more sense in the context of a
time series, where we are missing specific values in our sequence. To illustrate
this, we will look at the effect of using the `Interpolate()` method on a sine
wave that has 20% of its data missing.

This code just generates the time series and puts it into a dataframe.

```python
np.random.seed(42)
n = 100
t = np.linspace(0, 2 * np.pi, n)         # 0 → 2π
y = np.sin(t)
y_noisy = y + np.random.normal(0, 0.1, n)

# knock out ~10 % of points
mask = np.random.choice(n, size=int(0.20 * n), replace=False)
y_noisy[mask] = np.nan

dfsine = pd.DataFrame({"t": t, "sine": y, "noisy_sine": y_noisy})
```

We can apply the interpolate function to the sine wave to estimate the missing
values.

```python
dfsine_interp = dfsine.interpolate().copy()
```

```python
plt.scatter(dfsine['t'], dfsine['noisy_sine'], marker='o', facecolors='r', edgecolors='r', label="original")
plt.scatter(dfsine_interp['t'], dfsine_interp['noisy_sine'], marker='o', facecolors='none', edgecolors='k', label="interpolated")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
```

:::{admonition} What the plot shows
:class: note
A scatter plot of a noisy sine wave over one period (t from 0 to 2π). Red filled
dots are the surviving original points; black open circles are the values
`interpolate()` filled in where data was missing. The filled-in circles land
**along the sine curve, between their neighbors** — showing interpolation works
well when points are ordered in a sequence, unlike the independent-fire case.
:::

Using interpolation or filling missing values can potentially bias our data sets,
so it's important to keep in mind what problem we are interested in solving when
choosing a method to deal with missing data.

## Feature Scaling

The `sci-kit learn` package has a number of built in functions for scaling
variables.

MinMaxScaler scales value between a minimum and maximum. We can set what range we
want to end up with. For a sample x, the MinMaxScaler is calculated as:

```
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min
```

In words: subtract the column minimum and divide by the column range to get a
value between 0 and 1 (`X_std`), then stretch that onto your chosen `[min, max]`
range.

```python
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
latitude_min_max_scaled = min_max_scaler.fit_transform(df[["latitude"]])
```

The StandardScaler removes the mean and scales to unit variance. For a sample x,
the standard scaler is calculated as `z = (x - u) / s`, where `u` is the column
mean and `s` is the column standard deviation. The result has mean 0 and standard
deviation 1.

```python
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
latitude_std_scaled = std_scaler.fit_transform(df[["latitude"]])
```

```python
fig, axs = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
axs[0].hist(df["latitude"], bins=50)
axs[1].hist(latitude_min_max_scaled, bins=50)
axs[2].hist(latitude_std_scaled, bins=50)
axs[0].set_xlabel("Latitude")
axs[1].set_xlabel("Latitude (Min_Max_Scaled)")
axs[2].set_xlabel("Latitude (Std_Scaled)")
axs[0].set_ylabel("Number of Fires")
plt.show()
```

:::{admonition} What the plot shows
:class: note
Three histograms of the same `latitude` data. They have the **identical shape** —
scaling only changes the numbers on the x-axis, not the distribution. The first is
on the raw latitude scale (roughly 25–50); the MinMax-scaled one runs from −1 to
+1; the Standard-scaled one is centered on 0 with most values within about ±3.
:::

We can also do custom transformations using `sci-kit learn`'s FunctionTransformer
method. For example, fire_size is not normally distributed, but rather strongly
skewed towards smaller fires. We can log scale this variable using the
FunctionTransformer.

```python
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_fire_size = log_transformer.transform(df_filtered[["fire_size"]])
```

```python
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
axs[0].hist(df_filtered["fire_size"], bins=50)
axs[1].hist(log_fire_size, bins=50)
axs[0].set_xlabel("Fire Size")
axs[1].set_xlabel("Log(Fire Size)")
axs[0].set_ylabel("Number of Fires")
plt.show()
```

:::{admonition} What the plot shows
:class: note
Two histograms of `fire_size`. The left (raw) one is **extremely right-skewed** —
a tall bar at small sizes and an almost-invisible tail. The right one, after a log
transform, is **much more symmetric / bell-shaped**. This is why log-scaling a
skewed variable helps many ML models: it spreads the crowded small values out.
:::

```python
df_filtered["log_fire_size"] = log_fire_size
```

## Encoding categorical variables

The `fire_size_class` is a categorical variable (it is a letter between A - G).

```python
fire_size_class = df[["fire_size_class"]]
fire_size_class.head(8)
```

We can also check if our data set is balanced or not, by printing out the number
of samples of each class in our data set.

```python
fire_size_class_counts = df["fire_size_class"].value_counts()
print(fire_size_class_counts)
```

There are several different ways that we can encode this categorical variable for
machine learning models. One is using the OrdinalEncoder function from `sci-kit
learn`. This function will transform categories (B, C, D, E, ...) to an integer
label (0, 1, 2, 3, ...). We will end up with a single column per original feature
(or variable) in our data set.

OrdinalEncoder can be useful if our categories have a particular ordering to them
(like fire size class does). However, it might not be a good choice for a feature
like the `state` in which a fire occurred, since we don't have a particular
ordering.

```python
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
fire_size_class_encoded = ordinal_encoder.fit_transform(fire_size_class)
```

```python
fire_size_class
```

```python
fire_size_class_encoded[:8]
```

```python
ordinal_encoder.categories_
```

An alternative way that we can encode a categorical variable is using the
OneHotEncoder function from `sci-kit learn`. This function converts each category
into a separate binary column. For example, a sample that is class B, would be one
hot encoded as `[1, 0, 0, 0, 0, 0]`, or for a sample that is a fire of class E, it
would be encoded as `[0, 0, 0, 1, 0, 0]`.

We will end up with as many columns as we have classes in our data set. For the
fire_size_class variable, this would add 6 columns, since there are 6 possible
classes represented in our data set.

This is best to use when our categorical variables are nominal (unordered), and is
best for many of the types of models that assume inputs are numerical and
unstructured (linear models, neural networks), which we will discuss later in the
course.

```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
fire_size_class_hot = cat_encoder.fit_transform(fire_size_class)
```

```python
fire_size_class_hot
```

```python
fire_size_class_hot.toarray()
```

```python
cat_encoder.categories_
```

## Creating a pipeline to preprocess our features

```python
df_filtered
```

Let's say that we are interested in predicting the time it takes to put out a
fire, given the other variables in our data set. First, we will scale the
`putout_time_float` using the standard scaler and put it into an array called `y`,
since it's our target variable.

```python
std_scaler = StandardScaler()
y = std_scaler.fit_transform(df_filtered[["putout_time_float"]])
```

Next we will create a feature matrix `X`, which will be the input to our model.
First let's remove our target variable from the dataframe.

```python
features = df_filtered.copy()
features = df_filtered.drop(["putout_time_float"], axis=1)
```

```python
features
```

Now we can apply a pipeline of transformations to our data set. `sci-kit learn`
allows us to create separate pipelines for categorical and numerical variables and
then run them on our entire dataframe simultaneously.

```python
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
```

We need to separate our variables based on how we want to transform them though.

```python
categorical_cols = ["fire_size_class"]
numerical_cols = ["fire_size", "latitude", "longitude", "disc_pre_year", "Vegetation",
                  "Temp_cont", "Wind_cont", "Hum_cont", "Prec_cont", "log_fire_size"]
```

```python
cat_pipeline = make_pipeline(OrdinalEncoder(), StandardScaler())
num_pipeline = make_pipeline(MinMaxScaler())
```

```python
preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_cols),
    ("cat", cat_pipeline, categorical_cols)])
```

```python
X = preprocessor.fit_transform(features)
```

```python
X.shape
```

```python
preprocessor.get_feature_names_out()
```

```python
g = sns.PairGrid(pd.DataFrame(X, columns=preprocessor.get_feature_names_out()))
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
```

:::{admonition} What the plot shows
:class: note
Another pair plot, now of the fully preprocessed feature matrix `X`. Because every
numerical column was MinMax-scaled, the axes all run over the same small range, so
you can visually confirm the features are now on comparable scales. As before, the
`X.shape` and `get_feature_names_out()` output above give the accessible summary of
what the matrix contains.
:::

## Creating a training, validation, and test data set

`Sci-kit learn` has several built in functions to facilitate creating separate
training and test data sets. The simplest one to use is `train_test_split`, which
returns the features and targets split into train and test data sets. If we want
to also create a validation data set using this function, we will need to split
the data set twice.

```python
from sklearn.model_selection import train_test_split
```

```python
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.5, random_state=42)
```

```python
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)
```

```python
print(X_train.shape, X_val.shape, X_test.shape)
```

## Exercises

Try these short extensions of the tutorial. Each one reuses a method we covered
above, applied to a different variable or with a different option. Add each in a
new section of your script. For any plot, use a MAIDR render (or add a printed
`describe()` / `value_counts()`) so you can read the result.

1. **Histograms and bins.** Plot a histogram of just the `Wind_cont` column using
   `bins=30` instead of `bins=50`. How does changing the number of bins change
   what you can see in the distribution?

2. **Correlation heatmap.** Regenerate the correlation heatmap with a different
   colormap (e.g. `cmap='viridis'`). Which pair of variables is the most strongly
   correlated, and which is the least? (Read the printed `corr()` table to answer.)

3. **Why is data missing?** The tutorial compared the mean `fire_size` for fires
   with vs. without a `putout_time`. Repeat this comparison for a different
   variable: is the mean `fire_size` different for fires that are missing
   `Temp_cont` compared to those that have it?

4. **Filling with the median.** The tutorial filled the missing
   `putout_time_float` values with the column *mean*. Make a new dataframe that
   fills them with the column *median* instead. How do the mean and median of this
   column differ, and why?

5. **Choosing an outlier threshold.** The tutorial dropped the top 0.005% of
   `putout_time_float` values (the 99.995th percentile). Redo the filtering with a
   less extreme cutoff — the 99th percentile — and report how many rows remain
   compared to the tutorial.

6. **MinMaxScaler on a new variable.** The tutorial applied a
   `MinMaxScaler(feature_range=(-1, 1))` to `latitude`. Apply a MinMaxScaler with
   the range `(0, 1)` to `longitude` instead, and plot a histogram of the scaled
   values.

7. **Scaling a skewed variable.** Apply the `StandardScaler` to `fire_size`, and
   separately apply a square-root transform using a
   `FunctionTransformer(np.sqrt)`. Plot both results and compare them to the log
   transform from the tutorial. Which transform makes `fire_size` look most
   symmetric?

8. **Encoding a different category.** The tutorial one-hot-encoded
   `fire_size_class`. Apply the `OneHotEncoder` to the `Vegetation` column
   instead. How many new columns are created, and what determines that number?

9. **Changing a pipeline step.** In the tutorial's `num_pipeline`, the numerical
   columns are scaled with `MinMaxScaler`. Change that step to use `StandardScaler`
   instead, re-run the `ColumnTransformer`, and check the shape and feature names
   of the resulting `X`.

10. **A different train/test split.** The tutorial made a 50/25/25
    train/validation/test split. Use `train_test_split` to make an 80/10/10 split
    instead, and print the shape of each of the three sets.