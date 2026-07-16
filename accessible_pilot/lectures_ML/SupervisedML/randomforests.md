# Tutorial on Random Forests (Wildfire cause prediction)

:::{admonition} How to work through this tutorial (accessible version)
:class: important
On the standard site this is a Jupyter notebook. Here, work through it as a
**Python script** in VS Code — see [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Put the code into a file such as `randomforests.py`, adding the pieces below one
section at a time, and run it in the terminal with `python randomforests.py` (or
run the sections as `# %%` cells if you set that up). Read printed output with
the terminal's Accessible View.

**Plots.** Each `plt.show()` below has a **"What the plot shows"** description so
you know what a sighted reader would see. To explore a plot yourself with sound
and a text/braille readout, render it through **MAIDR** instead of `plt.show()` —
see [Accessible plots with
MAIDR](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/sci_python/trying_maidr.html).
MAIDR handles histograms, bar charts, and heatmaps, which covers the plots here.
For confusion matrices and classification results, we also print the numbers so
you never have to rely on the picture.
:::

In this lesson, we will learn about decision trees and random forests and how
they can be used for supervised machine learning tasks such as classification. A
decision tree is an algorithm that classifies or predicts a target by making a
sequence of yes/no decisions about the values of different features. A random
forest uses the ensemble vote of many decision trees to classify or predict a
value.

We will use the data set from the paper "Inference of Wildfire Causes From Their
Physical, Biological, Social and Management Attributes" by Pourmohamad et al.,
Earth's Future, 2025
([paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024EF005187)). In
this paper, they explored whether it is possible to determine the cause of a
wildfire (in cases where the cause is unknown) based on data from other wildfires
where the cause was known.

**References:**

[1] Pourmohamad, Y., Abatzoglou, J. T., Fleishman, E., Short, K. C., Shuman, J.,
AghaKouchak, A., et al. (2025). Inference of wildfire causes from their physical,
biological, social and management attributes. Earth's Future, 13, e2024EF005187.
<https://doi.org/10.1029/2024EF005187>

[2] Pourmohamad, Y., Abatzoglou, J. T., Belval, E. J., Fleishman, E., Short, K.,
Reeves, M. C., Nauslar, N., Higuera, P. E., Henderson, E., Ball, S., AghaKouchak,
A., Prestemon, J. P., Olszewski, J., and Sadegh, M.: Physical, social, and
biological attributes for improved understanding and prediction of wildfires: FPA
FOD-Attributes dataset, Earth Syst. Sci. Data, 16, 3045–3060,
<https://doi.org/10.5194/essd-16-3045-2024>, 2024.

[3] Pourmohamad, Y. (2024). Inference of Wildfire Causes from Their Physical,
Biological, Social and Management Attributes (0.1). Zenodo.
<https://doi.org/10.5281/zenodo.11510677>

## Load in the data set

The data set can be downloaded from
[https://zenodo.org/records/11510677](https://zenodo.org/records/11510677). In
the accessible workflow, download the file `FPA_FOD_west_cleaned.csv` from that
Zenodo record (about 140 MB) into the same folder as your script — for example,
run this once in the terminal (not in the script):

```python
# wget "https://zenodo.org/records/11510677/files/FPA_FOD_west_cleaned.csv"
```

Then read it in:

```python
import pandas as pd
import seaborn as sns
import os
import numpy as np

data = pd.read_csv("FPA_FOD_west_cleaned.csv")
```

Look at the first few rows and the column names:

```python
data.head()
data.columns
```

The data set includes meteorological, topological, social, and fire management
variables:

- `DISCOVERY_DOY`: Day of year on which the fire was discovered or confirmed to exist
- `FIRE_YEAR`: Calendar year in which the fire was discovered or confirmed to exist
- `STATE`: Two-letter alphabetic code for the state in which the fire burned (or originated)
- `FIPS_CODE`: Five digit FIPS 6-4 code for counties and equivalent entities
- `Annual_etr`: Annual total reference evapotranspiration (mm)
- `Annual_temperature`: Annual average temperature (K)
- `pr`: Precipitation amount (mm)
- `tmmn`: Minimum temperature (K)
- `vs`: Wind velocity at 10 m above ground (m/s)
- `fm100`: 100-hour dead fuel moisture (%)
- `fm1000`: 1000-hour dead fuel moisture (%)
- `bi`: Burning index (NFDRS fire danger index)
- `vpd`: Mean vapor pressure deficit (kPa)
- `erc`: Energy release component (NFDRS fire danger index)
- `Elevation_1km`: Average elevation in 1 km radius around the ignition point
- `Aspect_1km`: Average aspect in 1 km radius around the ignition point
- `erc_Percentile`: Percentile range of energy release component
- `Slope_1km`: Average slope in 1 km radius around the ignition point
- `TPI_1km`: Average Topographic Position Index in 1 km radius around the ignition point
- `EVC`: Existing Vegetation Cover — percent cover of the live canopy layer (%)
- `Evacuation`: Estimated ground transport time (hours) from the ignition point to a hospital
- `SDI`: Suppression difficulty index — relative difficulty of fire control
- `FRG`: Fire regime group — presumed historical fire regime
- `No_FireStation_5.0km`: Number of fire stations within a 5 km radius of the ignition point
- `Mang_Name`: The land manager or administrative agency, standardized for the US
- `GAP_Sts`: GAP status code — management intent to conserve biodiversity
- `GACC_PL`: Geographic Area Coordination Center (GACC) Preparedness Level
- `GDP`: Annual Gross Domestic Product Per Capita
- `GHM`: Cumulative measure of human modification of lands within 1 km of the ignition point
- `NDVI-1day`: Normalized Difference Vegetation Index on the day prior to ignition
- `NPL`: National Preparedness Level
- `Popo_1km`: Average population density within a 1 km radius of the ignition point
- `RPL_THEMES`: Social Vulnerability Index (overall percentile ranking)
- `RPL_THEME1`–`RPL_THEME4`: Percentile rankings for the four SVI theme summaries
- `Distance2road`: Distance to the nearest road

Check the size of the data set and the data types:

```python
len(data)
data.info()
```

Count how many fires there are of each cause:

```python
firecauses = data['NWCG_GENERAL_CAUSE'].value_counts()
print(firecauses)
```

This prints the count of fires per cause. There are 13 categories. The largest
are `Natural` (168,349) and `Missing data/not specified/undetermined` (150,427),
followed by `Equipment and vehicle use` (48,994), `Debris and open burning`
(40,516), `Recreation and ceremony` (38,665), and `Arson/incendiarism` (28,090).
The smallest categories have only 1,500–6,500 fires each. This imbalance between
classes matters later when we evaluate the classifier.

Some columns use sentinel values (like -9999 or -999) to mean "missing." Replace
those with `NaN` so they can be dropped:

```python
data.loc[data["GHM"] < 0.0, "GHM"] = np.nan
data.loc[data["SDI"] < 0.0, "SDI"] = np.nan
data['FRG'] = data['FRG'].replace(-9999, np.nan)
data["RPL_THEMES"] = data["RPL_THEMES"].replace(-999.0, np.nan)
data["RPL_THEME1"] = data["RPL_THEME1"].replace(-999.0, np.nan)
data["RPL_THEME2"] = data["RPL_THEME2"].replace(-999.0, np.nan)
data["RPL_THEME3"] = data["RPL_THEME3"].replace(-999.0, np.nan)
data["RPL_THEME4"] = data["RPL_THEME4"].replace(-999.0, np.nan)
```

Look at the distribution of every numeric attribute:

```python
import matplotlib.pyplot as plt

data.hist(bins=50, figsize=(12, 8))
plt.show()
```

:::{admonition} What the plot shows
:class: note
A grid of histograms, one per numeric column, each with 50 bins. It gives a
quick sense of the shape and range of every variable — some (like `FIRE_YEAR` or
day-of-year) are roughly uniform, several meteorological variables are bell-
shaped, and several social/percentile variables are skewed or pile up at 0 or
100. To read any single column's distribution yourself, render just that column
through MAIDR, for example `data["erc"].hist(bins=50)`, or print
`data["erc"].describe()` for the min, max, mean, and quartiles.
:::

Drop any row that still has a missing value:

```python
data_cleaned = data.dropna().reset_index(drop=True)
```

## Separate out the fires with no known cause

First, separate all of the fires where `NWCG_GENERAL_CAUSE` has the label
`Missing data/not specified/undetermined`. We sort them to the front, then split
into an "unknown" set and a "known" set:

```python
data_sorted = data_cleaned.iloc[
    np.where(data_cleaned['NWCG_GENERAL_CAUSE'] == 'Missing data/not specified/undetermined')[0].tolist() +
    np.where(data_cleaned['NWCG_GENERAL_CAUSE'] != 'Missing data/not specified/undetermined')[0].tolist()
].reset_index(drop=True).copy()

data_unknown = data_sorted.loc[data_sorted["NWCG_GENERAL_CAUSE"] == "Missing data/not specified/undetermined"].reset_index(drop=True).copy()
data_known = data_sorted.loc[data_sorted["NWCG_GENERAL_CAUSE"] != "Missing data/not specified/undetermined"].reset_index(drop=True).copy()

data_known["NWCG_GENERAL_CAUSE"].value_counts()
```

Among the fires with a known cause, the counts are: Natural 168,126; Equipment
and vehicle use 48,895; Debris and open burning 40,450; Recreation and ceremony
38,498; Arson/incendiarism 28,035; Smoking 13,510; Misuse of fire by a minor
11,508; Power generation/transmission/distribution 6,453; Fireworks 6,348;
Railroad operations and maintenance 3,062; Other causes 2,064; Firearms and
explosives use 1,584.

Only the `Natural` class is due to natural causes (usually lightning); all the
other categories are related to human activity. So we can also label fires as
"natural" or "anthropogenic". We create a binary variable `IsNatural` that is 1
(True) for natural fires and 0 (False) for everything else:

```python
data_known["IsNatural"] = (data_known["NWCG_GENERAL_CAUSE"] == "Natural").astype(int)
data_known["IsNatural"].value_counts()
```

This gives 200,407 anthropogenic fires (0) and 168,126 natural fires (1) — a
fairly balanced binary split.

## Data pre-processing

For decision trees and random forests we generally don't have to worry as much
about scaling (compared with models like neural networks), since they work by
finding threshold values in the data.

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

causes = data_known[['NWCG_GENERAL_CAUSE']]
isnatural = data_known[["IsNatural"]]
features = data_known.copy().drop(["NWCG_GENERAL_CAUSE", "IsNatural"], axis=1)
features_unknown = data_unknown.copy().drop(["NWCG_GENERAL_CAUSE"], axis=1)
```

We create two sets of labels. The first is the **binary** label (natural vs.
anthropogenic):

```python
y_binary = isnatural.to_numpy()
classnames_binary = ["Anthropogenic", "Natural"]
```

The second is **multi-class**, covering all the possible causes in the
`NWCG_GENERAL_CAUSE` column. We turn the text labels into integers with an
ordinal encoder:

```python
ordenc = OrdinalEncoder()
y_multiclass = ordenc.fit_transform(causes)
classnames_multi = ordenc.categories_[0]
print(classnames_multi)
```

The 12 class names, in encoded order, are: Arson/incendiarism; Debris and open
burning; Equipment and vehicle use; Firearms and explosives use; Fireworks;
Misuse of fire by a minor; Natural; Other causes; Power
generation/transmission/distribution; Railroad operations and maintenance;
Recreation and ceremony; Smoking.

Now build a pipeline to transform the feature columns. The single categorical
column (`STATE`) is ordinal-encoded then scaled; the numeric columns are scaled:

```python
categorical_cols = ["STATE"]
numerical_cols = ['DISCOVERY_DOY', 'FIRE_YEAR', 'FIPS_CODE', 'Annual_etr', 'Annual_precipitation', 'Annual_tempreture',
                  'pr', 'tmmn', 'vs', 'fm100', 'fm1000', 'bi', 'vpd', 'erc', 'Elevation_1km', 'Aspect_1km', 'erc_Percentile',
                  'Slope_1km', 'TPI_1km', 'EVC', 'Evacuation', 'SDI', 'FRG', 'No_FireStation_5.0km', 'Mang_Name', 'GAP_Sts',
                  'GACC_PL', 'GDP', 'GHM', 'NDVI-1day', 'NPL', 'Popo_1km', 'RPL_THEMES', 'RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3',
                  'RPL_THEME4', 'Distance2road']

cat_pipeline = make_pipeline(OrdinalEncoder(), StandardScaler())
num_pipeline = make_pipeline(StandardScaler())

preprocessor = ColumnTransformer([
    ("n", num_pipeline, numerical_cols),
    ("c", cat_pipeline, categorical_cols)])

X_known = preprocessor.fit_transform(features)
```

We apply the **same** pipeline to the unknown fires, using `transform` rather
than `fit_transform`. This matters: the scalings are learned from `features`
(the known fires) and then applied identically to both sets, so the models we
train later see consistent inputs.

```python
X_unknown = preprocessor.transform(features_unknown)
print(X_known.shape, X_unknown.shape)

featurenames = preprocessor.get_feature_names_out()
print(featurenames)
```

## Training, validation, and test split

Split the fires with a known cause into training, validation, and test sets. We
first build an index array `z_known` and split the index, so we can later pull
out either the binary or multi-class labels for each split.

```python
from sklearn.model_selection import train_test_split

z_known = np.arange(0, X_known.shape[0])
X_train, X_val_test, z_train, z_val_test = train_test_split(X_known, z_known, test_size=0.2, random_state=42)
X_val, X_test, z_val, z_test = train_test_split(X_val_test, z_val_test, test_size=0.5, random_state=42)

y_multiclass_train = y_multiclass[z_train].ravel()
y_multiclass_test = y_multiclass[z_test].ravel()
y_multiclass_val = y_multiclass[z_val].ravel()

y_binary_train = y_binary[z_train].ravel()
y_binary_test = y_binary[z_test].ravel()
y_binary_val = y_binary[z_val].ravel()

print(X_train.shape, X_val.shape, X_test.shape)
print(y_binary_train.shape, y_binary_val.shape, y_binary_test.shape)
```

This gives an 80% training set, 10% validation set, and 10% test set.

## Train logistic regression (natural vs. human causes)

As a baseline for the binary problem, train a logistic regression classifier:

```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="lbfgs", random_state=42)
log_reg.fit(X_train, y_binary_train)

log_reg.score(X_train, y_binary_train)   # ~0.881 training accuracy
log_reg.score(X_val, y_binary_val)       # ~0.882 validation accuracy
```

The training and validation accuracies are both about 0.88, so the model is not
over-fitting.

Get predictions and build a **confusion matrix** — a 2×2 table of predicted vs.
true labels:

```python
y_train_predicted = log_reg.predict(X_train)
y_val_predicted = log_reg.predict(X_val)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

print(confusion_matrix(y_binary_val, y_val_predicted))
```

Rather than the color heatmap that `ConfusionMatrixDisplay` draws, **print the
matrix as numbers** as above. Each row is a true class (row 0 = Anthropogenic,
row 1 = Natural) and each column is a predicted class. The diagonal entries are
correct predictions; off-diagonal entries are errors. To see the display
normalized by true class (each row sums to 1), you can print it directly:

```python
cm = confusion_matrix(y_binary_val, y_val_predicted, normalize='true')
print("Rows = true [Anthropogenic, Natural], Cols = predicted")
print(cm)
```

### Classification metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

**Accuracy** = (TP + TN) / (TP + TN + FP + FN), where TP = true positive, TN =
true negative, FP = false positive, FN = false negative. Accuracy of 1.0 is a
perfect classifier and 0.0 is no skill. Accuracy can be misleading when the
classes are imbalanced.

```python
accuracy_score(y_binary_val, y_val_predicted)   # ~0.882
```

**Precision** = TP / (TP + FP) — how often the class the model predicts is
actually correct. High precision means few false positives.

```python
precision_score(y_binary_val, y_val_predicted)  # ~0.866
```

**Recall** = TP / (TP + FN) — how many of the true members of a class the model
finds. High recall means few false negatives.

```python
recall_score(y_binary_val, y_val_predicted)     # ~0.880
```

**F1 score** balances precision and recall: F1 = 2 / (recall⁻¹ + precision⁻¹).

```python
f1_score(y_binary_val, y_val_predicted)         # ~0.873
```

### ROC curve

The Receiver Operator Characteristic (ROC) curve evaluates a binary classifier.
There is a trade-off between true positives and false positives depending on
where we set the decision threshold, and the ROC curve visualizes this trade-off.
A classifier with no skill lies on the diagonal dashed line; a perfect classifier
reaches the top-left corner.

```python
from sklearn.metrics import RocCurveDisplay

svc_disp = RocCurveDisplay.from_estimator(log_reg, X_val, y_binary_val)
plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), linestyle='--')
plt.show()
```

:::{admonition} What the plot shows
:class: note
The x-axis is the false positive rate (0 to 1) and the y-axis is the true
positive rate (0 to 1). The logistic-regression curve bows up toward the top-left
corner, well above the diagonal "no-skill" dashed line, indicating a good
classifier. The single number that summarizes this is the **area under the curve
(AUC)** — 0.5 is no skill, 1.0 is perfect. Print it directly with:
`from sklearn.metrics import roc_auc_score; print(roc_auc_score(y_binary_val, log_reg.predict_proba(X_val)[:, 1]))`.
:::

### Importance of different features for logistic regression

For logistic regression, the size of each coefficient indicates how strongly that
(standardized) feature pushes the prediction toward one class.

```python
coefficients = log_reg.coef_
coefficients.shape

x = plt.bar(featurenames, coefficients[0, :])
plt.ylabel("Coefficient values")
plt.xlabel("Feature")
plt.xticks(rotation=90)
plt.show()
```

:::{admonition} What the plot shows
:class: note
A bar chart with one bar per feature (39 features along the x-axis). Bars above
zero push the prediction toward one class; bars below zero push toward the other;
the taller the bar (in either direction) the more influential the feature. To
read the ranking as text, sort the coefficients:
`import pandas as pd; print(pd.Series(coefficients[0], index=featurenames).sort_values())`.
This prints the most negative to most positive coefficients with their feature
names.
:::

## Train a decision tree classifier

A decision tree splits the data by asking threshold questions about features. We
start with a shallow tree (`max_depth=2`) so it is easy to inspect:

```python
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_train, y_binary_train)
```

We can visualize the tree with graphviz (install once in the terminal with
`pip install graphviz`) and look at the thresholds it uses at each node:

```python
from graphviz import Source
from sklearn.tree import export_graphviz

export_graphviz(
    tree_clf,
    out_file="decision_tree.dot",
    feature_names=featurenames,
    class_names=classnames_binary,
    rounded=True,
    filled=True)

with open("decision_tree.dot") as f:
    dot_graph = f.read()

dot_graph = 'digraph Tree {\ndpi=50;\n' + dot_graph.split('\n', 1)[1]
Source(dot_graph)
```

:::{admonition} What the tree diagram shows (and how to read it as text)
:class: note
The diagram is a branching flowchart. The top (root) node states a rule like
"feature ≤ threshold"; samples for which the rule is true go left, and the rest
go right, down to the leaves, which give the predicted class. Instead of the
picture, print the same tree as indented text:
`from sklearn.tree import export_text; print(export_text(tree_clf, feature_names=list(featurenames)))`.
Each line shows the feature, the threshold, and — at the leaves — the predicted
class, so you get the full decision logic in a screen-reader-friendly form.
:::

Now train decision trees of increasing depth and see how they do on the
validation set:

```python
depths = [2, 10, 20, 50]
trained_decisiontrees = []
for i in depths:
    tree_clf = DecisionTreeClassifier(max_depth=i, random_state=42)
    trained_decisiontrees.append(tree_clf.fit(X_train, y_binary_train))
```

Compare their confusion matrices on the validation set. Instead of the four
heatmaps the notebook draws, print each normalized matrix:

```python
for depth, clf in zip(depths, trained_decisiontrees):
    cm = confusion_matrix(y_binary_val, clf.predict(X_val), normalize='true')
    print(f"max_depth={depth}  (rows=true [Anthropogenic, Natural], cols=predicted)")
    print(cm)
```

Deeper trees fit the training data more closely; the diagonal entries (correct
rate for each class) are what to watch. Compare the decision tree to logistic
regression with an ROC curve:

```python
ax = plt.gca()
RocCurveDisplay.from_estimator(log_reg, X_val, y_binary_val, ax=ax)
RocCurveDisplay.from_estimator(trained_decisiontrees[1], X_val, y_binary_val, ax=ax)
ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), linestyle='--')
plt.show()
```

:::{admonition} What the plot shows
:class: note
Two ROC curves on the same axes (false positive rate vs. true positive rate),
plus the diagonal no-skill line. One curve is logistic regression, the other is
the depth-10 decision tree. The curve that sits higher/closer to the top-left
corner is the better classifier. Print each AUC to compare them as numbers:
`roc_auc_score(y_binary_val, log_reg.predict_proba(X_val)[:, 1])` and
`roc_auc_score(y_binary_val, trained_decisiontrees[1].predict_proba(X_val)[:, 1])`.
:::

## Train a random forest classifier

A random forest is an ensemble of decision trees. Each tree is grown on a
different sub-sample of the data, and their ensemble vote is typically better
than any single tree. Random forests are powerful, still widely used in
environmental and climate research, and especially good on tabular data — but
they can be slow to train on large data sets.

```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf.fit(X_train, y_binary_train)

cm = confusion_matrix(y_binary_val, rnd_clf.predict(X_val), normalize='true')
print("Random forest (rows=true [Anthropogenic, Natural], cols=predicted)")
print(cm)
```

Compare all three models with an ROC curve:

```python
ax = plt.gca()
RocCurveDisplay.from_estimator(log_reg, X_val, y_binary_val, ax=ax)
RocCurveDisplay.from_estimator(trained_decisiontrees[1], X_val, y_binary_val, ax=ax)
RocCurveDisplay.from_estimator(rnd_clf, X_val, y_binary_val, ax=ax)
ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), linestyle='--')
plt.show()
```

:::{admonition} What the plot shows
:class: note
Three ROC curves plus the diagonal no-skill line: logistic regression, the
depth-10 decision tree, and the random forest. The random forest curve sits
highest (closest to the top-left corner), so in this case the random forest gives
some improvement over the other two. Compare the three AUCs numerically with
`roc_auc_score(...)` as shown above, using each model's `predict_proba`.
:::

### Feature importance

With random forests we can also get a sense of which features matter most:

```python
rnd_clf.feature_importances_

x = plt.bar(featurenames, rnd_clf.feature_importances_)
plt.ylabel("Feature Importance")
plt.xlabel("Feature")
plt.xticks(rotation=90)
plt.show()
```

:::{admonition} What the plot shows
:class: note
A bar chart with one bar per feature; taller bars are more important to the
forest's predictions (importances are all ≥ 0 and sum to 1). Read the ranking as
text by sorting:
`import pandas as pd; print(pd.Series(rnd_clf.feature_importances_, index=featurenames).sort_values(ascending=False))`.
This prints the most important features first with their scores.
:::

### Multiclass classification with the random forest

Now predict the full 12-class cause instead of the binary natural/anthropogenic
label. We use `class_weight="balanced"` to partly compensate for the class
imbalance:

```python
rnd_multiclass_clf = RandomForestClassifier(n_estimators=30, random_state=42, class_weight="balanced")

import time
start = time.time()
rnd_multiclass_clf.fit(X_train, y_multiclass_train)
end = time.time()
print(end - start)   # ~30 seconds

# Print the depth of each tree
for i, tree in enumerate(rnd_multiclass_clf.estimators_):
    print(f"Tree {i+1}: Depth = {tree.get_depth()}")
```

The 30 trees grow to depths of roughly 43–59 nodes. Training takes about half a
minute here.

Training can be slow, so we can save the trained model with `pickle` and reload
it later instead of retraining:

```python
import pickle

filename = 'rnd_multiclass_clf.pkl'
with open(filename, 'wb') as file:
    pickle.dump(rnd_multiclass_clf, file)

# Later, reload with:
loaded_model = pickle.load(open(filename, 'rb'))
```

Evaluate the multi-class classifier. The notebook draws a 12×12 confusion-matrix
heatmap; instead, print the **classification report**, which gives precision,
recall, and F1 for every class as a table of numbers:

```python
cm = confusion_matrix(y_multiclass_val, rnd_multiclass_clf.predict(X_val), normalize='true')
print(cm)   # 12x12, rows=true class, cols=predicted class, in the order of classnames_multi

from sklearn.metrics import classification_report
y_val_predicted = rnd_multiclass_clf.predict(X_val)
print(classification_report(y_val_predicted, y_multiclass_val, target_names=classnames_multi))
```

The report shows an overall accuracy of about **0.68**. The `Natural` class does
very well (precision 0.96, recall 0.80, F1 0.87) because it is large and
distinctive. The small classes do poorly — for example `Misuse of fire by a
minor` (F1 0.16), `Power generation/transmission/distribution` (F1 0.10), and
`Smoking` (F1 0.16). This is the imbalance problem: the classifier has few
examples of the rare causes to learn from.

### Handling class imbalance with SMOTE

One approach to the imbalance is to **over-sample** the under-represented classes
(install once with `pip install imbalanced-learn`):

```python
from imblearn.over_sampling import SMOTE
```

SMOTE (Synthetic Minority Over-sampling Technique) interpolates between points
within each class to create new synthetic examples similar to the training data,
augmenting the rare classes:

```python
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_multiclass_train)

X_resampled.shape                        # (1611324, 39)
X_resampled.shape[0] / X_train.shape[0]  # ~5.5x larger
```

This makes the data set about 5.5× larger. To keep training time manageable, we
randomly sample it back down to the original training-set size (now with balanced
classes):

```python
z = np.arange(0, X_resampled.shape[0])
idx = np.random.choice(z, size=X_train.shape[0], replace=False)
X_balanced = X_resampled[idx]
y_balanced = y_resampled[idx]

rnd_multiclass_clf2 = RandomForestClassifier(n_estimators=30, random_state=42, class_weight="balanced")
start = time.time()
rnd_multiclass_clf2.fit(X_balanced, y_balanced)
end = time.time()
print(end - start)
```

Evaluate the re-balanced classifier:

```python
cm = confusion_matrix(y_multiclass_val, rnd_multiclass_clf2.predict(X_val), normalize='true')
print(cm)

y_val_predicted_oversampled = rnd_multiclass_clf2.predict(X_val)
print(classification_report(y_val_predicted_oversampled, y_multiclass_val, target_names=classnames_multi))
```

Overall accuracy is about **0.63** — slightly lower than before — but the
trade-off is that several of the rare classes are now recalled somewhat better,
because SMOTE gave the model more (synthetic) examples of them. Whether this
trade-off is worthwhile depends on whether you care more about overall accuracy
or about correctly identifying the rare causes.

Finally, look at feature importance for the re-balanced model:

```python
x = plt.bar(featurenames, rnd_multiclass_clf2.feature_importances_)
plt.ylabel("Feature Importance")
plt.xlabel("Feature")
plt.xticks(rotation=90)
plt.show()
```

:::{admonition} What the plot shows
:class: note
A bar chart with one bar per feature; taller bars matter more to this model's
predictions. Read it as a sorted list with
`import pandas as pd; print(pd.Series(rnd_multiclass_clf2.feature_importances_, index=featurenames).sort_values(ascending=False))`.
:::

## Exercises

Try these short extensions of the tutorial. Each one reuses a method, model, or
library we covered above, applied to a different variable, split, or setting. Add
each in a new section of your script. For any plot, use a MAIDR render (or print a
sorted `pandas` Series / `value_counts()` / classification report) so you can read
the result.

1. **Class balance.** Using `value_counts()` on `data_known["NWCG_GENERAL_CAUSE"]`,
   print (or MAIDR-render as a bar plot) how many fires there are of each cause.
   Which three causes are the most common, and which is the rarest?

2. **A different tree depth.** The tutorial trained decision trees at depths 2, 10,
   20, and 50. Train one more `DecisionTreeClassifier` with `max_depth=5` on the
   binary target and print its `.score()` on the validation set. Where does its
   accuracy fall relative to the depth-2 and depth-10 trees?

3. **Metrics for the decision tree.** Using `precision_score`, `recall_score`, and
   `f1_score`, compute these three metrics for the depth-10 decision tree
   (`trained_decisiontrees[1]`) on the validation set, and compare them to the
   logistic-regression values from the tutorial.

4. **Number of trees.** Retrain the binary `RandomForestClassifier` twice, once
   with `n_estimators=30` and once with `n_estimators=200` (keep `random_state=42`).
   Print the validation accuracy of each. Does adding more trees keep improving the
   score?

5. **Top features.** The tutorial plotted `rnd_clf.feature_importances_` as a bar
   chart. Instead, build a `pandas` Series from `rnd_clf.feature_importances_`
   indexed by `featurenames`, sort it in descending order, and print the 5 most
   important features for the binary classifier.

6. **Confusion matrix on the test set.** The tutorial evaluated on the validation
   set. Print the confusion matrix (as numbers) for the binary random forest on the
   held-out **test** set (`X_test`, `y_binary_test`) instead. Are the results
   similar to the validation set?

7. **ROC on the test set.** Plot the ROC curve of the binary random forest
   (`rnd_clf`) on the test set with `RocCurveDisplay.from_estimator` (or print its
   AUC with `roc_auc_score`), and add the diagonal no-skill line as in the tutorial.

8. **Does class weighting matter?** The multiclass forest used
   `class_weight="balanced"`. Train a second multiclass `RandomForestClassifier`
   with the same settings but `class_weight=None`, and compare the two with
   `classification_report`. Which classes change the most?

9. **Predict the unknown fires.** Use the trained multiclass forest
   (`rnd_multiclass_clf`) to predict causes for the fires with unknown cause
   (`X_unknown`). Turn the predictions back into class names with `classnames_multi`
   and use `value_counts()` to show how many unknown fires are predicted for each
   cause.

10. **Save and reload a model.** Using `pickle`, save your binary random forest
    `rnd_clf` to a file, load it back into a new variable, and confirm that the
    reloaded model gives the same predictions on `X_val` as the original (e.g. with
    `np.array_equal`).
