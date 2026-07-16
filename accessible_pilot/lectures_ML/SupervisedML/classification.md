# Supervised Classification

:::{admonition} How to work through these notes (accessible version)
:class: important
On the standard site this page is a Jupyter notebook. Here, the code is written to be run
as a **Python script** in VS Code — see [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Put the code into a file such as `classification.py`, adding the sections below one at a
time, and run it with `python classification.py`. Read printed output with the terminal's
Accessible View.

**Plots.** Each figure has a **"What the plot shows"** description. Where a plot carries an
argument, the code also **prints the numbers** behind it, so you never need the picture to
follow the reasoning. To explore a plot yourself with sound and a text/braille readout, use
**MAIDR** — see [Accessible plots with
MAIDR](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/sci_python/trying_maidr.html).
The confusion matrix is a heatmap, so we also print it as plain numbers.
:::

**Classification** refers to when we are interested in predicting a *discrete* or
*categorical* label for each sample from a set of input features. This page covers several
core concepts related to classification that we discussed in the lecture:

- **logistic regression** (binary classification)
- **softmax regression** (multi-class classification)
- the loss functions used to train classifiers
- how we evaluate a trained classifier

## Binary, multi-class, multi-label

We first make the distinction between binary classification, multi-class classification, and
multi-label classification. In this lecture, we will focus on binary and multi-class
classification.

- **Binary classification** — assign one of *two* mutually exclusive classes.
  *Is there currently a tornado? (true / false)*
- **Multi-class classification** — assign one of *three or more* mutually exclusive classes.
  *Which cloud type is this? Which land-use category?*
- **Multi-label classification** — assign several *non*-exclusive labels to each sample (e.g.
  an image can contain *both* clouds *and* snow).

## Notation

For a data set of $N_s$ samples, each with $m$ features:

- $X$ — the feature matrix, shape $N_s \times m$. Each **row** is a sample, each **column**
  is a feature.
- $Y$ — the ground-truth labels.
- A classifier learns a function $f$ that maps features to predicted labels:
  $\hat{Y} = f(X|w)$. $w$ refers to the weights of the machine learning model (which we learn
  through training on the training data set).

## Logistic Regression & the sigmoid function

The simplest supervised machine learning model that we can use for binary classification
problems is logistic regression. **Logistic regression** takes a linear combination of the
features,

$$z = b + w_1 x_1 + w_2 x_2 + \dots + w_m x_m = \mathbf{w}\cdot\mathbf{x} + b$$

and squashes it through the **sigmoid** (logistic) function to get a probability between 0
and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

$p = \sigma(z)$ is the predicted probability of the positive ("red") class; $1-p$ is the
probability of the other ("blue") class. We assume a threshold of $0.5$ (i.e. $z=0$) to assign
the final label to any sample in our data set.

```python
import numpy as np
from sklearn.datasets import make_classification

# The same toy dataset used in the logistic-regression section below.
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, class_sep=0.7, random_state=0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(f"{X.shape[0]} samples, {X.shape[1]} features")
print(f"class balance: {(y==0).sum()} blue (y=0), {(y==1).sum()} red (y=1)")

# The sigmoid as numbers rather than a curve.
print("\n     z : sigma(z) = P(red)")
for z in [-8, -4, -2, -1, 0, 1, 2, 4, 8]:
    print(f"  {z:4d} : {sigmoid(z):.4f}")
```

:::{admonition} What the plot shows
:class: note
Two panels side by side.

**Left — "Two classes to separate"**: a scatter of 200 points in two features, roughly evenly
split (99 blue, 101 red). Red points (y=1) cluster toward one side, blue points (y=0) toward
the other, but the two groups **overlap substantially** through the middle, where red and blue
points are thoroughly mixed. No straight line separates them cleanly — this data set is
deliberately not easy, and the best a linear model manages is about 84% accuracy.

**Right — "Sigmoid maps z to a probability"**: a smooth S-shaped curve rising from 0 on the
left to 1 on the right, crossing 0.5 exactly at $z=0$, where a vertical dashed line marks the
threshold. A horizontal dashed line marks 0.5. Arrows label the upper region "red class" and
the lower region "blue class". The curve is steepest near $z=0$ and flattens toward both
ends.

The printed table above is that curve as numbers: note it passes through exactly 0.5 at
$z=0$, is symmetric ($\sigma(-z) = 1-\sigma(z)$), and saturates near 0 and 1 by about
$z=\pm 4$ — well before $z=\pm 8$. That saturation is why very confident predictions are hard
to move.
:::

## Binary cross-entropy loss

We train logistic regression by minimizing the **binary cross-entropy** (also known as the
log loss). For a single sample with true label $y \in \{0, 1\}$ and predicted probability $p$:

$$\mathcal{L} = -\big[\, y \log p + (1-y)\log(1-p) \,\big]$$

This loss is **small** when the model is confident and correct, and blows up when the model is
confident and *wrong* — that steep penalty is what makes binary cross entropy a good loss
function for logistic regression, as it is more useful than simply counting mistakes. To
determine the total loss of our classifier, we sum over the loss of all of the individual
samples in our training data set.

```python
# What the loss costs you, for a sample whose true label is y = 1.
print("  p (predicted P(y=1)) :  loss = -log(p)")
for p in [0.99, 0.9, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001]:
    print(f"           {p:6.3f}        : {-np.log(p):8.3f}")
```

:::{admonition} What the plot shows
:class: note
Two curves on the same axes, loss (vertical) against predicted probability $p$ (horizontal,
0 to 1).

- The curve for **true label y = 1** ($-\log p$) starts extremely high on the left (as
  $p \to 0$ it shoots toward infinity) and falls to 0 at $p = 1$.
- The curve for **true label y = 0** ($-\log(1-p)$) is its mirror image: 0 at $p = 0$, rising
  steeply toward infinity as $p \to 1$.

The two cross at $p = 0.5$. The shape is the whole point: the penalty is not linear. Being
confidently wrong is punished enormously more than being merely uncertain.

The printed table shows this asymmetry directly for the y=1 case: predicting 0.99 costs about
0.01, predicting 0.5 costs about 0.69, but predicting 0.001 costs about 6.9 — roughly **700
times** the penalty of the confident correct answer.
:::

## Logistic regression: a learned decision boundary

Because the boundary $z = \mathbf{w}\cdot\mathbf{x} + b = 0$ is linear, logistic regression
separates the classes with a straight line (a hyperplane in higher dimensions). The predicted
probability grades smoothly from one class to the other across the decision boundary.

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X, y)

# The boundary is a line. Here it is, explicitly.
(w1, w2), b = clf.coef_[0], clf.intercept_[0]
print(f"learned weights: w1 = {w1:.3f}, w2 = {w2:.3f}, b = {b:.3f}")
print(f"decision boundary:  {w1:.3f}*x1 + {w2:.3f}*x2 + {b:.3f} = 0")
print(f"  i.e.  x2 = {-w1/w2:.3f}*x1 + {-b/w2:.3f}")
print(f"\ntraining accuracy: {clf.score(X, y):.3f}")
```

:::{admonition} What the plot shows
:class: note
A single panel filled with a smooth color gradient running left-to-right, blue on the left
through pale shades to red on the right, with a colorbar labeled "P(class = 1)" running 0 to
1. A solid black line runs **almost vertically** down the panel — the decision boundary, where
the predicted probability is exactly 0.5. The 200 data points are drawn on top, colored by
their true class.

Three things to notice. First, the boundary is **straight** — that is the defining limitation
of logistic regression. Second, it is near-vertical rather than slanted, and the printed
weights say why: $w_1 = 2.49$ is about **15 times** larger than $w_2 = 0.16$, so the model has
learned that feature 1 carries nearly all the information and feature 2 very little. The
boundary is therefore roughly "feature 1 is above about 0.11", almost regardless of feature 2.
Third, the color does not flip abruptly at the line; it **grades** smoothly from red to blue
over a band around it. Points far from the boundary sit in saturated red or blue (confident
predictions); points near it sit in pale, washed-out color (the model is unsure). **33 of the
200 points sit on the wrong side of the line** — the 83.5% training accuracy printed above. In
the overlap zone the two classes are genuinely mixed, and no straight line can do better.

The printed equation gives you that black line exactly, as a formula you can evaluate.
:::

:::{admonition} Reading the boundary equation
:class: tip
The printed form `x2 = -15.463*x1 + 1.655` looks alarming — a slope of −15! But that is just
what a near-vertical line looks like when you force it into "$y = mx + c$" form: a very large
slope means the line barely moves horizontally as you travel up it. The other printed form,
`2.491*x1 + 0.161*x2 - 0.267 = 0`, describes the same line without the distortion, which is
why the general form is the safer one to reason from.
:::

## Softmax regression for multi-class problems

**Softmax regression** generalizes logistic regression to $K$ mutually exclusive classes. Each
class $k$ gets its own score $z_k = \mathbf{w}_k\cdot\mathbf{x} + b_k$, and the **softmax**
function turns the scores into a probability distribution that sums to 1:

$$p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

The predicted class is the one with the highest probability.

Some environmental applications of multi-class classification include land-cover or land usage
classification using satellite images, cloud-type classification, ice-crystal morphology
classification, and aurora classification.

Below we fit softmax regression to the three-class Iris data set from the `scikit-learn`
library (using two features so we can draw the decision boundaries). The Iris data set consists
of 3 different classes of irises (Setosa, Versicolor, and Virginica), and measurements of their
sepal length, sepal width, petal length and petal width, stored in a 150x4 `numpy.ndarray`.

```python
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

iris = load_iris()
Xi = iris.data[:, 2:4]      # petal length, petal width
yi = iris.target
soft = LogisticRegression(max_iter=1000).fit(Xi, yi)

print("classes:", list(iris.target_names))
print(f"training accuracy: {soft.score(Xi, yi):.3f}")

# One row of weights per class — that is what "one score per class" means.
print("\nshape of the weight matrix (one row per class):", soft.coef_.shape)

# The softmax probabilities for one example sum to 1.
probs = soft.predict_proba(Xi[[0, 70, 140]])
print("\nsoftmax probabilities for three example flowers:")
for name, row in zip(["a setosa", "a versicolor", "a virginica"], probs):
    print(f"  {name:14s}: " + ", ".join(f"{n}={p:.3f}" for n, p in zip(iris.target_names, row))
          + f"   (sums to {row.sum():.3f})")

print("\nconfusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(yi, soft.predict(Xi)))
```

:::{admonition} What the plot shows
:class: note
A scatter of the 150 iris flowers, petal width (vertical) against petal length (horizontal),
with the background divided into **three colored regions** — one per class — by the fitted
model. The three species are drawn as differently colored point groups.

*Setosa* occupies a small, tight group in the bottom-left, separated from the rest by a wide
empty gap, and its region is cleanly its own. *Versicolor* and *virginica* sit adjacent in the
upper right and **touch along a boundary**, with a few points from each straying across into
the other's region.

Unlike the binary case, the regions here are bounded by several straight boundary segments
meeting at junctions — one region per class, each still delimited by linear boundaries.

The printed confusion matrix says the same thing precisely: setosa is perfect (50 of 50),
while versicolor and virginica each lose a couple of flowers to the other — 3 and 2
respectively, for 96.7% accuracy. The overlap you see is real, not a drawing artifact.
:::

Notice that every row of `predict_proba` sums to 1. That is the softmax doing its job: the
scores become a genuine probability distribution over the mutually exclusive classes.

Look closely at the middle flower. It is a real *versicolor*, but the model gives **virginica
0.545 against versicolor 0.454** — so it gets this one **wrong**, and only just. That is not a
typo in the example; it is one of the 3 versicolors in the confusion matrix's off-diagonal, and
it shows you what that number looks like from the inside. A misclassification here is not a
wild error but a coin-flip between two genuinely overlapping classes. Compare it with the
setosa above (0.980 confident) and the virginica below (0.983 confident): the model is sure
about the easy cases and hesitant exactly where the species overlap.

## Training: gradient descent

Both models minimize their loss with **gradient descent** — iteratively nudging the weights
$\mathbf{w}$ in the direction that reduces the loss. In practice we use **mini-batch stochastic
gradient descent**: we update the weights using small random batches of samples, which is more
memory-efficient. The logistic and softmax losses are convex (they have a single global
minimum), so here the stochastic noise matters less than it does for non-convex models such as
neural networks.

- **Batch GD** — use all samples for each update (accurate, slow).
- **Stochastic GD** — one sample per update (noisy, fast).
- **Mini-batch GD** — a small batch per update (the practical middle ground).

```python
# Gradient descent on the binary cross-entropy loss surface.
# We reuse the two-feature dataset (X, y) from the logistic-regression section.
Xs = (X - X.mean(0)) / X.std(0)   # standardize so a bias-free model works well

def bce(w):
    p = sigmoid(Xs @ w)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def grad(w):
    return Xs.T @ (sigmoid(Xs @ w) - y) / len(y)

w = np.array([4.0, -4.0])        # deliberately poor starting point
lr, path = 0.5, [w.copy()]
for _ in range(50):
    w = w - lr * grad(w)
    path.append(w.copy())
path = np.array(path)
losses = [bce(wi) for wi in path]

print(f"start:  w = [{path[0][0]:6.3f}, {path[0][1]:6.3f}]   loss = {losses[0]:.4f}")
for i in [1, 2, 5, 10, 20, 50]:
    print(f"step {i:2d}: w = [{path[i][0]:6.3f}, {path[i][1]:6.3f}]   loss = {losses[i]:.4f}")

# Is the loss decreasing at every single step, as claimed?
drops = np.diff(losses)
print(f"\nsteps where the loss went UP: {(drops > 0).sum()} of {len(drops)}")
```

:::{admonition} What the plot shows
:class: note
Two panels.

**Left — "Loss surface + gradient-descent path"**: a filled contour map of the binary
cross-entropy loss over the two weights $w_1$ (horizontal) and $w_2$ (vertical), each from −5
to 5. The color runs from dark (low loss) to bright (high loss), forming a smooth bowl with a
single basin — no secondary dips, because the loss is **convex**. A red line of connected dots
traces the path of gradient descent from a white square marker at the start (in a high-loss
region) down into the basin, ending at a white star that the legend labels "end (minimum)".
The steps are **large at first** — where the surface is steep — and get **shorter** as the path
approaches the bottom and the gradient flattens: the printed trace shows the weights moving
about 0.14 per step at the start but only about 0.03 per step by the end, and the loss falling
from 1.03 to 0.40, most of it in the first ten steps.

**Right — "Loss decreases each step"**: loss against iteration number, a red curve dropping
steeply over the first few iterations and then flattening into a long, nearly horizontal tail.

The printed trace above is the same path as numbers: watch the weights move and the loss fall,
fast at first and then barely at all. The final check confirms the claim in the panel title —
the loss decreases at *every* step, never once increasing.
:::

## Evaluating a classifier

### Confusion matrix

One way that we can evaluate how well a trained classifier is doing is by looking at its
**confusion matrix**. The **confusion matrix** counts true positives (TP), true negatives (TN),
false positives (FP), and false negatives (FN) associated with each class. It reveals *which*
classes get confused.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
clf = LogisticRegression().fit(Xtr, ytr)

print("confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(yte, clf.predict(Xte)))
print()
print(classification_report(yte, clf.predict(Xte)))
```

:::{admonition} What the plot shows
:class: note
A 2x2 grid of squares shaded blue by count, with the number printed in each cell. Rows are the
true label, columns the predicted label. The **diagonal** (top-left, bottom-right) holds the
correct predictions and is darkly shaded; the **off-diagonal** cells hold the mistakes and are
palely shaded.

For this data set the layout is:

```
              predicted 0   predicted 1
   true 0          30             8      <- 8 false positives
   true 1           2            20      <- 2 false negatives
```

The two darkest cells are the diagonal (30 and 20 correct). The **8** in the top-right is the
interesting one: those are blue points the model called red — **false positives**. The **2** in
the bottom-left are red points it called blue — **false negatives**.

This asymmetry is the whole reason the confusion matrix is worth looking at. Accuracy alone
would tell you "83%" and stop; the matrix tells you the model's mistakes are **four times more
often false alarms than misses**. Whether that is good or bad depends entirely on your problem —
which is the subject of the next section.
:::

### Accuracy, precision, recall, F1

Some other metrics that can be used to describe how well a classifier fits the data are defined
below:

$$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}
\qquad
\text{Precision} = \frac{TP}{TP+FP}
\qquad
\text{Recall} = \frac{TP}{TP+FN}$$

- **Precision** — of everything we *predicted* positive, how much really is? (Penalizes false
  positives.)
- **Recall (sensitivity)** — of all the *actual* positives, how many did we catch? (Penalizes
  false negatives.)
- **F1 score** — the harmonic mean of precision and recall, useful when both error types matter
  or the classes are imbalanced:
  $F_1 = 2\cdot\frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}$

### Precision vs. recall — a domain choice

The threshold you pick in defining how to separate classes can trade off between precision and
recall. If we want to correctly identify more positive cases (in order to increase recall), our
model will become less selective, leading to more false alarms (decreasing precision). Depending
on what problem we are trying to solve, we may favor a classifier with either high recall or
high precision.

- **Favor high recall** when a *miss* is costly. For example, if we are interested in forecasting
  dangerous storms or a flash flood, a few false alarms are acceptable if we catch (almost) every
  real case.
- **Favor high precision** when a *false alarm* is costly. Mapping mountain permafrost, or
  drilling for oil — if the model says "positive," we want it to really be there before taking an
  action that might be costly.

You can see this trade-off directly by moving the threshold away from its 0.5 default:

```python
from sklearn.metrics import precision_score, recall_score

probs_te = clf.predict_proba(Xte)[:, 1]
print("threshold : precision : recall")
for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
    pred = (probs_te >= t).astype(int)
    pr = precision_score(yte, pred, zero_division=0)
    rc = recall_score(yte, pred, zero_division=0)
    print(f"   {t:.1f}    :   {pr:.3f}   : {rc:.3f}")
```

The trade-off is stark, and runs in exactly one direction. At a threshold of **0.1** the model
catches **every** real positive (recall 1.000) but less than half of what it flags is real
(precision 0.468) — it is crying wolf constantly. At **0.9** everything it flags is genuinely
positive (precision 1.000) but it only catches a third of them (recall 0.318) — it misses most
of the real cases. The default 0.5 sits in between at precision 0.714, recall 0.909.

Lowering the threshold makes the model say "positive" more readily — catching more real
positives (recall up) at the cost of more false alarms (precision down). Raising it does the
reverse. **No threshold gives you both**, and nothing in the data tells you which end to prefer:
that is a judgment about the cost of each error type in your problem, not a question the model
can answer.

### ROC curve and AUC

The **Receiver Operating Characteristic (ROC)** curve sweeps the classification threshold and
plots the true-positive rate against the false-positive rate. A perfect classifier hugs the
top-left corner; the diagonal is a "no-skill" random guess. The **area under the curve (AUC)**
summarizes overall performance in a single number (0.5 = random, 1.0 = perfect). The curves
below show several classifiers of increasing skill, all falling between the no-skill diagonal
and the perfect corner.

```python
from sklearn.metrics import roc_curve, auc

# The trained model's own scores on the test set, progressively blended with random
# noise to simulate weaker classifiers.
scores = clf.predict_proba(Xte)[:, 1]
rng = np.random.default_rng(0)
noise = rng.random(len(scores))

for w_, label in [(1.0, "strong"), (0.4, "moderate"), (0.2, "weak")]:
    blended = w_ * scores + (1 - w_) * noise
    fpr, tpr, _ = roc_curve(yte, blended)
    print(f"{label:8s}: AUC = {auc(fpr, tpr):.2f}")
print(f"{'no skill':8s}: AUC = 0.50")
```

:::{admonition} What the plot shows
:class: note
A square panel with false-positive rate on the horizontal axis and true-positive rate on the
vertical, both 0 to 1. Four lines:

- **"strong" (AUC = 0.97)** — bows sharply up toward the top-left corner, rising almost to the
  top before bending right. It comes close to the corner but does not quite touch it.
- **"moderate" (AUC = 0.84)** — a visibly shallower bow, well inside the strong curve.
- **"weak" (AUC = 0.69)** — shallower still, hugging fairly close to the diagonal.
- **"no skill" (AUC = 0.50)** — a straight dashed diagonal from bottom-left to top-right.

The ordering is the message: the **further a curve bows toward the top-left corner, the better
the classifier**, and the AUC — the area underneath it — captures that in one number. The four
curves nest inside one another, each lying between the diagonal and the top-left corner.

Note that even our best model reaches only 0.97, not 1.00 — on data with genuinely overlapping
classes, no threshold catches every positive without any false alarms. A curve that did reach
the corner would be suspicious.

The printed AUCs above give you the same ranking without the picture.
:::

Note how these curves are constructed: only **"strong"** is a real classifier — the logistic
regression we just trained. "Moderate" and "weak" are made by blending its scores with random
noise, to simulate weaker models on the same data.

## Summary

This page covered the core building blocks of **supervised classification** — predicting a
discrete label from a set of input features.

- **Logistic regression** computes a linear score $z = \mathbf{w}\cdot\mathbf{x} + b$ and passes
  it through the **sigmoid** to get $p = \sigma(z)$, the probability of the positive class.
  Comparing $p$ to a **threshold** (0.5 by default, i.e. $z = 0$) produces the final label. The
  decision boundary is *linear*: a straight line in 2-D, a hyperplane in higher dimensions.
- We train it by **minimizing the binary cross-entropy loss**, which rewards confident correct
  predictions and heavily penalizes confident wrong ones.
- **Softmax (multinomial) regression** generalizes logistic regression to $K$ mutually exclusive
  classes: it produces one score per class and normalizes them with the softmax so the
  probabilities are non-negative and sum to 1. The predicted class is the one with the highest
  probability.
- Both models are fit by **gradient descent** (batch, stochastic, or mini-batch). Because their
  losses are convex, gradient descent reliably finds the single global minimum.
- **Evaluating a classifier** goes well beyond simply evaluating its accuracy, which can be
  misleading on imbalanced data:
  - The **confusion matrix** breaks predictions into true/false positives and negatives, showing
    *which* classes get confused.
  - **Precision** ("of the positives I predicted, how many were right?") and **recall** ("of the
    actual positives, how many did I catch?") trade off against each other; the **F1 score** is
    their harmonic mean.
  - The **ROC curve** sweeps the threshold and plots true-positive vs. false-positive rate; the
    **AUC** summarizes it in a single threshold-free number (0.5 = random, 1.0 = perfect).
- Which metric matters depends on the **cost of each error type** in your problem. A missed
  wildfire detection and a false alarm are not equally expensive, so choose the operating
  threshold accordingly rather than defaulting to 0.5.
