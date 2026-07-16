# Supervised Regression

:::{admonition} How to work through these notes (accessible version)
:class: important
On the standard site this page is a Jupyter notebook. Here, the code is written to be run as
a **Python script** in VS Code — see [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Put the code into a file such as `regression.py`, adding the sections below one at a time, and
run it with `python regression.py`. Read printed output with the terminal's Accessible View.

**Plots.** Each figure has a **"What the plot shows"** description. Where a plot carries an
argument, the code also **prints the numbers** behind it, so you never need the picture to
follow the reasoning. To explore a plot yourself with sound and a text/braille readout, use
**MAIDR** — see [Accessible plots with
MAIDR](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/sci_python/trying_maidr.html).
:::

**Regression** predicts a *continuous* output from a set of input features. The setup mirrors
classification: a model learns a function $f$ mapping the feature matrix $X$ to predictions
$\hat{Y} = f(X)$, and we minimize the difference between $\hat{Y}$ and the ground-truth $Y$.
This page covers **multiple linear regression**, loss functions, over/under-fitting,
benchmarking with $R^2$, and **regularization** (Ridge and Lasso).

If you have not read the [Supervised Classification](classification.md) notes yet, start there
— the loss-and-gradient-descent machinery is the same, and this page assumes it.

## Multiple linear regression

The simplest regressor writes the prediction as a weighted linear combination of the features
plus a bias (intercept):

$$\hat{y} = b + w_1 x_1 + w_2 x_2 + \dots + w_m x_m = \mathbf{w}\cdot\mathbf{x} + b$$

Each weight $w_j$ is the (linear) importance of feature $j$: how much $\hat{y}$ changes per
unit change in $x_j$.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
x = np.linspace(0, 10, 60)
y = 2.5 * x + 4 + rng.normal(0, 3, x.size)     # true slope 2.5, intercept 4

lin = LinearRegression().fit(x.reshape(-1, 1), y)
print(f"learned slope = {lin.coef_[0]:.2f}, intercept = {lin.intercept_:.2f}")
print(f"true    slope = 2.50, intercept = 4.00")
```

:::{admonition} What the plot shows
:class: note
A scatter of 60 points marching steadily up and to the right from around $(0, 4)$ to around
$(10, 29)$, with a visible vertical scatter of a few units around the trend (that is the
`normal(0, 3)` noise). A red straight line runs through the middle of the cloud from
bottom-left to top-right — the fitted model.

The line clearly captures the trend, and the points sit above and below it in roughly equal
numbers. This is what a *well-specified* linear fit looks like: the data really was generated
by a straight line plus noise, so a straight line is the right model.

The printed numbers are the point of the figure: the model recovered a slope of **2.66**
against a true **2.50**, and an intercept of **3.41** against a true **4.00** — close, but not
exact. With only 60 noisy samples, the fit lands near the truth rather than on it.
:::

## Loss functions for regression

We measure error as the gap between truth $y_i$ and prediction $\hat{y}_i$:

- **Mean Squared Error (MSE, L2 loss)** — $\frac{1}{N}\sum (y_i-\hat{y}_i)^2$. Squares the
  error, so large mistakes are penalized heavily. Units are *squared*.
- **Mean Absolute Error (MAE, L1 loss)** — $\frac{1}{N}\sum |y_i-\hat{y}_i|$. Keeps the
  original units and is **less sensitive to outliers**.
- **Root Mean Squared Error (RMSE)** — $\sqrt{\text{MSE}}$. Popular because it preserves the
  units of the target.

```python
# What each loss charges you for an error of a given size.
print("  error : L2 penalty (squared) : L1 penalty (absolute) : ratio L2/L1")
for e in [0.25, 0.5, 1.0, 2.0, 3.0]:
    print(f"  {e:5.2f} : {e**2:19.3f} : {abs(e):21.3f} : {e**2/abs(e):11.2f}")
```

:::{admonition} What the plot shows
:class: note
Two curves on the same axes, penalty (vertical) against prediction error $y - \hat{y}$
(horizontal, −3 to 3). Both are symmetric about zero and both are zero at zero.

- **Squared error (L2)** is a **parabola**: flat-bottomed near zero, then curving upward
  increasingly steeply, reaching 9 at an error of ±3.
- **Absolute error (L1)** is a **V shape**: two straight lines meeting at a point at zero,
  reaching 3 at an error of ±3.

The two cross at an error of ±1, where both penalties equal 1. Inside that range the parabola
lies *below* the V; outside it, the parabola climbs far above.

The printed table is the same comparison as numbers. Read the last column: for a small error
of 0.25, L2 charges only a **quarter** of what L1 does — it barely cares. For a large error of
3.0, L2 charges **three times** as much. That crossover is the whole story: L2 shrugs at small
errors and panics at large ones, which is exactly why a single wild outlier can drag an
L2-fitted model toward itself, and why L1 is the more robust choice when outliers are expected.
:::

## Normal equation vs. gradient descent

Linear regression has a closed-form solution, the **normal equation**:

$$\mathbf{w} = (X^\top X)^{-1} X^\top \mathbf{y}$$

It's exact, but requires inverting an $m \times m$ matrix — prohibitive when there are many
features, and it needs $X^\top X$ to be invertible. **Gradient descent** avoids the inversion,
scales to large data sets that don't fit in memory, and is what we use in practice for big
problems.

## Polynomial regression, underfitting and overfitting

We can capture non-linear relationships by appending polynomial terms ($x^2, x^3, \dots$) to
the feature vector and *still* solving it as linear regression (scikit-learn's
`PolynomialFeatures` does the appending).

Model complexity has to be tuned:

- Too simple (a straight line through curved data) → **underfitting**: poor on both training
  *and* test data.
- Too complex (a very high-degree polynomial) → **overfitting**: it memorizes the training
  noise and generalizes badly.

Below we fit degree 1, 4, and 15 to the same noisy quadratic.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

rng = np.random.default_rng(1)
xs = np.sort(rng.uniform(-3, 3, 30))
ys = 0.5 * xs**2 - xs + 2 + rng.normal(0, 1.5, xs.size)   # true: a quadratic + noise
grid = np.linspace(-3, 3, 200)

print("degree :  R2 on the 30 points :  fitted curve's range over the plot")
for deg in [1, 4, 15]:
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(xs.reshape(-1, 1), ys)
    curve = model.predict(grid.reshape(-1, 1))
    r2 = r2_score(ys, model.predict(xs.reshape(-1, 1)))
    print(f"  {deg:2d}   : {r2:19.3f} :  [{curve.min():9.2f}, {curve.max():8.2f}]")
```

:::{admonition} What the plot shows
:class: note
Three panels side by side, each showing the same 30 scattered points (a noisy upward-curving
quadratic, higher at the left and right edges than in the middle) with a red fitted curve on
top. All three panels are clipped to a vertical range of −3 to 8.

- **"degree 1 (underfit)"** — a **straight line** sloping gently downward. It cannot bend, so
  it misses the curvature entirely: it sits above the data in the middle and below it at the
  right edge. This is underfitting made visible — the model is too rigid for the shape.
- **"degree 4 (good)"** — a smooth **U-shaped curve** that follows the bulk of the points,
  bending upward at both ends. It passes through the middle of the scatter without chasing
  individual points.
- **"degree 15 (overfit)"** — a violently **wiggling curve** that snakes up and down to pass
  near individual points, and shoots vertically off the top and bottom of the frame near the
  edges.

The printed table quantifies all three. Note the trap: $R^2$ on the training points **rises
monotonically** with degree (0.300 → 0.706 → 0.888), so by that number alone degree 15 looks
best. It is not — which is the entire point of the next section.

The curve-range column is where the overfitting becomes undeniable. Degree 1 stays within
[1.11, 5.67] and degree 4 within [1.24, 9.29] — both sensible for data that lives between
about 0 and 8. Degree 15 ranges from **−3687 to +409**. The panel's −3 to 8 limits are hiding
almost all of that: what you see as "shooting off the frame" is a curve heading for minus three
thousand. A model that predicts −3687 where the data says roughly 5 has not learned the
pattern; it has threaded a needle through noise.
:::

## Benchmarking with $R^2$

**Step 1 — pick a metric** (MSE, RMSE, MAE...).

**Step 2 — report the coefficient of determination $R^2$**, the fraction of the variance in
$y$ that the model explains:

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

$R^2 = 1$ is perfect; $R^2 = 0$ is no better than predicting the mean; $R^2 < 0$ means the
model does *worse* than the mean. (Note $R^2 \ne r$, the correlation coefficient.)

**Step 3 — report on training, validation, and test sets separately.** The **test** score
estimates *generalization* — how the model will do on new, unseen data. A high training score
with a low test score is the signature of overfitting.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# high-degree model to expose overfitting
X_ = xs.reshape(-1, 1)
Xtr, Xte, ytr, yte = train_test_split(X_, ys, test_size=0.4, random_state=0)
of = make_pipeline(PolynomialFeatures(15), LinearRegression()).fit(Xtr, ytr)

for name, Xs, ysub in [("train", Xtr, ytr), ("test", Xte, yte)]:
    pred = of.predict(Xs)
    print(f"{name:5s}  R^2 = {r2_score(ysub, pred):6.2f}   "
          f"RMSE = {np.sqrt(mean_squared_error(ysub, pred)):5.2f}")
```

This prints:

```
train  R^2 =   0.98   RMSE =  0.38
test   R^2 = -2055712.30   RMSE = 2244.23
```

This is the punchline of the whole section, and it is worth sitting with.

On the data it was **trained** on, the degree-15 model looks superb: it explains 98% of the
variance, with a typical error of 0.38 — on data whose noise has a standard deviation of 1.5.
That should already be suspicious: **the model is fitting more precisely than the noise floor
allows**. It is not learning the signal; it is memorizing the noise.

On **held-out** data the same model scores $R^2 = -2{,}055{,}712$. Recall that $R^2 = 0$ means
"no better than just predicting the mean" — so this model is roughly **two million times worse
than a horizontal line**. Its typical error is 2244, against data that ranges from about 0 to
8.

This is why step 3 is not optional. Judged only on training data, this is the best model on the
page. Judged the way that actually matters, it is catastrophically the worst. A single held-out
split is all it took to tell them apart.

## Regularization: Ridge and Lasso

Regularization adds a penalty on the size of the weights to the loss, discouraging the model
from fitting noise:

- **Ridge regression (L2)** — penalizes $\sum w_j^2$. Shrinks all coefficients toward zero but
  keeps every feature.
- **Lasso regression (L1)** — penalizes $\sum |w_j|$. Can drive some coefficients *exactly* to
  zero, performing implicit **feature selection**.

The plot below fits Lasso to a problem with many mostly-irrelevant features and increases the
penalty strength $\alpha$: watch the coefficients collapse to zero (sparse solution).

```python
from sklearn.linear_model import Lasso, Ridge
from sklearn.datasets import make_regression

# 10 features, but only 3 of them actually carry signal.
Xr, yr = make_regression(n_samples=100, n_features=10, n_informative=3,
                         noise=10, random_state=0)

print("Lasso: how many of the 10 coefficients survive as alpha grows?")
print("  alpha : nonzero coefs")
for a in [0.1, 0.5, 1.0, 3.0, 10.0, 31.6]:
    coef = Lasso(alpha=a, max_iter=10000).fit(Xr, yr).coef_
    print(f"  {a:5.1f} : {(np.abs(coef) > 1e-8).sum():2d}")

# The claim above says Ridge keeps every feature and Lasso zeroes some. Check it.
print("\nAt alpha = 10, nonzero coefficients out of 10:")
print(f"  Ridge (L2): {(np.abs(Ridge(alpha=10).fit(Xr, yr).coef_) > 1e-8).sum()}")
print(f"  Lasso (L1): {(np.abs(Lasso(alpha=10, max_iter=10000).fit(Xr, yr).coef_) > 1e-8).sum()}")
```

:::{admonition} What the plot shows
:class: note
Ten lines (one per coefficient) plotted against regularization strength $\alpha$ on a
logarithmic horizontal axis, with a black horizontal line marking zero.

On the **left** (weak penalty, $\alpha = 0.1$) the lines are spread out: three of them sit high
above zero — at roughly 70, 84 and 88 — and the other seven cluster in a band close to zero
(between about −2 and +2). Those three are the genuinely informative features; the other seven
are noise that the unregularized fit has given small non-zero weights to.

Moving **right** (stronger penalty), the seven small lines converge onto the zero line and
**stay there — exactly zero, not merely close**. The three large ones survive much longer but
are steadily dragged downward, so that by $\alpha \approx 31.6$ they have shrunk from
(70, 84, 88) to roughly (35, 52, 47).

The printed count tells the same story: 10 nonzero coefficients at $\alpha = 0.1$, down to
exactly **3** by $\alpha = 3.0$ — and 3 is precisely the number of informative features we
built into the data. Lasso found them without being told.

The second printed comparison verifies the distinction between the two methods: at the same
$\alpha = 10$, **Ridge keeps all 10** coefficients (merely shrunken) while **Lasso keeps 3**
(the rest driven exactly to zero). That is the difference between shrinkage and selection.
:::

Note the tension in the Lasso result. By $\alpha = 31.6$ it has the right *three* features, but
their coefficients have been shrunk to roughly half their true size — so the penalty that
performs the feature selection also **biases** the surviving estimates toward zero. Choosing
$\alpha$ is a trade-off, not a free win, and it is chosen the same way as any other
hyperparameter: on a validation set.

## Summary

This page covered **supervised regression** — predicting a *continuous* target from a set of
input features, using the same predict-then-minimize-a-loss recipe as classification.

- **Multiple linear regression** predicts a weighted sum of the features plus a bias,
  $\hat{y} = b + w_1 x_1 + \dots + w_m x_m = \mathbf{w}\cdot\mathbf{x} + b$. Each weight $w_j$
  is the linear importance of feature $j$: how much the prediction changes per unit change in
  that feature.
- **Polynomial features** ($x^2, x^3, \dots$) let the *same* linear machinery fit non-linear
  curves — you expand the feature vector and still solve a linear problem.
- We fit the weights by **minimizing a loss** that measures how far predictions fall from the
  truth:
  - **MSE / L2** squares the errors, so it punishes large mistakes hard and is sensitive to
    outliers; it has a clean closed-form solution.
  - **MAE / L1** takes absolute errors, so it is more robust to outliers.
  - **RMSE** is the square root of MSE, reported to keep the error in the original units of the
    target.
- Linear regression can be solved exactly with the **normal equation**
  $\mathbf{w} = (X^\top X)^{-1} X^\top \mathbf{y}$, or iteratively with **gradient descent**
  when there are too many features to invert the matrix or the data don't fit in memory.
- **Model complexity must be tuned** between two failure modes: too simple **underfits** (poor
  on both training and test data), too complex **overfits** (memorizes training noise and
  generalizes badly). Always benchmark with $R^2$ — the fraction of variance the model explains
  — on a held-out **test** set, never on the training data.
- **Regularization** adds a penalty on the size of the weights to fight overfitting:
  - **Ridge (L2)** penalizes $\sum w_j^2$, shrinking all coefficients toward zero but keeping
    every feature.
  - **Lasso (L1)** penalizes $\sum |w_j|$ and can drive coefficients *exactly* to zero,
    performing implicit **feature selection**.

Next: [Decision Trees and Random Forests](treesforests.md), which drop the straight-line
assumption entirely and partition the feature space into regions.
