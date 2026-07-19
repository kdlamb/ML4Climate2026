# Artificial Neural Networks (ANN): a hands-on tutorial

This hands-on tutorial accompanies the [Introduction to Artificial Neural
Networks](ann.md) notes. Neural networks can approximate any non-linear function.
Here we use them to build a **climate model emulator**: a fast surrogate that
predicts the climate a simulation would produce directly from the values of its
tunable parameters. We first fit an emulator with scikit-learn, then rebuild it
with Keras/TensorFlow, and finally reuse the same code on a second climate model.

We work with the **perturbed parameter ensembles (PPEs)** of Yang et al. (2024). A
PPE is a large batch of climate-model runs in which the uncertain internal
parameters (cloud, convection, and microphysics knobs) are varied together across
their plausible ranges. Each run is expensive, so a PPE is sparse and
high-dimensional, which is exactly where an emulator is particularly useful: once
trained, it maps *parameters to climate output* in microseconds.

> **Dataset & paper.** Yang, Q., Elsaesser, G. S., Van Lier-Walqui, M., &
> Eidhammer, T. (2024). *A simple emulator that enables interpretation of
> parameter-output relationships, applied to two climate model PPEs.* Journal of
> Advances in Modeling Earth Systems (JAMES).
> [doi:10.1029/2024MS004766](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004766)
> · [arXiv:2410.00931](https://arxiv.org/abs/2410.00931). Data:
> [Zenodo 14853175](https://zenodo.org/records/14853175). That paper introduces the
> SAGE emulator and *compares it against neural networks* on these two PPEs, so the
> ANN we build here is a direct baseline from the study.

:::{admonition} How to use this page (accessible version)
:class: important
On the standard site this is a Jupyter notebook. Here, run the code as a **Python
script**. See [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Part 1 (scikit-learn) runs anywhere. Parts 2 and 3 use **TensorFlow**, which you
install once with `pip install tensorflow`. The first run **downloads** the PPE data
(about 0.6 MB) from Zenodo. There are no images to view: results are shown as scatter
plots, which this page **describes** and backs with **printed numbers** ($R^2$
scores, shapes) you can read. Any plot can also be rendered with
[MAIDR](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/sci_python/trying_maidr.html).
The scores quoted below are from one real run; your numbers will be very close.
:::

## Part 0: Downloading the PPE data

The ensembles ship as CSVs inside a single archive on Zenodo. We download it once
into memory and read out the input (parameter) and output (climatology) tables for
each model.

```python
import io, zipfile, urllib.request
import pandas as pd

ZENODO_URL = "https://zenodo.org/api/records/14853175/files/yiqioyang/ppesage-v0.zip/content"

with urllib.request.urlopen(ZENODO_URL) as r:
    ppe_zip = zipfile.ZipFile(io.BytesIO(r.read()))

def load_ppe(model):
    """Return (inputs, outputs) DataFrames for 'cam6' or 'modele3'."""
    def read(kind):
        name = next(n for n in ppe_zip.namelist()
                    if n.endswith(f"data/{model}_ppe_{kind}.csv"))
        return pd.read_csv(ppe_zip.open(name))
    return read("input"), read("output")

cam6_X, cam6_y = load_ppe("cam6")
print("CAM6 parameters (inputs):", cam6_X.shape)
print("CAM6 climatologies (outputs):", cam6_y.shape)
```

**What the numbers show.** `CAM6 parameters (inputs): (262, 45)` and
`CAM6 climatologies (outputs): (262, 21)`, meaning 262 ensemble members, each a full
climate run with a different setting of 45 parameters, producing 21 global-mean
outputs.

The **outputs** are **emergent** climatologies: not knobs we set, but properties of
the simulated climate that the model predicts. The ones referred to below are:

- `netrad_toa`: net downward radiation at the top of the atmosphere (W m⁻²), the
  planet's overall energy balance.
- `albedo`: the fraction of incoming sunlight the planet reflects back to space (%).
- `sw_cre`, `lw_cre`: the shortwave and longwave *cloud radiative effects* (W m⁻²),
  meaning how much clouds cool the planet by reflecting sunlight and warm it by
  trapping outgoing heat.
- `olr`: outgoing longwave radiation (W m⁻²), the infrared energy the planet emits
  to space.
- `pwv`: precipitable water vapor, the total column water vapor.
- `prec`: the mean precipitation rate.

Instead of viewing the tables, print a readable summary of the target column we will
emulate:

```python
print(cam6_y["netrad_toa"].describe().round(2))
```

**What the numbers show.** Across the ensemble `netrad_toa` ranges from about **-24**
to **+34** W m⁻² with a mean near **+12**; that spread of about 58 W m⁻² is the
variance the emulator must learn to reproduce from the parameters.

## Part 1: Predicting a climate output with a scikit-learn MLP

Emulating one continuous climatology is a **regression** problem, so we use
`MLPRegressor` (not the `MLPClassifier` from the notes). Everything else is the
familiar setup: a `train_test_split` and a `make_pipeline` that standardizes the
features first. Standardizing is essential here, since the parameters span many
orders of magnitude (some about 10⁻⁶, some about 10⁹).

```python
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

target = "netrad_toa"
X, y = cam6_X, cam6_y[target]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42)

mlp_reg = MLPRegressor(hidden_layer_sizes=[64, 64], max_iter=5_000, random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)

print(f"Validation R^2: {pipeline.score(X_valid, y_valid):.3f}")
```

**What the number shows.** `.score()` for a regressor is the **coefficient of
determination** $R^2$, the fraction of the ensemble's `netrad_toa` variance the
emulator explains (1.0 is perfect). You should see roughly `Validation R^2: 0.67`.
With only about 200 training members over 45 inputs the data is sparse, so a moderate
$R^2$ is expected.

The standard site now draws a scatter plot of predicted vs. true `netrad_toa` on the
held-out test members. Read it as numbers instead:

```python
import numpy as np
y_pred = pipeline.predict(X_test)
print("true vs predicted netrad_toa (W m^-2):")
for t, p in zip(y_test.to_numpy(), y_pred):
    print(f"  true {t:7.2f}   predicted {p:7.2f}")
print(f"mean absolute error: {np.mean(np.abs(y_test.to_numpy() - y_pred)):.2f} W m^-2")
```

**What the plot shows.** Each held-out member is a point with true `netrad_toa` on
the x-axis and the emulated value on the y-axis; a dashed 1:1 line marks a perfect
emulator. Points cluster along that line. The printed pairs above are the same
information, and the mean absolute error (a few W m⁻²) is the typical vertical
distance from the line. Render the scatter through MAIDR to hear it.

## Part 2: Building the emulator with Keras/TensorFlow

For a production network we switch to **TensorFlow** with the **Keras** Sequential
API. One difference from a classification model: the target is a physical quantity
spanning tens of W m⁻², so we **standardize the target too** and train with a
**mean-squared-error** loss. We invert the target scaling before reporting, so
numbers stay in physical units.

```python
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train.to_numpy().reshape(-1, 1))
sx = lambda d: x_scaler.transform(d)
sy = lambda s: y_scaler.transform(s.to_numpy().reshape(-1, 1))

tf.keras.backend.clear_session()
tf.random.set_seed(42)

n_features = X_train.shape[1]
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[n_features]),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1),
])
model.summary()
```

**What `model.summary()` shows.** A text table of layers and parameter counts:
`Dense(64)` on 45 inputs is 45 times 64 plus 64, which is **2,944** parameters; the
next `Dense(64)` is 64 times 64 plus 64, which is **4,160**; the output `Dense(1)` is
64 plus 1, which is **65**; giving **7,169 trainable parameters** in total. The single
output unit with no activation is standard for regression.

```python
model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])

history = model.fit(sx(X_train), sy(y_train), epochs=200,
                    validation_data=(sx(X_valid), sy(y_valid)), verbose=0)
print(f"Trained for {len(history.epoch)} epochs.")
print("final training loss  :", round(history.history['loss'][-1], 3))
print("final validation loss:", round(history.history['val_loss'][-1], 3))
```

**What the plot and numbers show.** The standard site plots the training and
validation loss (and MAE) against epoch. The training loss falls almost to **0**,
while the **validation loss stays well above it** (a few tenths in standardized
units) and stops improving. That gap between a tiny training loss and a much larger
validation loss is a clear sign of **overfitting**: with only about 200 training
members and 45 inputs, the network has enough capacity to memorize quirks of the
training runs that do not generalize to unseen ones. On a PPE this small, some
overfitting is hard to avoid, and it is one reason the held-out $R^2$ is only
moderate. Render `history.history` (a dict of per-epoch lists) through MAIDR to hear
the two curves diverge.

Evaluate on the held-out test members, converting predictions back to W m⁻²:

```python
y_pred_keras = y_scaler.inverse_transform(model.predict(sx(X_test), verbose=0)).ravel()
ss_res = np.sum((y_test.to_numpy() - y_pred_keras) ** 2)
ss_tot = np.sum((y_test.to_numpy() - y_test.mean()) ** 2)
print(f"Keras test R^2: {1 - ss_res / ss_tot:.3f}")
```

**What the number shows.** A test $R^2$ in the same moderate range as the
scikit-learn emulator (roughly 0.4 to 0.7), which varies noticeably from run to run
because the network is small and the data sparse. The accompanying scatter plot of
predicted vs. true is read the same way as in Part 1, with points along a dashed 1:1
line.

## Using the emulator in practice: a new parameter setting

This is what makes the emulator useful. Suppose we propose a **new** set of the 45
parameters that was never run through the climate model. The emulator predicts the
resulting top-of-atmosphere radiation instantly, so we can screen thousands or
millions of candidate settings (to calibrate the model, or to map how output responds
to the parameters) without paying for a single expensive simulation. To imitate a
genuinely unseen setting, we draw one member at random from the held-out **test** set.

```python
rng = np.random.default_rng(0)
i = rng.integers(len(X_test))
new_params = X_test.iloc[[i]]   # a 1-row, 45-column "proposed" parameter set

pred = y_scaler.inverse_transform(model.predict(sx(new_params), verbose=0))[0, 0]
truth = y_test.iloc[i]
print(f"Emulator-predicted netrad_toa: {pred:6.2f} W m^-2")
print(f"Actual CAM6 netrad_toa:        {truth:6.2f} W m^-2")
```

**What the numbers show.** The emulator returns a `netrad_toa` estimate for a
parameter combination it never trained on, close to (but not exactly) the value the
full CAM6 run produced. In practice you would call this on millions of proposed
settings instead of running the climate model even once.

The standard site now plots a 9-by-5 grid of histograms, one per parameter, showing
how each of the 45 parameters was sampled across the 262 CAM6 members, with a **red
vertical line** marking the new selection's value in each panel. To read that as
numbers, print where the selection falls within each parameter's ensemble
distribution (its percentile, from 0 = below every member to 100 = above every
member):

```python
print("new selection's percentile within the ensemble, per parameter:")
for p in cam6_X.columns:
    pct = (cam6_X[p] < new_params[p].values[0]).mean() * 100
    print(f"  {p:28s} {pct:5.1f}th percentile")
```

**What the figure and table show.** Each red line (each percentile) says where the
proposed value sits inside the range the PPE actually sampled for that parameter. All
of them land somewhere between the ensemble minimum and maximum, because the proposal
was itself drawn from the PPE, so it is a plausible combination the emulator can be
trusted to interpolate. A percentile near 50 means a typical value; near 0 or 100
means an extreme one. Render any single parameter's histogram through MAIDR if you
want to hear the distribution.

## Part 3: The same workflow on a second model, ModelE3

The point of Yang et al. (2024) is that one emulator recipe should transfer across
*different* climate models. NASA-GISS's **ModelE3** PPE perturbs its own set of about
45 parameters over 751 members. We wrap the identical load, split, scale, and train
steps in a function so the reuse is explicit.

```python
modele_X, modele_y = load_ppe("modele3")
print("ModelE3 parameters:", modele_X.shape)
print("ModelE3 climatologies:", modele_y.shape)

def train_keras_emulator(X, y, epochs=200, seed=42):
    """Train a Keras MLP emulator for one climatology; return test R^2."""
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=0.1, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_full, y_tr_full, test_size=0.1, random_state=seed)

    xs = StandardScaler().fit(X_tr)
    ys = StandardScaler().fit(y_tr.to_numpy().reshape(-1, 1))
    gx = lambda d: xs.transform(d)
    gy = lambda s: ys.transform(s.to_numpy().reshape(-1, 1))

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    m = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[X.shape[1]]),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    m.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])
    m.fit(gx(X_tr), gy(y_tr), epochs=epochs,
          validation_data=(gx(X_va), gy(y_va)), verbose=0)

    y_pred = ys.inverse_transform(m.predict(gx(X_te), verbose=0)).ravel()
    ss_res = np.sum((y_te.to_numpy() - y_pred) ** 2)
    ss_tot = np.sum((y_te.to_numpy() - y_te.mean()) ** 2)
    return 1 - ss_res / ss_tot

r2_modele = train_keras_emulator(modele_X, modele_y["netrad_toa"])
print(f"ModelE3 netrad_toa test R^2: {r2_modele:.3f}")
```

**What the numbers show.** `ModelE3 parameters: (751, 45)` and `(751, 36)`
climatologies, a larger ensemble than CAM6. The same emulator code, with nothing
model-specific hard-coded, produces a comparable test $R^2$ (around 0.55 to 0.65).
The scatter plot is read exactly as before.

## Wrap-up

We built a neural-network **emulator** of two climate model PPEs, predicting
top-of-atmosphere net radiation from the model's tunable parameters: first with a
scikit-learn `MLPRegressor`, then with a Keras/TensorFlow MLP, then reusing the
identical pipeline on a second climate model.

- **Emulation is regression.** Moving from a classifier to an emulator is mostly
  swapping the last layer, the loss (MSE), and the metric ($R^2$).
- **Scaling matters more here.** Parameters span many orders of magnitude and the
  target is physical, so we standardize *both* inputs and target.
- **Data is sparse.** A few hundred expensive runs over about 45 parameters is a hard
  regime; the moderate $R^2$ and the training-versus-validation gap reflect that, and
  it is why Yang et al. (2024) argue for simpler, interpretable emulators (SAGE)
  alongside neural networks.

## Try it yourself

Use the code above as a starting point for a few short experiments:

1. **Emulate a different output.** Re-run the scikit-learn emulator with `target` set
   to `"sw_cre"` (shortwave cloud radiative effect) or `"albedo"` instead of
   `"netrad_toa"`. Which climatology is easiest for the MLP to predict (highest
   validation $R^2$), and which is hardest?
2. **Change the network size.** Try `hidden_layer_sizes=[16]`, `[64, 64]`, and
   `[128, 128, 128]` in the `MLPRegressor`. Does a bigger network always give a better
   validation $R^2$ on this small dataset? What does that tell you about overfitting?
3. **Train longer and watch the gap.** Increase the Keras `epochs` to 1000 and print
   the final training and validation loss again. Does the gap between them grow? What
   does a widening gap mean for how well the emulator will generalize to new parameter
   settings?
4. **Compare the two models.** Run `train_keras_emulator` on the CAM6 and ModelE3 PPEs
   for the same output (`netrad_toa`). Which PPE is easier to emulate, and can you
   think of a reason why (hint: compare the number of ensemble members)?
