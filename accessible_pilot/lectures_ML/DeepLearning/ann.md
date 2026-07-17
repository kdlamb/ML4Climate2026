# Introduction to Artificial Neural Networks

Deep learning is where much of the recent innovation in machine learning for
climate science and environmental sustainability is happening. In this section we
will:

- Define the **artificial neural network** (ANN) and see how it generalizes the
  linear models we've already met.
- Understand **activation functions** and where the non-linearity of a neural
  network comes from.
- Walk through the steps and best practices to **build a neural network in
  practice**.
- Know the main **hyperparameters** that define a network's architecture and the
  ones that control its training.
- Discuss applications of neural networks to **climate model parameterizations and
  tuning**.

:::{admonition} How to use this page (accessible version)
:class: important
On the standard site this is a Jupyter notebook whose figures are generated from
data. Here, the same code is given as blocks you can run as a **single Python
script** — see [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Each figure is followed by a **"What the plot shows"** description and the key
**printed numbers**, so you can follow the argument from the numbers rather than the
picture. Any plot can also be rendered through
[MAIDR](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/sci_python/trying_maidr.html)
to explore it by sound. Formulas are written in words: "phi of (W times x plus b)"
means the function `phi` applied to `W` times `x` plus `b`.
:::

Put these imports at the top of your script; every code block below builds on them.

```python
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import r2_score, log_loss

warnings.filterwarnings("ignore")
np.random.seed(0)
```

## From biological to artificial neural networks

The conceptual roots of the neural network were recognized by the **2024 Nobel
Prize in Physics**, awarded to John Hopfield and Geoffrey Hinton. Hopfield
introduced the **Hopfield network** (1982), a simple recurrent network that can
store and recall patterns. Hinton and collaborators introduced the **Boltzmann
machine**, a stochastic neural network in which nodes make probabilistic decisions
about being on or off. Both led to what we now call the artificial neural network.

The name comes from a loose analogy with the brain. A biological neuron accumulates
a membrane potential and, once it crosses a threshold, "fires." An artificial neural
network mimics this with **activation functions** that introduce non-linearities
between layers — enough to learn essentially any non-linear function.

## The perceptron and the fully connected layer

The simplest neural network is the **perceptron**: take some inputs, multiply them by
weights, and apply a step-function activation (negative values map to 0, positive
values map to 1), producing a single output.

Connecting the inputs to a whole layer of outputs gives a **fully connected layer**.
Written out, a layer looks like linear regression: the **output** equals the
**activation function phi** applied to (**weight matrix W** times **input vector x**,
plus **bias vector b**). In symbols: output = phi(W x + b). The only difference from
linear regression is the non-linear `phi` applied to the weighted sum.

## From a layer to a Multilayer Perceptron

Stacking fully connected layers gives a **Multilayer Perceptron (MLP)**. With one
hidden layer:

- **Hidden layer** = phi_bottom applied to (W_bottom times Input, plus b_bottom).
- **Output** = phi_top applied to (W_top times Hidden, plus b_top).

Depth rule of thumb: 1 layer is roughly logistic regression, 2 layers is a *shallow*
network, 3 or more is *deep* learning. The hidden layers hold **hidden variables** —
learned features that are hard to interpret directly, which is why networks are often
called **black boxes**. **Explainable-AI (XAI)** methods probe why a network produces
its mapping, but interpretability remains a trade-off for the flexibility.

## Activation functions — where the non-linearity arises

Activation functions are where the non-linearity comes from. Without them, a stack of
linear layers — even hundreds — collapses to a single linear transformation.

```python
z = np.linspace(-4, 4, 400)

def relu(z):         return np.maximum(0.0, z)
def leaky(z, a=0.1): return np.where(z > 0, z, a * z)
def elu(z, a=1.0):   return np.where(z > 0, z, a * (np.exp(z) - 1))
sigmoid = 1.0 / (1.0 + np.exp(-z))
tanh    = np.tanh(z)

funcs = {"ReLU": relu(z), "Leaky ReLU": leaky(z), "ELU": elu(z),
         "sigmoid": sigmoid, "tanh": tanh}

# one row per activation: function on the left, its derivative on the right
fig, axes = plt.subplots(len(funcs), 2, figsize=(9, 2.3 * len(funcs)))
for row, (name, yv) in enumerate(funcs.items()):
    axL, axR = axes[row]
    axL.plot(z, yv, color="C0")
    axR.plot(z, np.gradient(yv, z), color="C1")
    axL.set_ylabel(name)
    for ax in (axL, axR):
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    if row == 0:
        axL.set_title("Activation phi(z)"); axR.set_title("Derivative phi'(z)")
    if row == len(funcs) - 1:
        axL.set_xlabel("z"); axR.set_xlabel("z")
plt.tight_layout(); plt.show()
```

**What the plot shows.** A grid with **one row per activation function** — the function
itself on the left and its derivative on the right, over inputs from -4 to 4:

- **ReLU**: flat at 0 for negative inputs, then rising as a straight line (a sharp
  "kink" at 0). Its derivative is a **step** — exactly 0 for negatives, exactly 1 for
  positives.
- **Leaky ReLU**: like ReLU but sloping gently downward instead of flat for negatives;
  its derivative is a small positive constant for negatives and 1 for positives.
- **ELU**: curves smoothly down toward about -1 for very negative inputs; its
  derivative is smooth (no hard step).
- **sigmoid**: rises smoothly from 0 to 1; its derivative is a smooth bump that peaks
  at 0 and falls to nearly 0 at both edges.
- **tanh**: rises smoothly from -1 to 1; its derivative is a taller smooth bump,
  peaking at 1 at the center.

The smooth derivatives (ELU, sigmoid, tanh) help gradient flow; ReLU's bounded (0 or 1)
derivative and sigmoid/tanh's ability to squash outputs into a fixed range are each
useful in different situations.

## Neural networks as universal approximators

Neural networks are **universal function approximators**: by making a network **wide
enough** or **deep enough**, it can represent any continuous function (Cybenko &
Hornik, 1989; Leshno et al., 1993). A neural network can approximate genuinely
non-linear functions. Below we fit a linear regression and an MLP to the same
non-linear, noisy target and compare their R^2 (higher is better).

```python
X = np.linspace(-3, 3, 300).reshape(-1, 1)
y = np.sin(1.5 * X).ravel() + 0.3 * X.ravel() + 0.1 * np.random.randn(X.shape[0])

Xs = StandardScaler().fit_transform(X)
linear = LinearRegression().fit(X, y)
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu",
                   max_iter=4000, random_state=0).fit(Xs, y)

print(f"Linear regression R^2 : {r2_score(y, linear.predict(X)):.3f}")
print(f"MLP (64, 64)      R^2 : {r2_score(y, mlp.predict(Xs)):.3f}")

plt.figure(figsize=(7, 4.5))
plt.scatter(X, y, s=8, alpha=0.35, label="data")
plt.plot(X, linear.predict(X), lw=2, label="linear regression")
plt.plot(X, mlp.predict(Xs), lw=2, label="MLP (64, 64)")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.tight_layout(); plt.show()
```

**What the plot shows.** The scattered data follows a wavy, S-through-rising curve. The
**linear regression** is a single straight line that cannot follow the waves. The
**MLP** traces the curve closely. The printed scores make it concrete:

```
Linear regression R^2 : 0.349
MLP (64, 64)      R^2 : 0.984
```

The straight line explains only about 35% of the variance; the neural network explains
about 98%.

### Wide enough to approximate

"Wide enough" is also an important point. The same target is fit by single-hidden-layer
MLPs of increasing width.

```python
for w in [1, 4, 16]:
    m = MLPRegressor(hidden_layer_sizes=(w,), activation="relu", solver="lbfgs",
                     max_iter=8000, random_state=0).fit(Xs, y)
    print(f"hidden width {w:2d}: R^2 = {r2_score(y, m.predict(Xs)):.3f}")
```

**What the numbers show.** R^2 rises with width as the network gains capacity to bend
the line:

```
hidden width  1: R^2 = 0.434
hidden width  4: R^2 = 0.515
hidden width 16: R^2 = 0.988
```

One hidden unit fits poorly; sixteen units track the curve almost perfectly.

## Learning the weights: backpropagation

The learnable parameters are the **weight matrices** and **bias vectors** of every
layer — a deep network can have hundreds of thousands of them. Training uses **backpropagation**: a
**forward pass** produces predictions, a **loss** compares them to the targets, and a
**backward pass** computes the **gradient of the loss** with respect to every weight
(via **automatic differentiation**, the chain rule applied over the computational
graph) and steps the weights to reduce the loss.

A key difficulty is the **exploding / vanishing gradient problem**: repeated
multiplication through the chain rule can make weights unstable (loss diverges) or stall
updates. Remedies include gradient clipping, careful weight initialization, smaller
learning rates, batch normalization, residual connections, and the choice of activation
function.

### The loss landscape and the learning rate

The **learning rate** sets how far gradient descent moves at each step. Fitting a line
`y = w*x + b` by minimizing mean squared error gives a bowl-shaped **loss surface** over
`(w, b)`; gradient descent walks downhill. Three runs start from the same point with
different learning rates.

```python
xd = np.random.randn(80)
yd = 2.0 * xd + 1.0 + 0.3 * np.random.randn(80)

def mse(w, b):  return np.mean((w * xd + b - yd) ** 2)
def grad(w, b):
    p = w * xd + b
    return 2 * np.mean((p - yd) * xd), 2 * np.mean(p - yd)

ws = np.linspace(-1, 5, 200); bs = np.linspace(-2, 4, 200)
WW, BB = np.meshgrid(ws, bs)
ZZ = np.array([[mse(w, b) for w in ws] for b in bs])

def descend(lr, steps=40, start=(-0.5, -1.5)):
    w, b = start; path = [(w, b)]; loss = [mse(w, b)]
    for _ in range(steps):
        gw, gb = grad(w, b); w -= lr * gw; b -= lr * gb
        path.append((w, b)); loss.append(mse(w, b))
    return np.array(path), np.array(loss)

rates = {"too small (0.02)": 0.02, "well chosen (0.3)": 0.3, "too large (1.02)": 1.02}
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
cf = axL.contourf(WW, BB, ZZ, levels=30, cmap="viridis")
fig.colorbar(cf, ax=axL, label="MSE loss")
for label, lr in rates.items():
    path, loss = descend(lr)
    axL.plot(path[:, 0], path[:, 1], marker="o", ms=3, lw=1.3, label=label)
    axR.plot(loss, marker="o", ms=3, lw=1.3, label=label)
    print(f"{label:20s} -> final (w, b) = ({path[-1,0]:8.2f}, {path[-1,1]:8.2f}), "
          f"final loss = {loss[-1]:.2e}")
print("(true parameters: w = 2.0, b = 1.0)")
axL.scatter([2.0], [1.0], c="red", marker="*", s=220, label="minimum")
axL.set(xlabel="w (slope)", ylabel="b (intercept)", xlim=(-1, 5), ylim=(-2, 4),
        title="Loss surface with gradient-descent paths")
axL.legend(fontsize=8)
axR.set(xlabel="iteration", ylabel="MSE loss (log scale)", yscale="log",
        title="Loss vs. iteration")
axR.legend(fontsize=8)
plt.tight_layout(); plt.show()
```

**What the plot shows.** Two panels. **Left**, the loss surface: a filled colored map
over the two parameters `(w, b)`, a bowl whose lowest (darkest) point is the true
`(w, b) = (2, 1)`, marked with a star. Three descent paths start from the same
lower-left corner. **Right**, the same three runs as **loss versus iteration** on a log
scale. The printed endpoints and final losses tell the story:

```
too small (0.02)     -> final (w, b) = (    1.16,     0.30), final loss = 9.91e-01
well chosen (0.3)    -> final (w, b) = (    2.06,     0.99), final loss = 8.70e-02
too large (1.02)     -> final (w, b) = ( 3920.90, -6692.61), final loss = 6.65e+07
```

With **too small** a rate the point barely moves — after 40 steps it is still far from
the minimum and the loss is still near 1. With a **well-chosen** rate it lands almost
exactly on `(2, 1)` and the loss drops toward zero (about 0.09). With **too large** a
rate it overshoots, flies out of the frame, and the loss *grows* to about 66 million —
it diverges. Real networks have millions of parameters, but the intuition is the same.

## Building a neural network in practice

The workflow parallels what we discussed earlier in the course for supervised and
unsupervised learning: **define the problem** (classification vs.
regression, which fixes the differentiable **loss function**), **pre-process** (train/
validation/test split and input scaling), and **choose the architecture**. For real
deep learning we use **TensorFlow/Keras** or **PyTorch** (an **API**, Application
Programming Interface, is just a way for programs to communicate); scikit-learn's MLP is
enough for these small illustrations.

### Epochs, batches, and monitoring training

**Stochastic gradient descent** uses a random **batch** each step; one **epoch** is a
full pass through the training set. We watch the **training loss** and a held-out
**validation loss** fall over epochs.

```python
Xm, ym = make_moons(n_samples=1000, noise=0.25, random_state=0)
Xtr, Xval, ytr, yval = train_test_split(Xm, ym, test_size=0.3, random_state=0)
sc = StandardScaler().fit(Xtr); Xtr, Xval = sc.transform(Xtr), sc.transform(Xval)

clf = MLPClassifier(hidden_layer_sizes=(32, 32), activation="relu",
                    max_iter=1, warm_start=True, random_state=0)
train_loss, val_loss = [], []
for epoch in range(150):
    clf.partial_fit(Xtr, ytr, classes=[0, 1])
    train_loss.append(log_loss(ytr, clf.predict_proba(Xtr), labels=[0, 1]))
    val_loss.append(log_loss(yval, clf.predict_proba(Xval), labels=[0, 1]))

print(f"final training loss   : {train_loss[-1]:.3f}")
print(f"final validation loss : {val_loss[-1]:.3f}")
print(f"validation accuracy   : {clf.score(Xval, yval):.3f}")

plt.figure(figsize=(7, 4.5))
plt.plot(train_loss, label="training loss")
plt.plot(val_loss, label="validation loss")
plt.xlabel("epoch"); plt.ylabel("cross-entropy loss"); plt.legend()
plt.tight_layout(); plt.show()
```

**What the plot shows.** Both curves start high and fall steeply in the first ~20 epochs,
then flatten. The **validation** curve sits just **slightly above** the **training**
curve — the healthy, expected pattern. If the validation curve instead turned back
**upward** while training kept falling, that would signal **overfitting**, and **early
stopping** would halt training at the turn. The final numbers:

```
final training loss   : 0.154
final validation loss : 0.158
validation accuracy   : 0.933
```

Training and validation loss end close together (0.154 vs. 0.158) — little overfitting —
and the classifier reaches about 93% validation accuracy.

#### What overfitting looks like

To see **overfitting**, we make the problem hard to memorize *and* generalize at once:
shrink the training set, add a lot of noise, and use a much larger network with almost
no regularization. Now the two losses part ways.

```python
Xo, yo = make_moons(n_samples=120, noise=0.45, random_state=0)
Xtr2, Xval2, ytr2, yval2 = train_test_split(Xo, yo, test_size=0.4, random_state=0)
sc2 = StandardScaler().fit(Xtr2); Xtr2, Xval2 = sc2.transform(Xtr2), sc2.transform(Xval2)

over = MLPClassifier(hidden_layer_sizes=(200, 200), activation="relu", alpha=1e-6,
                     max_iter=1, warm_start=True, random_state=0)
tr_loss, va_loss = [], []
for epoch in range(400):
    over.partial_fit(Xtr2, ytr2, classes=[0, 1])
    tr_loss.append(log_loss(ytr2, over.predict_proba(Xtr2), labels=[0, 1]))
    va_loss.append(log_loss(yval2, over.predict_proba(Xval2), labels=[0, 1]))

best = int(np.argmin(va_loss))
print(f"validation loss is lowest at epoch {best} (loss {va_loss[best]:.3f})")
print(f"final training loss   : {tr_loss[-1]:.3f}")
print(f"final validation loss : {va_loss[-1]:.3f}  <- much higher than at its minimum")

plt.figure(figsize=(7, 4.5))
plt.plot(tr_loss, label="training loss")
plt.plot(va_loss, label="validation loss")
plt.axvline(best, color="gray", ls="--", label=f"early-stopping point (epoch {best})")
plt.xlabel("epoch"); plt.ylabel("cross-entropy loss"); plt.legend()
plt.tight_layout(); plt.show()
```

**What the plot shows.** The **training loss keeps falling** toward zero as the
oversized network memorizes the noisy training points, but the **validation loss bottoms
out early and then climbs back up**. A vertical dashed line marks the validation minimum.
The numbers:

```
validation loss is lowest at epoch 26 (loss 0.386)
final training loss   : 0.145
final validation loss : 0.973  <- much higher than at its minimum
```

After epoch 26 the model is fitting noise rather than signal, so it generalizes *worse*
even as the training loss keeps improving — the validation loss more than doubles from
its minimum (0.386) to the end (0.973). **Early stopping** ends training at the dashed
line (the validation minimum), keeping the most generalizable model. More training data
and regularization (L1/L2, dropout) are the other standard remedies.

### Architecture, training hyperparameters, and regularization

Networks expose many choices: the number of **hidden layers**, **neurons per layer**,
and the **activation** at each layer and the output. Rules of thumb: keep layer sizes
near a **power of two**, change sizes **gradually**, and use a **softmax** output for
classification. Training knobs include the **learning rate**, the **loss function**, the
**batch size**, and the **optimizer** — all variants of **stochastic gradient descent
(SGD)**: plain SGD, SGD with **momentum**, **AdaGrad / AdaDelta / RMSprop** (which adapt
the learning rate), and **Adam**, which combines RMSprop and momentum and is the most
common default.

Because networks have many weights, they can overfit. **L1 / L2 regularization**
penalizes the weights (L2 discourages large weights; L1 encourages sparsity); **batch
normalization** rescales a layer's inputs using batch statistics; and **dropout**
randomly removes weights during training (and, at inference, is used to add an
**uncertainty** estimate to an otherwise single-valued network).

---

# Neural Networks in Climate and Environmental Science

A recurring question is: **how can we emulate a complex, non-linear, high-dimensional
environmental process with machine learning?** Two applications — emulating sub-grid
parameterizations and tuning climate-model parameters — put the universal-approximation
property to work.

## Emulating sub-grid processes (parameterizations)

A climate model divides the Earth into a 3-D grid and steps the variables forward in
time. Because projections run for **hundreds of years**, grid boxes are large — a
**typical spacing is ~25–100 km**. Everything smaller — convection, cloud microphysics,
the radiative effects of clouds — still affects the grid-box averages and must be
represented by **parameterizations**: simplified models of the *aggregated* sub-grid
effect over a time step.

A concrete example is **super-parameterization**. Standard **CAM5** uses conventional
parameterizations at ~100 km; **SPCAM5** embeds a much higher-resolution model *inside
each grid box* — far more accurate but far too expensive for long runs. The idea is to
**train a neural network on SPCAM5** and drop it back into the coarse model in place of
the parameterization. Because inference is **fast**, the emulator delivers close to
SPCAM5 quality much more cheaply. One of the first papers to demonstrate this (published
in 2018) used super-parameterized CAM as ground truth.

A practical obstacle is software: many climate models are written in **Fortran** (huge
legacy codebases) while ML libraries are in **Python**. Coupling them is hard; groups
bridge Fortran and Python, rewrite parts in Python, or rebuild the model in **Julia**.

## Tuning climate-model parameters

Neural networks can also help **tune** a model's parameters. Parameterizations contain
parameters that start from physical understanding and are then **tuned to match
large-scale observations** (for example the **entrainment rate** of convective clouds or
the **fall velocity** of ice crystals). Doing this by hand is slow. A
**perturbed-parameter ensemble (PPE)** runs the model many times — say ~500 runs over
~40 parameters. A neural network then learns the **non-linear mapping** from input
parameters to outputs, acting as a cheap **surrogate** so we can explore far more of the
parameter space and find which parameters the model is most **sensitive** to.

### Parametric vs. structural error

Tuning often reveals that **no** parameter set fits *all* observations — a sign of
**structural error** (the equations are wrong: the truth is really "x squared", not
"x") rather than **parametric error** (the equations are right but a value like "w1" in
"y = w1 times x" is off). Tuning and PPE surrogates address parametric error; structural
error sends us back to improve the parameterizations themselves.
