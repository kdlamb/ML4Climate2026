# Introduction to physics-informed machine learning

This page demonstrates how we might include physics into our machine learning
models.

In the environmental sciences, we generally have some prior physical knowledge
about the system we are interested in modeling. In this module, we will
introduce some ways in which physical knowledge can be integrated with machine
learning to improve predictions.

**Some examples in the environmental sciences**

- [T. Beucler et al. "Enforcing Analytic Constraints in Neural Networks
  Emulating Physical Systems." Phys. Rev. Lett. 126,
  098302](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.098302)
- [K. Kashinath et al. 2021 "Physics-informed machine learning: case studies
  for weather and climate modelling." Phil. Trans. R. Soc.
  A.37920200093](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2020.0093)
- [R. ElGhawi et al. "Hybrid modeling of evapotranspiration: inferring stomatal
  and aerodynamic resistances using combined physics-based and machine
  learning" Env. Res. Lett. 18, 034039,
  2023](https://iopscience.iop.org/article/10.1088/1748-9326/acbbe0/meta)
- [D. Kochkov et al. "Neural general circulation models for weather and
  climate" Nature, 632, 1060–1066
  (2024)](https://www.nature.com/articles/s41586-024-07744-y)

**Goal:** using simple examples of physical systems, we will explore how prior
physical knowledge can be included in machine learning models.

**Methods to be discussed:**

- Soft and Hard Physical Constraints
- Hybrid physics-ML approaches
- Physics-Informed Neural Networks
- Neural Ordinary Differential Equations

:::{admonition} How to use this page (accessible version)
:class: important
On the standard site this is a Jupyter notebook. Here the same code is given
as blocks you can run as a **single Python script**: see [Setting up an
accessible workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
You will need `tensorflow`, `numpy`, and `matplotlib` installed
(`pip install tensorflow` in your course environment). Each figure is followed
by a **"What the plot shows"** description and the key **printed numbers**, so
you can follow the argument from the numbers rather than the picture. Any plot
can also be rendered through
[MAIDR](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/sci_python/trying_maidr.html)
to explore it by sound. The four training runs on this page take a few minutes
in total on a laptop; each prints its loss every 10% of the way so you can
hear the progress. Neural network training is not exactly reproducible across
machines (parallel hardware sums numbers in different orders), so your printed
numbers will differ a little from the ones quoted below — the *qualitative*
statements are what to check.
:::

## Embedding physical knowledge into our neural networks

Neural networks can act as universal function approximators (Hornik et al.
1989). This property makes them very useful for supervised regression tasks
where we want to learn a non-linear mapping between inputs and outputs.
However, unlike physical models, which generally have some limiting factors in
terms of how "wrong" they can be, when neural networks are wrong, they
generally produce extremely unphysical predictions. In particular,
unconstrained neural networks will generalize very poorly outside of the data
regime in which they are trained.

### Soft and hard physical constraints

One way that we can improve a neural network's prediction for physical models
is to include constraints on the neural network's prediction. We often
distinguish between two types of constraints:

- **Soft physical constraints** refer to adding an additional term to the loss
  function in order to impose a physical constraint that we know a priori
  about our system. These will be problem specific, as they will depend on the
  specific physical laws that we want to enforce.
- **Hard physical constraints** refer to updating the architecture of our
  neural network to impose specific physical functions.

### Symmetries, invariances, and equivariances

By including known symmetries into our machine learning models, we can often
reduce the complexity of our machine learning models, and improve the
robustness of our predictions.

One way to include this type of prior physical knowledge is through the
training process. For example, if we don't expect that rotations of our input
would have a physical effect on our output, we can impose translational or
rotational invariance by augmenting our data sets with random translations or
rotations.

Another approach is through the architecture of the network itself. Different
types of models implicitly assume different types of invariances.

:::{admonition} Figure description: inductive biases of network architectures
On the standard site this section shows a figure from
[Battaglia et al. 2018](https://arxiv.org/pdf/1806.01261.pdf). Its core is a
table with one row per architecture component, listing the entities it
operates on, the relations between them, the *relational inductive bias* it
assumes, and the invariance that follows:

| Component | Entities | Relations | Inductive bias | Invariance |
|---|---|---|---|---|
| Fully connected | Units | All-to-all | Weak | — |
| Convolutional | Grid elements | Local | Locality | Spatial translation |
| Recurrent | Timesteps | Sequential | Sequentiality | Time translation |
| Graph network | Nodes | Edges | Arbitrary | Node, edge permutations |

Diagrams below the table illustrate each row: a fully connected layer draws an
arrow from every input unit to every output unit; a convolutional layer reuses
the same small set of weights at each spatial position ("sharing in space"); a
recurrent layer reuses the same weights at each timestep ("sharing in time").
A second panel shows physical systems drawn as graphs: a water molecule, a
mass-spring system, an n-body system, a rigid-body collision, a sentence parse
tree, and an image scene — each rendered as nodes connected by edges, the
natural representation for a graph network.
:::

## Hybrid Physics-ML approaches

Sometimes we want to integrate a lot of prior physical knowledge into our
system, but we also want to include a neural network prediction to model parts
of the system that we don't know. One way to do this is through using
hybrid-physics machine learning models. Hybrid physics-machine learning models
are ones that include both domain knowledge (first principles physical
knowledge about our system) as well as data-driven methods.

Physical models lend themselves to interpretability, but suffer from high
computational costs (in some cases) and can also have uncertainty in terms of
both parameters in the equations and the structure of the equations
themselves.

Data-driven models are learned directly from data or higher resolution
simulations that explicitly resolve physics that may be uncertain at lower
resolutions. However, they are hard to interpret, and the parameters that they
learn do not correspond to parameters of the system, and thus may not be
physically interpretable.

By combining both physical models and data-driven models, hybrid models will
have improved interpretability over purely data-driven models, but also may
have improved accuracy by their ability to learn unknown or uncertain physics
directly from observations.

## Physics-Informed Neural Networks

One type of hybrid-physics machine learning model is a physics-informed neural
network. Physics-informed neural networks (Raissi et al. 2019) are an approach
that we can use to enforce physical constraints on our machine learning model.

When we are in a regime where we don't have a lot of data but do have prior
physical knowledge, we can regularize our neural network with differential
equations. The example that we will go through below is adapted (from PyTorch
to TensorFlow/Keras) from the tutorial by
[Theo Wolf](https://medium.com/@theo.wolf/physics-informed-neural-networks-a-simple-tutorial-with-pytorch-f28a890b874a).
We'll start by loading some python packages.

```python
import functools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
```

As an example, we will use Newton's law of cooling. This equation describes
the temperature of an object (for example, a cup of coffee) placed into an
environment that is at a different temperature (here, we'll assume the cup of
coffee is placed onto the table at room temperature). We can write this as an
ordinary differential equation describing the rate of heat loss from the
object:

$$\frac{dT(t)}{dt} = r(T_{env}-T(t))$$

That is: the rate of change of the coffee's temperature T at time t equals the
cooling rate r times the difference between the environment temperature T_env
and the current temperature T(t).

In this case we can find an analytical solution to this ordinary differential
equation, describing the temperature of our cup of coffee after t seconds as

$$T(t) = T_{env}+(T_{0}-T_{env})e^{-rt}$$

read as: T of t equals T_env plus (T_0 minus T_env) times e to the power of
minus r times t, where T_0 is the initial temperature of our cup of coffee.
We write the time dependence of our cup of coffee in python:

```python
def cooling_law(time, Tenv, T0, R):
    T = Tenv + (T0 - Tenv) * np.exp(-R * time)
    return T
```

We will use the analytical solution to show the true solution of our
differential equation, given an initial temperature of our cup of coffee at
T_0 = 100 degrees C, and the temperature of the environment at
T_env = 25 degrees C. We plot this function below, and also create a training
data set of 10 data points for our machine learning model. These 10 data
points represent measurements of our cup of coffee over time (so we'll add
some normally distributed noise). These 10 measurements are evenly spaced over
the first 300 seconds that we are observing our cup of coffee.

```python
np.random.seed(10)
tf.random.set_seed(0)

Tenv = 25
T0 = 100
R = 0.005
times = np.linspace(0, 1000, 1000)
eq = functools.partial(cooling_law, Tenv=Tenv, T0=T0, R=R)
temps = eq(times)

# Make training data
t = np.linspace(0, 300, 10)
T = eq(t) + 2 * np.random.randn(10)

plt.plot(times, temps)
plt.plot(t, T, 'o')
plt.legend(['Equation', 'Training data'])
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')
plt.show()

print("true T at t=0, 300, 1000 s:",
      [round(float(eq(x)), 1) for x in (0, 300, 1000)])
print("training data spans t=0 to 300 s; noisy temps:",
      np.round(T, 1))
```

:::{admonition} What the plot shows
The smooth curve starts at 100 degrees C at t = 0 and decays exponentially
toward the room temperature of 25 degrees C: it drops to about 41.7 degrees by
t = 300 s and is essentially flat at about 25.5 degrees by t = 1000 s. The 10
circular markers (the training data) sit close to the curve but only over the
first 300 seconds — the rest of the curve has no data on it. That gap is the
point of the whole exercise: everything after t = 300 s must be
*extrapolated*.
:::

First, we will look at what happens if we train a simple neural network (a
fully connected multi-layer perceptron with ReLU activations). The code below
creates an MLP using Keras, along with a training function that we will reuse
for all of the networks on this page. The training function minimizes the mean
squared error on the training data, and can optionally add a second,
physics-based loss term (we will use this for the physics-informed networks
below).

```python
def make_mlp(n_units=100):
    """A fully-connected MLP that maps time t to temperature T(t)."""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(n_units, activation="relu", input_shape=(1,)),
        tf.keras.layers.Dense(n_units, activation="relu"),
        tf.keras.layers.Dense(n_units, activation="relu"),
        tf.keras.layers.Dense(n_units, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

def train(model, t, T, epochs=5000, lr=1e-4, physics_loss=None,
          physics_weight=1.0, extra_variables=None):
    """Train with MSE loss on (t, T), plus an optional physics loss term."""
    Xt = tf.constant(t.reshape(-1, 1), tf.float32)
    yt = tf.constant(T.reshape(-1, 1), tf.float32)
    # extra_variables lets us learn parameters that are not network weights
    # (we will use this to learn the cooling rate r later on)
    variables = model.trainable_variables + (extra_variables or [])
    optimizer = tf.keras.optimizers.Adam(lr)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((yt - model(Xt, training=True)) ** 2)
            if physics_loss is not None:
                loss += physics_weight * physics_loss(model)
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        return loss

    losses = []
    for ep in range(epochs):
        losses.append(float(train_step()))
        if ep % int(epochs / 10) == 0:
            print(f"Epoch {ep}/{epochs}, loss: {losses[-1]:.2f}")
    return losses
```

We will first train the neural network with mean squared error (MSE) loss. The
input for our network here is the 10 times that we measured our cup of coffee,
and the output here are the 10 temperatures that we measured at those times.

:::{admonition} Figure description: training a vanilla neural network
On the standard site this section shows a diagram (credit:
[B. Moseley](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/))
of a small fully connected network: an input node x on the left, two hidden
layers of nodes labeled with weights theta, and an output node u on the right.
An arrow from the output leads to a single box labeled "Compare to training
data" — the only training signal for a vanilla network is the mismatch between
its output and the observations.
:::

Because the neural network acts as a universal function approximator, we will
denote the neural network as f(t | theta). That is, it takes in the time t and
maps this to a prediction for the temperature, given the values of the weights
of the neural network theta.

MSE loss for the 10 data points is calculated by

$$Loss_{MSE} = \frac{1}{10}\sum_{j=0}^{10} (f(t_{j}|\theta)-T_{j})^2$$

read as: the mean over the 10 data points of the squared difference between
the network's prediction at time t_j and the measured temperature T_j.

The code below trains the MLP for 5000 epochs.

```python
net = make_mlp()
losses = train(net, t, T, epochs=5000, lr=1e-4)

plt.plot(losses)
plt.yscale('log')
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()
```

:::{admonition} What the plot shows
The loss curve (on a logarithmic y axis) starts near 4600, stays on a plateau
around 2000–3000 for the first thousand or so epochs, then falls steeply and
flattens out below 1 — of order the noise we added to the data. The training
data has been fit essentially as well as it can be.
:::

Now let's look at how well the trained neural network does. We first use the
trained neural network to predict the temperature of our cup of coffee over
the entire 1000 seconds in our data set, then plot the predictions against the
true equation.

```python
preds = net.predict(times.reshape(-1, 1), verbose=0)

plt.plot(times, temps, alpha=0.8)
plt.plot(t, T, 'o')
plt.plot(times, preds, alpha=0.8)
plt.legend(labels=['Equation', 'Training data', 'Vanilla Network'])
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')
plt.show()

for tq in (150, 300, 600, 999):
    print(f"t={tq:4d} s: true {eq(times[tq]):6.1f} C, "
          f"vanilla network {preds[tq][0]:6.1f} C")
```

:::{admonition} What the plot shows
Over the first 300 seconds (where the training data lives) the network's curve
lies almost on top of the true equation, apart from a brief overshoot right at
t = 0. Beyond t = 300 s the two curves peel apart: the true curve keeps
decaying toward 25 degrees, while the network's prediction stops decreasing
and climbs steadily *upward*. The printed numbers make the failure concrete:
at t = 300 s the network is within a fraction of a degree of the truth, but by
t = 999 s it predicts around 90 degrees in our run, where the truth is 25.5
degrees — the coffee has mysteriously started reheating itself.
:::

As can be seen, the neural network does a decent job of predicting the
temperature of the cup of coffee for the first 300 seconds, but it does a very
poor job extrapolating to future times (t greater than 300 seconds) because
the neural network has no physical constraints on its prediction. It also is
not quite sure how to handle the initial points.

:::{admonition} Figure description: training a physics-informed neural network
The companion diagram to the vanilla-network one (credit:
[B. Moseley](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/)):
the same network mapping input coordinates x to physical quantities u, but now
with *two* arrows leaving the output. One goes to the familiar "Compare to
training data" box; the second goes through the derivatives du/dx, d²u/dx²,
... to a box labeled "Compute derivatives and minimise underlying equation
residual". A PINN is trained on both signals at once.
:::

Next, we will look at how to use a physics-informed neural network instead.
The idea behind PINNs is that we can impose a physical constraint on our
neural network prediction, given our prior physical knowledge about the
system. To do this, we will use what are called collocation points.
Collocation points are points that we will evaluate our network at during the
training process to ensure that the neural network prediction will be
consistent with our known prior physical knowledge.

In this case, our prior physical knowledge is Newton's cooling law, so we can
write this constraint as

$$g(t,T) = \frac{dT(t)}{dt} - r(T_{env}-T(t)) = 0$$

We can use Newton's cooling law to write a physics loss term for the data as

$$g(t,f(t|\theta)) = \frac{df(t|\theta)}{dt} - r(T_{env}-f(t|\theta))$$

where we have just replaced the temperature with the prediction from our
neural network.

We will then evaluate this physics loss term at M = 1000 collocation points.
Our physics loss term is the mean over the collocation points of the squared
equation residual:

$$Loss_{physics} = \frac{1}{M}\sum_{i=0}^{M} \left(\frac{df(t_{i}|\theta)}{dt_{i}} - r(T_{env}-f(t_{i}|\theta))\right)^2$$

In order to evaluate this term, we need to take the derivative of our neural
network as it's being trained. We can use automatic differentiation to do
this.

:::{admonition} Figure description: automatic differentiation
A figure from [Baydin et al. 2015](https://arxiv.org/abs/1502.05767) showing a
small two-input, two-hidden-unit, one-output network drawn twice. In the
forward pass (left to right), inputs x1 and x2 flow through weighted
connections w1 through w6 to produce the output y3, and an error E(y3, t) is
computed against the target t. In the backward pass (right to left, drawn as
dashed arrows), the error adjoint is propagated back through the same
connections, yielding the gradient of E with respect to every weight — the
quantities gradient descent needs. The caption notes the gradient with respect
to the *inputs* can be computed in the same backward pass — which is exactly
what a PINN uses to get df/dt.
:::

TensorFlow uses `tf.GradientTape` to do automatic differentiation: operations
executed inside the tape's context are recorded, and the tape can then compute
the gradient of any recorded output with respect to any recorded input. We
already use one tape inside our training function to get the gradients of the
loss with respect to the network weights theta. For the physics loss we open a
second, nested tape that instead differentiates the network output with
respect to its *input* t, giving us df/dt at the collocation points. (PyTorch
and JAX have equivalent functionality: `torch.autograd.grad` and `jax.grad`.)

```python
# make M = 1000 collocation points
ts_colloc = tf.reshape(tf.linspace(0.0, 1000.0, 1000), (-1, 1))

def physics_loss(model):
    """The physics loss of the model"""
    with tf.GradientTape() as tape:
        # watch the collocation points so we can differentiate wrt input times
        tape.watch(ts_colloc)
        # run the collocation points through the network
        temps_c = model(ts_colloc)
    # get the gradient dT/dt
    dT = tape.gradient(temps_c, ts_colloc)
    # compute the ODE
    ode = dT - R * (Tenv - temps_c)
    # MSE of ODE
    return tf.reduce_mean(ode ** 2)
```

Now that we have the physics loss function, we will train the model with both
the MSE loss term and the physics loss term. The training objective is now to
minimize

$$Loss_{MSE}+\lambda Loss_{physics}$$

where lambda is a hyperparameter that weighs the second loss term. Here we
will choose lambda = 1. We will again train the network for 5000 epochs, but
now training for both loss terms.

```python
netpinn = make_mlp()
losses = train(netpinn, t, T, epochs=5000, lr=1e-4,
               physics_loss=physics_loss, physics_weight=1.0)

plt.plot(losses)
plt.yscale('log')
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()

predspinn = netpinn.predict(times.reshape(-1, 1), verbose=0)

plt.plot(times, temps, alpha=0.8)
plt.plot(t, T, 'o')
plt.plot(times, preds, alpha=0.8)
plt.plot(times, predspinn, alpha=0.8)
plt.legend(labels=['Equation', 'Training data', 'Vanilla Network', 'PINN'])
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')
plt.show()

for tq in (150, 300, 600, 999):
    print(f"t={tq:4d} s: true {eq(times[tq]):6.1f} C, "
          f"vanilla {preds[tq][0]:6.1f} C, PINN {predspinn[tq][0]:6.1f} C")
```

:::{admonition} What the plots show
The loss curve looks much like before: a plateau, a steep fall, and a floor of
a few degrees squared. The prediction plot is where the difference lives. Over
the training region both networks track the true curve, but beyond t = 300 s
the PINN keeps decaying toward room temperature while the vanilla network
climbs upward. In our run's printed numbers: at t = 600 s the truth is 28.7
degrees, the vanilla network says 58.9, the PINN 27.1; at t = 999 s the truth
is 25.5 degrees, the vanilla network says about 90, the PINN about 37 (it
drifts up a little at the very end of the collocation window, but stays in the
physically sensible range).
:::

The PINN performs much better than the original vanilla network that was
trained without any physics constraint.

### Using a PINN to discover unknown parameters

Next, we will look at how we can use the PINN to learn unknown parameters in
our model. For example, if we did not know the parameter r in our differential
equation, how can we use a PINN to estimate what it is?

To add r as a differentiable parameter that will be learned along with the
weights of the network, we define it as a `tf.Variable`, which means that it
is differentiable and can be updated by the optimizer. We pass it to our
training function through the `extra_variables` argument, so that the
optimizer updates it alongside the network weights. To update the physics loss
term, we use the current value of the r parameter, rather than the known value
of r (recall that we used r = 0.005 when we created our data set).

```python
r_learned = tf.Variable(0.0)

def physics_loss_discovery(model):
    with tf.GradientTape() as tape:
        tape.watch(ts_colloc)
        temps_c = model(ts_colloc)
    dT = tape.gradient(temps_c, ts_colloc)
    # note that here we now use r_learned rather than R because this
    # parameter will be updated as the model is trained
    ode = r_learned * (Tenv - temps_c) - dT
    return tf.reduce_mean(ode ** 2)
```

Now we will train the network with this updated loss function. The network
will learn how to predict the temperature, given the time, and the current
value for r.

One subtlety: the discovery problem has a trivial solution that the optimizer
can fall into — if the network predicts a flat temperature outside the
training data, then dT/dt is approximately 0 there, and the physics loss can
be minimized by simply driving r to zero. Weighting the physics term more
heavily (here we use lambda = 10) and training for longer pushes the optimizer
away from this trivial solution, so that the network has to find an r
consistent with the trend in the observed data.

```python
netdisc = make_mlp()
losses = train(netdisc, t, T, epochs=10000, lr=1e-4,
               physics_loss=physics_loss_discovery, physics_weight=10.0,
               extra_variables=[r_learned])

predspinndiscover = netdisc.predict(times.reshape(-1, 1), verbose=0)

plt.plot(times, temps, alpha=0.8)
plt.plot(t, T, 'o')
plt.plot(times, preds, alpha=0.8)
plt.plot(times, predspinn, alpha=0.8)
plt.plot(times, predspinndiscover, alpha=0.8)
plt.legend(labels=['Equation', 'Training data', 'Vanilla Neural Network',
                   'PINN', 'discovery PINN'])
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')
plt.show()

print(f"learned r = {float(r_learned):.5f} (true value 0.005)")
print(f"discovery PINN at t=999 s: {predspinndiscover[999][0]:.1f} C "
      f"(true {eq(times[999]):.1f} C)")
```

:::{admonition} What the plot shows
The discovery PINN's curve is nearly indistinguishable from the ordinary
PINN's: it tracks the data over the first 300 seconds and keeps decaying
toward room temperature afterwards, ending within a couple of degrees of the
true 25.5 degrees at t = 999 s (27.2 degrees in our run). Even though the
model had to learn the r parameter along with the temperature dependence, the
predictions are on par with the PINN that was given the true cooling rate. The
printed learned r comes out close to 0.005, the value we used to generate the
data — 0.0054 in our run, typically within about 10 percent.
:::

A great resource for more complex applications of PINNs is
[Wang et al. 2023](https://arxiv.org/abs/2308.08468).

## Neural Ordinary Differential Equations (NODEs)

Another way that we can include physics in our machine learning model is
through using neural ordinary differential equations (NODEs). NODEs were
introduced in [Chen et al. 2018](https://arxiv.org/abs/1806.07366), and use
neural networks to learn an unknown function that is the derivative of an
observed quantity. That is, we can use a neural network to parameterize the
function f(h(t), t, theta), where f is the derivative of a function h(t):

$$\frac{dh(t)}{dt} = f(h(t),t,\theta)$$

To learn the unknown function f, which is parameterized as a neural network
with weights theta, we can perform back-propagation through typical numerical
ODE solvers.

We'll continue with the example of Newton's Law of Cooling, and create a NODE
model trained on observed temperature points to then predict future states of
the system. In this case, we do not directly include the physical law function
(as we did with PINNs), but we use neural ODEs to learn the function dT/dt.

:::{admonition} Figure description: latent ODE computation graph
A figure from Chen et al. 2018 contrasting discrete and continuous sequence
models. Along the bottom, a time axis with observation times t_0 through t_N
labeled "Observed" and later times t_N+1 through t_M labeled "Unobserved". A
smooth curve x(t) passes through data points at the observed times. In the
model, an ODE solver takes the initial state and the list of query times and
produces states along a *continuous* trajectory — drawn as a smooth curve with
diamonds marking the query times — covering both the prediction interval and
the extrapolation interval beyond the data. The contrast with an RNN is that
the RNN only defines its state at the discrete observation steps, while the
ODE defines it at every time in between as well.
:::

We'll start by parameterizing the temperature derivative as a neural network.
The model below is an MLP that will be used to learn the temperature
derivative as an unknown function. It takes in the current temperature
(normalized to the expected temperature range) and outputs the current rate of
change of the temperature.

```python
def make_dTdt(n_units=100):
    """An MLP that maps the current temperature T to its derivative dT/dt."""
    out_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5)
    return tf.keras.Sequential([
        # do the normalization here (temperatures span roughly 0-105 C)
        tf.keras.layers.Rescaling(1.0 / 105.0, input_shape=(1,)),
        tf.keras.layers.Dense(n_units, activation="relu"),
        tf.keras.layers.Dense(n_units, activation="relu"),
        tf.keras.layers.Dense(n_units, activation="relu"),
        tf.keras.layers.Dense(1, kernel_initializer=out_init,
                              bias_initializer="ones"),
    ])
```

We then need to numerically integrate the function that we are learning. For
PyTorch and JAX there are libraries that provide differentiable ODE solvers
(`torchdiffeq`, from Chen et al. 2018, and `diffrax`). Here we instead write
our own small fixed-step integrator directly in TensorFlow, using the
classical Runge-Kutta 4th order method ("RK4"). Because each integration step
is built out of ordinary differentiable TensorFlow operations, gradients can
flow through the entire integration — this is all we need to be able to train
the network.

```python
def rk4_integrate(f, y0, ts):
    """Integrate dy/dt = f(y) with one RK4 step per interval in ts.

    f  - a function (here, a neural network) giving the derivative dy/dt
    y0 - the initial value of the temperature
    ts - the times we want the solution at
    """
    ys = [y0]
    y = y0
    for i in range(len(ts) - 1):
        h = ts[i + 1] - ts[i]
        k1 = f(y)
        k2 = f(y + h / 2 * k1)
        k3 = f(y + h / 2 * k2)
        k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(y)
    return tf.concat(ys, axis=0)
```

Recall that an ODE initial value problem has the form

$$\frac{dy}{dt} = f(y(t),t,\theta),\quad y(0) = y_{0}$$

We'll give the initial temperature point T_0 to our NODE model, along with a
vector of integration times. Then we will optimize the NODE model against the
10 observed temperature points, in order to learn the optimal parameters theta
for the neural network parameterizing dT/dt.

```python
num_iterations = 4000

dTdt = make_dTdt()
optimizer = tf.keras.optimizers.experimental.AdamW(1e-4)

# The observed temperatures, the time points we integrate over,
# and the initial observed temperature
yt = tf.constant(T.reshape(-1, 1), tf.float32)
inttimes = [float(ti) for ti in t]
Temp0 = tf.constant([[T[0]]], tf.float32)

@tf.function
def node_step():
    with tf.GradientTape() as tape:
        Temps = rk4_integrate(dTdt, Temp0, inttimes)
        # We'll use MSE loss here.
        loss = tf.reduce_mean((Temps - yt) ** 2)
    grads = tape.gradient(loss, dTdt.trainable_variables)
    optimizer.apply_gradients(zip(grads, dTdt.trainable_variables))
    return loss

losses = []
for itr in range(num_iterations):
    losses.append(float(node_step()))
    if itr % int(num_iterations / 10) == 0:
        print(f"Epoch {itr}/{num_iterations}, loss: {losses[-1]:.2f}")

plt.plot(losses)
plt.yscale("log")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()
```

:::{admonition} What the plot shows
The NODE loss starts very high (tens of thousands — the initial random
derivative sends the trajectory far from the data) but drops extremely fast,
reaching order 1 within the first thousand iterations and staying there. The
printed loss values show this: five digits at epoch 0, single digits well
before the halfway point.
:::

Now that the model is trained, we integrate the learned derivative forward
from the initial temperature over the full 1000 seconds, and compare against
our previous models.

```python
extraptimes = np.arange(0, 1000).astype(float)
extrapTemps = rk4_integrate(dTdt, Temp0, extraptimes)

plt.plot(times, temps, alpha=0.8)
plt.plot(t, T, 'o')
plt.plot(times, preds, alpha=0.8)
plt.plot(times, predspinn, alpha=0.8)
plt.plot(extraptimes, extrapTemps[:, 0], alpha=0.8)
plt.legend(labels=['Equation', 'Training data', 'Vanilla Neural Network',
                   'PINN', 'NODE'])
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')
plt.show()

for tq in (300, 600, 999):
    print(f"t={tq:4d} s: true {eq(float(tq)):6.1f} C, "
          f"NODE {float(extrapTemps[tq, 0]):6.1f} C")
```

:::{admonition} What the plot shows
The NODE trajectory starts at the first noisy measurement and follows the true
decay curve through the training window. In the extrapolation region it —
unlike the vanilla network — keeps decaying, but then levels off a few degrees
*above* room temperature instead of decaying all the way down. In our run the
printed numbers were: t = 300 s true 41.7, NODE 41.5; t = 600 s true 28.7,
NODE 31.3; t = 999 s true 25.5, NODE 30.1. The next code block shows exactly
why it flattens where it does.
:::

The NODE model doesn't do quite as well as the PINNs — its prediction levels
off above room temperature rather than decaying all the way — but it does
significantly better than the vanilla neural network. Recall that we have not explicitly included
any physical constraint here, except the assumption that our observed
temperature observations are the result of integrating a continuous function.

Since we have learned the derivative dT/dt here, we can also look at the
functional dependence of the trained neural network to try to better
understand the properties of this function. In cases where we don't have a
prior physical constraint on this function, or only have partial physical
understanding of this function, NODEs can be powerful tools to improve
predictions and to potentially even learn unknown physics by optimizing
against observations.

```python
r = 0.005
Tenv = 25

dTdtreal = []
dTdtnode = []
Temperatures = np.arange(25, 100, 1)

for Tcurr in Temperatures:
    dT = dTdt(tf.constant([[float(Tcurr)]]))
    dTdtreal.append(r * (Tenv - Tcurr))
    dTdtnode.append(float(dT))

plt.scatter(dTdtreal, dTdtnode, c=Temperatures)
plt.colorbar(label='Temperature (C)')
plt.xlabel('True dT/dt')
plt.ylabel('Learned dT/dt')
plt.show()

corr = np.corrcoef(dTdtreal, dTdtnode)[0, 1]
print(f"correlation between true and learned dT/dt: {corr:.3f}")
for Tq in (30, 60, 90):
    dT = float(dTdt(tf.constant([[float(Tq)]])))
    print(f"T={Tq} C: true dT/dt {r*(Tenv-Tq):+.3f}, learned {dT:+.3f}")
```

:::{admonition} What the plot shows
A scatter of the learned derivative against the true derivative
r(T_env − T), one point per temperature from 25 to 100 degrees C, colored by
temperature. Over the temperatures the training data covered (roughly 41 to
100 degrees), the points fall close to the one-to-one line — in our run the
correlation was 0.984, and at T = 60 degrees the true derivative was −0.175
against a learned −0.172. At the cold end, below the coldest observed
temperature, the curve bends away from the one-to-one line: at T = 30 degrees
the true derivative is −0.025 but the learned one is essentially zero. A
derivative of zero means "stop changing" — which is exactly the temperature
where the NODE's trajectory flattened out.
:::

Over the range of temperatures covered by the training data, the learned
derivative closely follows the true cooling law dT/dt = r(T_env − T) — the
network has recovered the underlying physics directly from the 10 noisy
observations. Below the coldest observed temperature, however, the learned
derivative reaches zero early, which is exactly why the NODE's prediction
flattens out above room temperature instead of decaying all the way down.
Like the vanilla network, the learned derivative is only trustworthy where
data constrained it.
