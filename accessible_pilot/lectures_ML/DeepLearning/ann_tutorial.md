# Artificial Neural Networks (ANN): a hands-on tutorial

This hands-on tutorial accompanies the [Introduction to Artificial Neural
Networks](ann.md) notes. We first implement a multilayer perceptron (MLP) with
scikit-learn, then build, train, and evaluate one with Keras/TensorFlow on an
image-classification dataset (based on a notebook from Géron's *Hands-On Machine
Learning*).

:::{admonition} How to use this page (accessible version)
:class: important
On the standard site this is a Jupyter notebook. Here, run the code as a **Python
script** — see [Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).
Part 1 (scikit-learn) runs anywhere. Part 2 uses **TensorFlow**, which you install
once with `pip install tensorflow`; the first run **downloads** the Fashion-MNIST
dataset. The dataset is made of small images, so instead of viewing them this page
**describes** what each image figure contains and backs every result with **printed
numbers** (shapes, accuracies, probabilities) you can read. Any plot can also be
rendered with [MAIDR](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/sci_python/trying_maidr.html).
The accuracies and losses quoted below are from one real run; your numbers will be
very close.
:::

## Part 1 — Implementing an MLP with scikit-learn

We start with scikit-learn's `MLPClassifier` on the **Iris** dataset: four numeric
features (sepal length, sepal width, petal length, petal width) and three iris
species (setosa, versicolor, virginica). The task is to classify each sample's
species from its four features.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

iris = load_iris()

# split off a test set (10%), then a validation set (10% of the rest)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    iris.data, iris.target, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42)

# one hidden layer of 5 neurons; scale inputs first (always, for neural nets)
mlp_clf = MLPClassifier(hidden_layer_sizes=[5], max_iter=10_000, random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_clf)
pipeline.fit(X_train, y_train)

print("validation accuracy:", pipeline.score(X_valid, y_valid))
```

With scikit-learn you specify whether you want the **classifier** (`MLPClassifier`)
or the **regressor** (`MLPRegressor`), plus hyperparameters such as
`hidden_layer_sizes` (neurons per hidden layer), `activation` (default `relu`), and
`max_iter`. Wrapping the scaler and the classifier in a `make_pipeline` guarantees
the same scaling is applied to training, validation, and any future data.

**What the number shows.** On this small, well-separated dataset the tiny network
classifies the validation set perfectly:

```
validation accuracy: 1.0
```

scikit-learn is convenient for a quick MLP, but for real deep learning we switch to a
dedicated library with far more flexibility.

## Part 2 — Implementing an MLP with Keras/TensorFlow

We use **TensorFlow** with its **Keras** API on **Fashion-MNIST**: 70,000 grayscale
images (28×28 pixels) of clothing items, each labeled with one of ten classes. It is
a drop-in, more interesting stand-in for the classic digit MNIST.

```python
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]   # 55,000 train
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]   #  5,000 validation

print("training set shape:", X_train.shape)   # (55000, 28, 28)
print("pixel dtype       :", X_train.dtype)   # uint8, values 0-255
```

**What the numbers show.** The training set is `(55000, 28, 28)` — 55,000 images of
28×28 pixels — and pixels are unsigned bytes from 0 to 255. Neural networks train
better on small, floating-point inputs, so we rescale to the 0–1 range:

```python
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.
```

The next two cells display images. The first shows a single training image; the
second shows a 4×10 grid of examples with their class names as titles.

```python
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

plt.imshow(X_train[0], cmap="binary"); plt.axis("off"); plt.show()

n_rows, n_cols = 4, 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis("off")
        plt.title(class_names[y_train[index]])
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
```

**What the figures show.** Each image is a small grayscale picture of a clothing item
on a white background (a boot, a shirt, a bag, and so on) — the silhouette in dark
pixels, roughly centered in the 28×28 frame. The grid shows ten such items per row
with their labels, so you can see the variety the model must learn to tell apart. The
labels themselves are integers 0–9; `class_names` maps each integer to a readable
name. You do not need to see the pixels to follow the rest of the tutorial — the model
works from the numeric pixel arrays.

### Building the model (Sequential API)

We stack layers with the Keras **Sequential** API: a `Flatten` layer turns each 28×28
image into a length-784 vector, two hidden `Dense` layers (300 and 100 neurons, ReLU
activation) learn features, and a final `Dense` layer of 10 neurons with **softmax**
outputs a probability for each class.

```python
tf.keras.backend.clear_session()
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.summary()
```

**What `model.summary()` shows.** A text table listing each layer, its output shape,
and its parameter count. The counts come straight from the layer sizes:

- `Flatten` → output length **784**, 0 parameters (it only reshapes).
- `Dense(300)` → 784×300 weights + 300 biases = **235,500** parameters.
- `Dense(100)` → 300×100 + 100 = **30,100** parameters.
- `Dense(10)` → 100×10 + 10 = **1,010** parameters.
- **Total: 266,610 trainable parameters** — a lot to learn even for this small MLP.

You can inspect a layer's learned arrays directly. The first hidden layer's weight
matrix has shape `(784, 300)` (one weight per input pixel per neuron), and its biases
start as a length-300 vector of zeros:

```python
hidden1 = model.layers[1]
weights, biases = hidden1.get_weights()
print("weights shape:", weights.shape)   # (784, 300)
print("biases shape :", biases.shape)    # (300,)  -- all zeros at initialization
```

### Compiling and training

Compiling fixes the **loss**, the **optimizer**, and the **metrics** to report. For
integer class labels we use sparse categorical cross-entropy; the optimizer is plain
stochastic gradient descent (`sgd`).

```python
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))
```

**What training prints.** For each of the 30 epochs Keras prints the training loss and
accuracy and the validation loss and accuracy. They improve steadily — for example the
first epoch reports roughly `loss 0.999, accuracy 0.69, val_loss 0.503,
val_accuracy 0.82`, and both losses fall over the run while both accuracies rise.

Plotting the recorded history makes the trend visible:

```python
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.legend(loc="lower left")
plt.show()
```

**What the plot shows.** Four curves against epoch (0 to 29), all between 0 and 1: the
two **loss** curves (training and validation, red dashed) start near 1 and **decrease**,
flattening out; the two **accuracy** curves (training and validation, blue solid)
**rise** toward about 0.85–0.90. Training and validation curves stay close together,
which means little overfitting. Render this through MAIDR to hear the four series, or
read them from `history.history` (a dictionary of per-epoch lists).

Finally, evaluate on the held-out test set:

```python
print(model.evaluate(X_test, y_test))
```

**What the number shows.** About `loss 0.368, accuracy 0.873` — the trained network
correctly classifies roughly **87%** of unseen test images.

### Making predictions

For the first three test images the model outputs a probability per class; the
predicted class is the one with the highest probability.

```python
X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))

y_pred = y_proba.argmax(axis=-1)
print("predicted:", np.array(class_names)[y_pred])
print("actual   :", np.array(class_names)[y_test[:3]])
```

**What the numbers show.** Each row of `y_proba` is ten probabilities that sum to 1.
For the first image the model is about **74%** sure it is an *Ankle boot* (with 25% on
*Sandal* — a reasonable confusion); the second and third rows are essentially 100% sure
of *Pullover* and *Trouser*. Taking the arg-max gives predictions
`['Ankle boot', 'Pullover', 'Trouser']`, which **match the true labels exactly**. The
first prediction being less confident illustrates how a softmax output doubles as a
rough confidence estimate.

The final figure shows those three test images with their true labels as titles — a
boot, a pullover, and a pair of trousers — confirming visually what the numbers already
told us.
