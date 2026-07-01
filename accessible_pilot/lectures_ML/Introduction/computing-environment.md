# Computing Environment for Class

:::{admonition} Accessible workflow — read this first
:class: important
The steps below describe the **standard** browser-based JupyterHub interface. If
you use a screen reader, use the **accessible workflow** instead — VS Code with
Python scripts and a readable terminal, connecting to the same LEAP hub. It is
set up once and reused for the whole course:

**[Setting up an accessible
workflow](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html)**
(from CLMT5045; applies unchanged here). The datasets, commands, and Python are
identical — only the interface changes.
:::

## Our Course JupyterHub

[JupyterHub](https://jupyter.org/hub) is a multi-user Jupyter environment
designed for companies, classrooms and research labs. This course will use a
cloud-based JupyterHub environment supported by the [NSF LEAP
STC](https://leap.columbia.edu/) and managed by [2i2c](https://2i2c.org/service/).

You launch it at **<https://leap.2i2c.cloud/>** (on the standard site this link is
shown as an orange "jupyterhub — leap.2i2c.cloud" badge). A lot of documentation
about this hub can be found at <https://leap-stc.github.io/introduction/>.

You should already be familiar with this from CLMT 5045. If you need a refresher,
please check out the material
[here](https://earth-ds-ml.github.io/summer_2026/lectures_DS/computing_env/jupyterlab_and_colab.html)
— or, for the screen-reader version, the [accessible setup
guide](https://earth-ds-ml.github.io/summer_2026/accessible/lectures_DS/computing_env/accessible_setup.html).

## Computing environment and software libraries for machine learning

There are a number of different software libraries that can be used for machine
learning. Python is the most popular programming language used for machine
learning. Many of the most popular libraries for machine learning have been
developed in Python, including `PyTorch`, `TensorFlow`, `JAX`, and `Keras`.
Julia, a newer programming language focused on performance computing in
scientific and technical fields also supports machine learning libraries.

In this class, we will work with Python. We will use classic machine learning
algorithms in the `sci-kit learn` library, as well as deep learning models
implemented in `Tensorflow`.