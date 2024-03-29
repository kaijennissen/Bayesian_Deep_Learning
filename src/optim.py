"""
Step 1: Build a simple (deterministic) MLP with JAX and Backpropagation
Step 2: Build a probabilistic MLP handling epistemic unvertainty i.e. learn
the parameters (mu and sigma) of a Gaussian distribution with first order optimization.
Step 3: Commpare results from 2 with a second order optimizer / natural gradient
Step 4: Add aleatoric uncertainty i.e. Bayesian neural network
"""
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, hessian, jit


# Second Order Optimizaiton
def rosenbrock(x, a, b):
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def rosenbrock2D(x, y, a, b):
    return (a - x) ** 2 + b * (y - x**2) ** 2


def himmelblau(x, y):
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def himmelblau2D(x):
    return np.log((x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2)


f = partial(rosenbrock2D, a=1, b=100)
g = partial(rosenbrock, a=1, b=100)

f = himmelblau
g = himmelblau2D

x_init = np.random.normal(size=(2,))

x1 = np.linspace(-6.0, 6.0, 200)
x2 = np.linspace(-6.0, 6.0, 200)


X1, X2 = np.meshgrid(x1, x2)
Y = f(X1, X2)

cm = plt.cm.get_cmap("viridis")
plt.scatter(X1, X2, c=Y, cmap=cm)
plt.show()

cp = plt.contour(X1, X2, Y, levels=30)
plt.clabel(cp, inline=1, fontsize=10)
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

levels = [
    100 * x
    for x in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 4.0, 8.0, 12.0, 14.0, 16, 20]
]
cp = plt.contour(
    X1, X2, Y, levels=levels, colors="black", linestyles="dashed", linewidths=1
)
plt.clabel(cp, inline=1, fontsize=10)
cp = plt.contourf(X1, X2, Y, levels=levels)
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


@jit
def update(x, learning_rate):
    grads = grad(g)(x)
    return x - learning_rate * grads


@jit
def update2(x, learning_rate):
    jitter = 1e-12
    grads = grad(g)(x)
    hess = hessian(g)(x)
    L = jnp.linalg.solve(hess + jitter * jnp.eye(x.shape[0]), -grads)
    return x + learning_rate * L


# Newtons Method with adaptive Stepsize
xes = np.zeros((201, 2))
epochs = 200
x_init = np.array([0.0, 0.0])
x = x_init
xes[0, :] = x_init
tol = 1e-8
x_min = np.ones(2)
learning_rate = 0.01
for epoch in range(epochs):
    x_new = update2(x, learning_rate)
    if g(x_new) < g(x):
        x = x_new
        learning_rate = learning_rate**0.5
    else:
        learning_rate = 0.1 * learning_rate
    xes[epoch + 1, :] = x

    if jnp.mean(jnp.abs(x_min - x)) < tol:
        print(f"Minimum at {x} after {epoch} epochs.")
        break

    if epoch % 10 == 0:
        val = g(x)
        print(f"Loss after epoch {epoch}: {val}")

all_ws = xes[:10, :]
levels = [100 * x for x in [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 14.0, 16, 20]]
plt.contourf(X1, X2, Y, levels=levels)
for i in range(len(all_ws) - 1):
    plt.annotate(
        "",
        xy=all_ws[i + 1, :],
        xytext=all_ws[i, :],
        arrowprops={"arrowstyle": "->", "color": "r", "lw": 1},
        va="center",
        ha="center",
    )
# plt.annotate("Min",xy=np.array([1,1]),arrowprops={"arrowstyle": "->", "color": "r", "lw": 1},
#         va="center",
#         ha="center",color="red")
cp = plt.contour(
    X1, X2, Y, levels=levels, colors="black", linestyles="dashed", linewidths=1
)
# plt.plot(1, 1, "x", color="red", markersize=10)
plt.clabel(cp, inline=1, fontsize=10)
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
