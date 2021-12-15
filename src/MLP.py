"""
Step 1: Build a simple (deterministic) MLP with JAX and Backpropagation
Step 2: Build a probabilistic MLP handling epistemic unvertainty i.e. learn
the parameters (mu and sigma) of a Gaussian distribution with first order optimization.
Step 3: Commpare results from 2 with a second order optimizer / natural gradient
Step 4: Add aleatoric uncertainty i.e. Bayesian neural network
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax import grad, hessian, jit, value_and_grad, vmap
from jax.nn import leaky_relu, relu, sigmoid, tanh


# Step 1: Simple NN
def get_data(N=50, D_X=3, sigma_obs=0.025, N_test=500):
    D_Y = 1  # create 1d outputs
    np.random.seed(0)
    X = jnp.linspace(-1, 1, N)
    X = jnp.power(X[:, np.newaxis], jnp.arange(D_X))
    W = 0.5 * np.random.randn(D_X)
    y = jnp.dot(X, W) + 0.5 * jnp.power(0.5 + X[:, 1], 2.0) * jnp.sin(4.0 * X[:, 1])
    y += sigma_obs * np.random.randn(N)
    y = y[..., None]
    y -= jnp.mean(y)
    y /= jnp.std(y)

    assert X.shape == (N, D_X)
    assert y.shape == (N, 1)

    X_test = jnp.linspace(-1.5, 1.5, N_test)
    X_test = jnp.power(X_test[:, np.newaxis], jnp.arange(D_X))

    return X, y, X_test


X, y, X_test = get_data()
# sns.scatterplot(X[:, 1], y.ravel(), markers="x")
# plt.show()


def MLP(layer_dims: list):
    params = []
    for m, n in zip(layer_dims[:-1], layer_dims[1:]):
        W = np.random.normal(size=(m, n))
        b = np.random.normal(size=1)
        params.append([W, b])
    return params


def predict(params, X):

    activations = X
    for (W, b) in params[:-1]:
        output = jnp.dot(activations, W) + b
        activations = sigmoid(output)

    final_W, final_b = params[-1]
    activations = jnp.dot(activations, final_W) + final_b

    return activations


@jit
def loss_fn(params, X, y):
    yhat = predict(params, X)
    loss = jnp.mean(jnp.square(yhat - y))
    return loss


@jit
def update(params, X, y):
    grads = grad(loss_fn)(params, X, y)
    return [
        (W - learning_rate * dW, b - learning_rate * db)
        for (W, b), (dW, db) in zip(params, grads)
    ]


learning_rate = 0.2
params = MLP([3, 4, 8, 4, 1])
epochs = 5000
for epoch in range(epochs):

    params = update(params, X, y)
    if epoch % 100 == 0:
        loss = loss_fn(params, X, y)
        print(f"Loss after epoch {epoch}: {loss}")


yhat = predict(params, X_test)
plt.plot(X_test[:, 1], yhat.ravel())
plt.plot(X[:, 1], y.ravel(), "x")
plt.show()
