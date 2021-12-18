"""
Step 1: Build a simple (deterministic) MLP with JAX and Backpropagation
Step 2: Build a probabilistic MLP handling epistemic unvertainty i.e. learn
the parameters (mu and sigma) of a Gaussian distribution with first order optimization.
Step 3: Commpare results from 2 with a second order optimizer / natural gradient
Step 4: Add aleatoric uncertainty i.e. Bayesian neural network
"""
from datetime import datetime
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import seaborn as sns
from jax import grad, jit, jvp, value_and_grad, vjp, vmap
from jax.nn import leaky_relu, relu, sigmoid, softplus, tanh
from jax.scipy.sparse.linalg import cg


# Step 1: Simple NN
def get_data(N=50, D_X=3, sigma_obs=0.025, N_test=500):
    D_Y = 1  # create 1d outputs
    np.random.seed(0)
    X = jnp.linspace(-1.25, 1, N)
    X = jnp.power(X[:, np.newaxis], jnp.arange(D_X))
    W = 0.5 * np.random.randn(D_X)
    y = jnp.dot(X, W) + 0.5 * jnp.power(0.5 + X[:, 1], 2.0) * jnp.sin(4.0 * X[:, 1])
    y += sigma_obs + (0.5 * X[:, 1]) ** 2 * np.random.randn(N)
    y = y[..., None]
    y -= jnp.mean(y)
    y /= jnp.std(y)

    assert X.shape == (N, D_X)
    assert y.shape == (N, 1)

    X_test = jnp.linspace(-1.75, 1.25, N_test)
    X_test = jnp.power(X_test[:, np.newaxis], jnp.arange(D_X))

    return X, y, X_test


def MLP(layer_dims: list):
    params = []
    for m, n in zip(layer_dims[:-1], layer_dims[1:]):
        W = np.random.normal(size=(m, n))
        b = np.random.normal(size=n)
        params.append([W, b])
    return params


def predict(params, X):

    activations = X
    for (W, b) in params[:-1]:
        output = jnp.dot(activations, W) + b
        activations = leaky_relu(output, 0.05)

    final_W, final_b = params[-1]
    activations = jnp.dot(activations, final_W) + final_b

    return activations


def predict_single(params, x):

    activation = x
    for (W, b) in params[:-1]:
        output = jnp.dot(activation, W) + b
        activation = sigmoid(output)

    final_W, final_b = params[-1]
    activation = jnp.dot(activation, final_W) + final_b

    return activation


batched_predict = vmap(predict_single, in_axes=[None, 0])
# batched_predict(params, X)


@jit
def mse(params, X, y):
    yhat = batched_predict(params, X)
    loss = jnp.mean(jnp.square(yhat - y))
    return loss


def predict_normal(params, X):

    activation = X
    for (W, b) in params[:-1]:
        output = jnp.dot(activation, W) + b
        activation = sigmoid(output)

    final_W, final_b = params[-1]
    activation = jnp.dot(activation, final_W) + final_b

    return activation[:, 0:1], softplus(activation[:, 1:])


@jit
def nll(params, X, y):
    mu, sigma = predict_normal(params, X)
    normal = dist.Normal(loc=mu, scale=sigma)
    ll = jnp.mean(normal.log_prob(y))
    return -ll


def fisher_vp(f, w, v):
    # J v
    _, Jv = jvp(f, (w,), (v,))
    # (J v)^T J = v^T (J^T J)
    _, f_vjp = vjp(f, w)
    return f_vjp(Jv)[0]


@jit
def natural_emp_step(params, X, y, learning_rate):

    loss, grads = value_and_grad(nll)(params, X, y)  # compute gradients
    f = lambda w: nll(w, X, y)  # setup mvp
    fvp = lambda v: fisher_vp(f, params, v)
    ngrads, _ = cg(fvp, grads, maxiter=20)  # approx solve with Conjugate Gradient
    params_new = [
        (W - learning_rate * dW, b - learning_rate * db)
        for (W, b), (dW, db) in zip(params, ngrads)
    ]

    if nll(params_new, X, y) < loss:
        params = params_new
        learning_rate = learning_rate ** 0.5
    else:
        learning_rate = 0.1 * learning_rate

    return loss, params, learning_rate, ngrads


def main(N: int = 50):
    D_X = 4
    # 1 - Predict Mean
    X, y, X_test = get_data(N=N, D_X=D_X)
    params = MLP([D_X, 4, 8, 4, 1])
    epochs = 5000
    learning_rate = 0.1
    loss_fn = mse

    @jit
    def update(params, learning_rate, X, y):
        loss, grads = value_and_grad(loss_fn)(params, X, y)  # compute gradients
        params_new = [
            (W - learning_rate * dW, b - learning_rate * db)
            for (W, b), (dW, db) in zip(params, grads)
        ]
        return loss, params_new

    for epoch in range(epochs):
        loss, params_new = update(params, learning_rate, X, y)

        if loss_fn(params_new, X, y) < loss:
            params = params_new
            learning_rate = learning_rate ** 0.5
        else:
            learning_rate = 0.1 * learning_rate

        if epoch % 1000 == 0:
            print(f"Loss after epoch {epoch}: {loss}")

    yhat = batched_predict(params, X_test)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(X_test[:, 1], yhat.ravel(), "tab:orange")
    ax.plot(X[:, 1], y.ravel(), "x", color="tab:blue", markersize=5)
    # plt.show()

    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig(f"plots/MLP1_{datetime_str}.jpg")

    # 2 - Probabilistic Prediction, Training with Gradient Descent
    X, y, X_test = get_data(N=N, D_X=D_X)
    params = MLP([D_X, 4, 8, 16, 8, 4, 2])
    epochs = 5000
    loss_fn = nll
    learning_rate = 0.5

    @jit  # type: ignore
    def update(params, learning_rate, X, y):
        loss, grads = value_and_grad(loss_fn)(params, X, y)  # compute gradients
        params_new = [
            (W - learning_rate * dW, b - learning_rate * db)
            for (W, b), (dW, db) in zip(params, grads)
        ]
        return loss, params_new

    for epoch in range(epochs):
        loss, params_new = update(params, learning_rate, X, y)

        if loss_fn(params_new, X, y) < loss:
            params = params_new
            learning_rate = learning_rate ** 0.5
        else:
            learning_rate = 0.1 * learning_rate

        if epoch % 1000 == 0:
            print(f"Loss after epoch {epoch}: {loss}")

    mu, sigma = predict_normal(params, X_test)
    mu = mu.ravel()
    sigma = sigma.ravel()
    x_test = X_test[:, 1]
    x = X[:, 1]
    y = y.ravel()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x_test, mu, "tab:orange")
    ax.plot(x, y, "x", color="tab:blue", markersize=5)
    ax.fill_between(
        x_test, mu + 1.96 * sigma, mu - 1.96 * sigma, alpha=0.5, color="green"
    )
    # plt.show()

    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig(f"plots/MLP2_{datetime_str}.jpg")

    # 3 - Probabilistic Prediction, Training with Natural Gradient
    X, y, X_test = get_data(N=N, D_X=D_X)
    params = MLP([D_X, 4, 8, 16, 8, 4, 2])
    epochs = 5000
    learning_rate = 1
    loss_fn = nll

    @jit  # type: ignore
    def update(params, learning_rate, X, y):
        loss, grads = jit(value_and_grad(loss_fn))(params, X, y)  # compute gradients
        f = lambda w: loss_fn(w, X, y)  # setup mvp
        fvp = lambda v: fisher_vp(f, params, v)
        ngrads, _ = cg(fvp, grads, maxiter=20)  # approx solve with Conjugate Gradient
        params = [
            (W - learning_rate * dW, b - learning_rate * db)
            for (W, b), (dW, db) in zip(params, ngrads)
        ]
        return loss, params

    for epoch in range(epochs):
        loss, params_new = update(params, learning_rate, X, y)
        if loss_fn(params_new, X, y) < loss:
            params = params_new
            learning_rate = learning_rate ** 0.5
        else:
            learning_rate = 0.1 * learning_rate

        if epoch % 1000 == 0:
            print(f"Loss after epoch {epoch}: {loss}")

    mu, sigma = predict_normal(params, X_test)
    mu = mu.ravel()
    sigma = sigma.ravel()
    x_test = X_test[:, 1]
    x = X[:, 1]
    y = y.ravel()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x_test, mu, "tab:orange")
    ax.plot(x, y, "x", color="tab:blue", markersize=5)
    ax.fill_between(
        x_test, mu + 1.96 * sigma, mu - 1.96 * sigma, alpha=0.5, color="green"
    )
    # plt.show()

    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig(f"plots/MLP3_{datetime_str}.jpg")


if __name__ == "__main__":
    main()
