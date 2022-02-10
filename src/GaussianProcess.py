import argparse
import warnings
from datetime import datetime
from typing import Tuple

import pandas as pd

warnings.simplefilter("ignore", FutureWarning)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import jit, random, vmap
from numpyro import handlers
from numpyro.diagnostics import hpdi
from numpyro.infer import (
    MCMC,
    NUTS,
    Predictive,
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)

numpyro.set_host_device_count(4)

# squared exponential kernel with diagonal noise


def rbf_kernel(
    X, Z, length, var, noise, jitter=1.0e-6, include_noise=True, *args, **kwargs
):

    deltaXsq = jnp.power((X - Z) / length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)

    if include_noise:
        k += (noise + jitter) * np.eye(X.shape[0])

    return k


def periodic_kernel(
    X, Z, var, period, length, noise, jitter=1.0e-6, include_noise=True, *args, **kwargs
):
    deltaX = jnp.pi * jnp.abs(X - Z) / period
    sin_deltaX = jnp.power(jnp.sin(deltaX) / length, 2.0)
    k = var * jnp.exp(-0.5 * sin_deltaX)
    if include_noise:
        k += (noise + jitter) * np.eye(X.shape[0])

    return k


def local_periodic(
    X, Z, var, period, length, noise, jitter=1.0e-6, include_noise=True, *args, **kwargs
):
    k = periodic_kernel(
        X=X,
        Z=Z,
        var=var,
        period=period,
        length=length,
        noise=noise,
        jitter=jitter,
        include_noise=False,
    ) * rbf_kernel(
        X=X,
        Z=Z,
        var=1,
        length=length,
        noise=noise,
        jitter=jitter,
        include_noise=include_noise,
    )

    return k


kernel = rbf_kernel


def GaussianProcess(X, y=None):
    N, _ = X.shape
    var = numpyro.sample("kernel_var", dist.LogNormal(loc=0.0, scale=1.0))
    length = numpyro.sample("kernel_length", dist.LogNormal(loc=0.0, scale=1.0))
    noise = numpyro.sample("kernel_noise", dist.LogNormal(loc=0.0, scale=1.0))
    # period = numpyro.sample("kernel_period", dist.InverseGamma(2, 1))

    k = numpyro.deterministic(
        "k", kernel(X=X, Z=X.T, var=var, length=length, noise=noise)
    )
    numpyro.sample(
        "y", dist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=k), obs=y
    )


def predict(rng_key, X, Y, X_test, var, length, noise):
    # compute kernels between train and test data, etc.

    k_pp = kernel(
        X=X_test, Z=X_test.T, include_noise=True, var=var, length=length, noise=noise
    )
    k_pX = kernel(
        X=X_test, Z=X.T, include_noise=False, var=var, length=length, noise=noise
    )
    k_XX = kernel(X=X, Z=X.T, include_noise=True, var=var, length=length, noise=noise)
    K_xx_inv = jnp.linalg.inv(k_XX)
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * random.normal(
        rng_key, X_test.shape[:1]
    )

    mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean, mean + sigma_noise


def get_data(N: int = 100, N_test: int = 1000):
    # X = jnp.asarray(np.random.uniform(-np.pi * 3 / 2, np.pi * 3 / 2, size=(N, 1)))
    X1 = jnp.asarray(
        np.random.uniform(-np.pi * 3 / 2, -np.pi * 1 / 2, size=(N // 2, 1))
    )
    X2 = jnp.asarray(np.random.uniform(np.pi * 1 / 2, np.pi * 3 / 2, size=(N // 2, 1)))
    X = jnp.vstack([X1, X2])
    y = (
        0.125 * X
        + jnp.asarray(np.sin(X * 2) + np.random.normal(loc=0, scale=0.2, size=(N, 1)))
    ).ravel()
    X_test = jnp.linspace(-3 * np.pi, 3 * np.pi, num=N_test).reshape((-1, 1))

    # sigma_obs = 0.15
    # X = jnp.linspace(-1, 1, N)
    # y = X + 0.2 * jnp.power(X, 3.0) + 0.5 * jnp.power(0.5 + X, 2.0) * jnp.sin(4.0 * X)
    # y += sigma_obs * np.random.randn(N)
    # y -= jnp.mean(y)
    # y /= jnp.std(y)
    # X = X[:, None]

    # X_test = jnp.linspace(-1.3, 1.3, N_test)[:, None]

    df = pd.read_csv("./data/motor.csv")
    X = jnp.asarray(df["times"].values[..., None])
    X_test = jnp.asarray(np.linspace(0, 62, N_test)[..., None])
    y = jnp.asarray(df["accel"].values)

    np.random.seed(89)
    X = np.random.uniform(-np.pi / 2, np.pi / 2, N)
    I = np.random.binomial(n=1, p=0.8, size=N)
    y = (
        np.sin(X * 2)
        + I * np.random.normal(0, 0.1, size=(N))
        + (1 - I) * np.random.normal(0, 3, size=(N))
    )

    X_test = jnp.linspace(-1.3 * (np.pi / 2), (np.pi / 2) * 1.3, N_test)
    return X[:, np.newaxis], X_test[:, np.newaxis], y


def main(N: int, num_warmup: int, num_samples: int, num_chains: int, post_samples: int):

    X, X_test, y = get_data(N=N)

    rng_key, rng_key_predict = random.split(random.PRNGKey(532))
    nuts_kernel = NUTS(GaussianProcess, init_strategy=init_to_median())
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="parallel",
    )
    mcmc.run(X=X, y=y, rng_key=rng_key)
    mcmc.print_summary()
    samples = mcmc.get_samples()

    idx = random.randint(
        random.PRNGKey(12), shape=(post_samples,), minval=0, maxval=num_samples
    )
    subsamples = map(
        lambda x: x[idx],
        (samples["kernel_var"], samples["kernel_length"], samples["kernel_noise"]),
    )
    vmap_args = (random.split(rng_key_predict, post_samples),) + tuple(subsamples)

    means, predictions = vmap(
        lambda rng_key, var, length, noise: predict(
            rng_key, X, y, X_test, var, length, noise
        )
    )(*vmap_args)

    yhat = jnp.mean(means, axis=0)
    y_hpdi = hpdi(predictions, prob=0.5)
    y_25 = y_hpdi[0, :]
    y_75 = y_hpdi[1, :]
    y_hpdi = hpdi(predictions, prob=0.9)
    y_05 = y_hpdi[0, :]
    y_95 = y_hpdi[1, :]

    make_plot(
        train_data=(X, y),
        test_data=(X_test, yhat),
        ci_intervals=[("90", y_05, y_95), ("50", y_25, y_75)],
    )


def make_plot(train_data: Tuple, test_data: Tuple, ci_intervals):
    X, y = train_data
    X_test, yhat = test_data
    colors = ["lightblue", "lightgreen", "green"]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

    ax.plot(X.ravel(), y.ravel(), ".", color="tab:blue", markersize=5, label="obs")
    ax.plot(X_test.ravel(), yhat.ravel(), "tab:orange", label="mean")
    for (ci_level, upper, lower), color in zip(ci_intervals, colors):
        ax.fill_between(
            X_test.ravel(),
            upper.ravel(),
            lower.ravel(),
            alpha=0.3,
            color=color,
            label=ci_level,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Gaussian-Process")
    plt.legend(loc="lower right")
    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig(f"plots/GaussianProcess_{datetime_str}.jpg", dpi=300)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", default=50, type=int)
    parser.add_argument("--num-warmup", default=1_000, type=int)
    parser.add_argument("--num-samples", default=4_000, type=int)
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("--post-samples", default=100, type=int)

    args = parser.parse_args()
    main(
        N=args.sample_size,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        post_samples=args.post_samples,
    )
