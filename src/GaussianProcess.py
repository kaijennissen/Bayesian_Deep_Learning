import argparse
import warnings
from datetime import datetime
from functools import partial

import pandas as pd

warnings.simplefilter("ignore", FutureWarning)
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

numpyro.set_host_device_count(4)

# squared exponential kernel with diagonal noise


def rbf_kernel(X, Z, length, var, noise, jitter=1.0e-6, include_noise=True):

    deltaXsq = jnp.power((X - Z) / length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)

    if include_noise:
        k += (noise + jitter) * np.eye(X.shape[0])

    return k


kernel = jax.jit(partial(rbf_kernel, include_noise=True, jitter=1.0e-6))


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


def GP1(X, y=None):
    N, _ = X.shape

    # Mean-GP
    var_k1 = numpyro.sample("var_k1", dist.HalfNormal(1.0))
    length_k1 = numpyro.sample("length_k1", dist.HalfNormal(1.0))
    noise_k1 = numpyro.sample("noise_k1", dist.HalfNormal(1.0))
    k1 = numpyro.deterministic(
        "k1",
        kernel(
            X=X,
            Z=X.T,
            var=var_k1,
            length=length_k1,
            noise=noise_k1,
            #    include_noise=True,
        ),
    )
    mean = numpyro.sample("mean", dist.MultivariateNormal(jnp.zeros(N), k1))

    scale = numpyro.sample("scale", dist.HalfNormal(1.0))

    assert mean.shape == (N,)
    assert scale.shape == ()
    if y is not None:
        assert y.shape == (N,)

    # Likelihood
    numpyro.sample("y", dist.Normal(loc=mean, scale=scale), obs=y)


def GP2(X, y=None):
    N, _ = X.shape

    # Mean-GP
    var_k1 = numpyro.sample("var_k1", dist.LogNormal(0, 1))
    length_k1 = numpyro.sample("length_k1", dist.LogNormal(0, 1))
    noise_k1 = numpyro.sample("noise_k1", dist.LogNormal(0, 1))
    k1 = numpyro.deterministic(
        "k1",
        kernel(
            X=X,
            Z=X.T,
            var=var_k1,
            length=length_k1,
            noise=noise_k1,
        ),
    )
    mean = numpyro.sample("mean", dist.MultivariateNormal(jnp.zeros(N), k1))

    # Variance-GP
    var_k2 = numpyro.sample("var_k2", dist.LogNormal(0, 1))
    length_k2 = numpyro.sample("length_k2", dist.LogNormal(0, 1))
    noise_k2 = numpyro.sample("noise_k2", dist.LogNormal(0, 1))
    k2 = numpyro.deterministic(
        "k2", kernel(X=X, Z=X.T, var=var_k2, length=length_k2, noise=noise_k2)
    )
    log_scale = numpyro.sample(
        "log_scale",
        dist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=k2),
    )
    scale = numpyro.deterministic("scale", jnp.exp(log_scale))

    assert mean.shape == (N,)
    assert scale.shape == (N,)
    if y is not None:
        assert y.shape == (N,)

    # Likelihood
    numpyro.sample("y", dist.Normal(loc=mean, scale=scale), obs=y)


def predict(rng_key, X, Y, X_test, var, length, noise):
    # compute kernels between train and test data, etc.

    k_pp = rbf_kernel(X=X_test, Z=X_test.T, var=var, length=length, noise=noise)
    k_pX = rbf_kernel(
        X=X_test, Z=X.T, include_noise=False, var=var, length=length, noise=noise
    )
    k_XX = rbf_kernel(X=X, Z=X.T, var=var, length=length, noise=noise)
    K_xx_inv = jnp.linalg.solve(k_XX, jnp.eye(k_XX.shape[0]))
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jax.random.normal(
        rng_key, X_test.shape[:1]
    )

    mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean, mean + sigma_noise, K


def get_data(dataset: str = "M", N: int = 100, N_test: int = 200):
    x, y = globals()[f"dataset_{dataset}"]()
    x_test = jnp.asarray(np.linspace(1.1 * np.min(x), 1.1 * np.max(x), N_test))
    return x[:, np.newaxis], x_test[:, np.newaxis], y


def dataset_M(N: int = 100):
    df = pd.read_csv("./data/motor.csv")
    x = jnp.asarray(df["times"])
    y = jnp.asarray(df["accel"].values)

    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    return x, y


def dataset_G(N: int = 100):
    # Goldberg et al. 1998
    x = np.random.uniform(0, 1, N)
    mean = 2 * np.sin(2 * np.pi * x)
    sd = 0.5 + x
    y = mean + sd * np.random.normal(size=N)
    return x, y


def dataset_Y(N: int = 100):
    # Yuan and Wahba 2004
    x = np.random.uniform(0, 1, N)
    mean = 2 * (np.exp(-30 * (x - 0.25) ** 2) + np.sin(np.pi * x**2)) - 2
    sd = np.sin(2 * np.pi * x)
    y = mean + np.exp(sd) * np.random.normal(size=N)
    return x, y


def dataset_W(N: int = 200):
    # Williams 1996
    x = np.random.uniform(0, np.pi, N)
    mean = np.sin(2.5 * x) * np.sin(1.5 * x)
    sd = 0.01 + 0.25 * (1 - np.sin(2.5 * x)) ** 2
    y = mean + np.exp(sd) * np.random.normal(size=N)
    return x, y


def dataset_L(N: int = 100):
    df = pd.read_csv("data/lidar.csv")
    x = df.range.values
    y = df.logratio.values
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    return x, y


def main(
    N: int,
    dataset: str,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    post_samples: int,
):

    X, X_test, y = get_data(dataset=dataset)

    nuts_kernel = NUTS(GP2, init_strategy=init_to_median())
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="parallel",
        jit_model_args=True,
    )
    mcmc.run(X=X, y=y, rng_key=jax.random.PRNGKey(532))
    mcmc.print_summary()
    samples = mcmc.get_samples()
    idx = jax.random.randint(
        jax.random.PRNGKey(12), shape=(post_samples,), minval=0, maxval=num_samples
    )
    subsamples = map(
        lambda x: x[idx],
        (samples["var_k1"], samples["length_k1"], samples["noise_k1"]),
    )
    vmap_args = (jax.random.split(jax.random.PRNGKey(324), post_samples),) + tuple(
        subsamples
    )
    mu, predictions, K1 = jax.vmap(
        lambda rng_key, var, length, noise: predict(
            rng_key, X, y, X_test, var, length, noise
        )
    )(*vmap_args)

    subsamples = map(
        lambda x: x[idx],
        (samples["var_k2"], samples["length_k2"], samples["noise_k2"]),
    )
    vmap_args = (jax.random.split(jax.random.PRNGKey(675), post_samples),) + tuple(
        subsamples
    )
    log_scale, predictions, K2 = jax.vmap(
        lambda rng_key, var, length, noise: predict(
            rng_key,
            X,
            jnp.mean(samples["log_scale"][idx], axis=0),
            X_test,
            var,
            length,
            noise,
        )
    )(*vmap_args)
    # TODO: make predicions
    mean = jnp.mean(mu, axis=0)
    std = jnp.exp(jnp.mean(log_scale, axis=0))
    # plt.plot(X.ravel(), y, ".", color="tab:blue", markersize=5, label="obs")
    # plt.plot(
    #     X.ravel(),
    #     y + jnp.exp(jnp.mean(samples["log_scale"][idx], axis=0)),
    #     "r--",
    #     color="tab:blue",
    # )
    # plt.plot(
    #     X.ravel(),
    #     y - jnp.exp(jnp.mean(samples["log_scale"][idx], axis=0)),
    #     "r--",
    #     color="tab:blue",
    # )
    # plt.show()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

    ax.plot(X.ravel(), y.ravel(), ".", color="tab:blue", markersize=5, label="obs")
    ax.plot(X_test.ravel(), jnp.mean(mu, axis=0), "tab:orange", label="mean")
    ax.fill_between(
        X_test.ravel(),
        mean + std,
        mean - std,
        alpha=0.4,
        color="green",
        label="mean +/- std",
    )
    ax.fill_between(
        X_test.ravel(),
        mean + 2 * std,
        mean - 2 * std,
        alpha=0.2,
        color="green",
        label="mean +/- 2 x std",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Gaussian-Process")
    plt.legend(loc="lower right")

    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig(f"plots/GaussianProcess_{dataset}.jpg", dpi=300)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", default=50, type=int)
    parser.add_argument("--dataset", default="M", type=str)
    parser.add_argument("--num-warmup", default=1_000, type=int)
    parser.add_argument("--num-samples", default=4_000, type=int)
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("--post-samples", default=100, type=int)

    args = parser.parse_args()
    main(
        N=args.sample_size,
        dataset=args.dataset,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        post_samples=args.post_samples,
    )
