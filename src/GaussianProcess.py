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
        k += (noise + jitter) * jnp.eye(X.shape[0])

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


def GP1(X, y=None):
    N, _ = X.shape

    # Mean-GP
    var_k1 = numpyro.sample("var_k1", dist.LogNormal(0.0, 1.0))
    length_k1 = numpyro.sample("length_k1", dist.LogNormal(0.0, 1.0))
    noise_k1 = numpyro.sample("noise_k1", dist.LogNormal(0.0, 1.0))
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
    intercept = numpyro.sample("intercept", dist.Normal(0, 1))
    slope = numpyro.sample("slope", dist.Normal(0, 1))
    log_scale = numpyro.deterministic(
        "log_scale", jnp.sum(intercept + slope * X, axis=1)
    )
    scale = numpyro.deterministic("scale", jnp.exp(log_scale))

    assert mean.shape == (N,)
    assert scale.shape == (N,)
    if y is not None:
        assert y.shape == (N,)

    # Likelihood
    numpyro.sample("y", dist.Normal(loc=mean, scale=scale), obs=y)


def GP3(X, y=None):
    N, _ = X.shape

    # Mean-GP
    var_k1 = numpyro.sample("var_k1", dist.LogNormal(-1, 1))
    length_k1 = numpyro.sample("length_k1", dist.LogNormal(-1, 1))
    noise_k1 = numpyro.sample("noise_k1", dist.HalfNormal(1))
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
    var_k2 = numpyro.sample("var_k2", dist.LogNormal(-1, 1))
    length_k2 = numpyro.sample("length_k2", dist.LogNormal(-1, 1))
    noise_k2 = numpyro.sample("noise_k2", dist.HalfNormal(1))
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

    k_22 = rbf_kernel(X=X_test, Z=X_test.T, var=var, length=length, noise=noise)
    k_21 = rbf_kernel(
        X=X_test, Z=X.T, include_noise=False, var=var, length=length, noise=noise
    )
    k_11 = rbf_kernel(X=X, Z=X.T, var=var, length=length, noise=noise)
    K_11_inv = jnp.linalg.solve(k_11, jnp.eye(k_11.shape[0]))
    K = k_22 - jnp.matmul(k_21, jnp.matmul(K_11_inv, jnp.transpose(k_21)))
    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jax.random.normal(
        rng_key, X_test.shape[:1]
    )

    mean = jnp.matmul(k_21, jnp.matmul(K_11_inv, Y))
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean, mean + sigma_noise


def get_data(dataset: str = "M", N: int = 100, N_test: int = 200):
    x, y = globals()[f"dataset_{dataset}"]()
    x_test = jnp.asarray(np.linspace(1.2 * np.min(x), 1.2 * np.max(x), N_test))
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
    model_str: str,
):

    X, X_test, y = get_data(dataset=dataset)

    if model_str == "GP1":
        model = GP1
    elif model_str == "GP2":
        model = GP2
    elif model_str == "GP3":
        model = GP3

    nuts_kernel = NUTS(model)
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
    idx2 = jax.random.randint(
        jax.random.PRNGKey(347), shape=(100,), minval=0, maxval=num_samples
    )

    if model_str == "GP1":
        # Predictions from mean GP
        subsamples = map(
            lambda x: x[idx],
            (samples["var_k1"], samples["length_k1"], samples["noise_k1"]),
        )
        vmap_args = (jax.random.split(jax.random.PRNGKey(324), post_samples),) + tuple(
            subsamples
        )
        mu, predictions = jax.vmap(
            lambda rng_key, var, length, noise: predict(
                rng_key, X, y, X_test, var, length, noise
            )
        )(*vmap_args)

        mean = jnp.mean(mu, axis=0)
        hpdi_9 = hpdi(predictions, prob=0.9, axis=0)
        lower = hpdi_9[0, ...]
        upper = hpdi_9[1, ...]
        scale = samples["scale"][idx]
        std = jnp.mean(scale)
    elif model_str == "GP2":
        # Predictions from mean GP
        subsamples = map(
            lambda x: x[idx],
            (samples["var_k1"], samples["length_k1"], samples["noise_k1"]),
        )
        vmap_args = (jax.random.split(jax.random.PRNGKey(324), post_samples),) + tuple(
            subsamples
        )
        mu, predictions = jax.vmap(
            lambda rng_key, var, length, noise: predict(
                rng_key, X, y, X_test, var, length, noise
            )
        )(*vmap_args)

        intercept, slope = map(
            lambda x: x[idx], (samples["intercept"], samples["slope"])
        )
        mean = jnp.mean(mu, axis=0)
        scale = jnp.mean(
            jax.vmap(lambda x: jnp.exp(intercept + slope * x))(X_test), axis=1
        )
        hpdi_9 = hpdi(predictions, prob=0.9, axis=0)
        lower_mean = hpdi_9[0, ...]
        upper_mean = hpdi_9[1, ...]

        upper_std = mean + 2 * scale
        lower_std = mean - 2 * scale

        upper_total = jnp.maximum(upper_mean, upper_std)
        lower_total = jnp.minimum(lower_mean, lower_std)

        # lower_total = hpdi_9[0, ...] - jnp.mean(scale, axis=0)
        # upper_total = hpdi_9[1, ...] + jnp.mean(scale, axis=0)

    elif model_str == "GP3":
        # Predictions from mean GP
        subsamples = map(
            lambda x: x[idx],
            (samples["var_k1"], samples["length_k1"], samples["noise_k1"]),
        )
        vmap_args = (jax.random.split(jax.random.PRNGKey(324), post_samples),) + tuple(
            subsamples
        )
        mu, predictions_mu = jax.vmap(
            lambda rng_key, var, length, noise: predict(
                rng_key, X, y, X_test, var, length, noise
            )
        )(*vmap_args)

        # Predictions from scale GP
        subsamples = map(
            lambda x: x[idx],
            (samples["var_k2"], samples["length_k2"], samples["noise_k2"]),
        )
        vmap_args = (jax.random.split(jax.random.PRNGKey(675), post_samples),) + tuple(
            subsamples
        )
        log_scale, predictions_log_scale = jax.vmap(
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
        scale = jnp.mean(jnp.exp(log_scale), axis=0)

        # combine mean and scale uncertainty
        hpdi_9 = hpdi(predictions_mu, prob=0.9, axis=0)
        lower_mean = hpdi_9[0, ...]
        upper_mean = hpdi_9[1, ...]

        upper_std = mean + 2 * scale
        lower_std = mean - 2 * scale

        upper_total = jnp.maximum(upper_mean, upper_std)
        lower_total = jnp.minimum(lower_mean, lower_std)

    x_test = X_test.ravel()
    x = X.ravel()
    # split plot into
    # - points + samples + mean
    # - points + mean + shaded area
    # - points + mean + scale
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 24))
    ax = axes[0]
    ax.plot(x, y, ".", color="tab:blue", markersize=5, label="obs")
    ax.plot(x_test, mu[idx2, :].T, alpha=0.2)
    ax.plot(x_test, mu[idx2, :].T, alpha=0.2)

    ax = axes[1]
    ax.plot(x, y, ".", color="tab:blue", markersize=5, label="obs")
    ax.plot(x_test, mean, "tab:orange", label="mean")
    ax.fill_between(
        x_test.ravel(),
        lower_mean,
        upper_mean,
        alpha=0.2,
        color="green",
        label="90% - HPDI",
    )
    ax = axes[2]
    ax.plot(x, y, ".", color="tab:blue", markersize=5, label="obs")
    ax.plot(x_test, mean, "tab:orange", label="mean")
    ax.fill_between(
        x_test.ravel(),
        lower_std,
        upper_std,
        alpha=0.2,
        color="green",
        label="mean +/- 2 x std",
    )
    ax = axes[3]
    ax.plot(x, y, ".", color="tab:blue", markersize=5, label="obs")
    ax.plot(x_test, mean, "tab:orange", label="mean")
    ax.fill_between(
        x_test.ravel(),
        lower_total,
        upper_total,
        alpha=0.2,
        color="orange",
        label="combined",
    )

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="lower left")

    fig.tight_layout()
    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig(
        f"plots/experiment3/GaussianProcess{model_str[-1]}_{dataset}.jpg", dpi=300
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", default=50, type=int)
    parser.add_argument("--dataset", default="M", type=str)
    parser.add_argument("--num-warmup", default=1_000, type=int)
    parser.add_argument("--num-samples", default=4_000, type=int)
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("--post-samples", default=100, type=int)
    parser.add_argument("--model", default="GP1", type=str)

    args = parser.parse_args()
    main(
        N=args.sample_size,
        dataset=args.dataset,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        post_samples=args.post_samples,
        model_str=args.model,
    )
