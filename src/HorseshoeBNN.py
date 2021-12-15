import argparse
import warnings
from datetime import datetime

from jax._src.numpy.lax_numpy import concatenate

warnings.simplefilter("ignore", FutureWarning)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import seaborn as sns
from jax import jit, random
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, Predictive
from scipy.special import expit

# https://arxiv.org/pdf/1905.02599.pdf


@jit
def nonlin(x):
    return jnp.tanh(x)


def HorseshoeBNN(X, y=None):

    N, feature_dim = X.shape
    out_dim = 1
    layer1_dim = 4
    layer2_dim = 4

    # layer 1
    lambdas = numpyro.sample(
        "lambdas", dist.InverseGamma(jnp.ones((feature_dim, 1)) / 2, 2)
    )
    rate_tau = numpyro.deterministic("rate_tau", 1 / lambdas)
    tau = numpyro.sample(
        "tau", dist.InverseGamma(jnp.ones((feature_dim, 1)) / 2, rate_tau)
    )

    rho = numpyro.sample("rho", dist.InverseGamma(1 / 2, 2))
    rate_nu = numpyro.deterministic("rate_tau", 1 / rho)
    nu = numpyro.sample("nu", dist.InverseGamma(1 / 2, rate_nu))

    # direct parametrization
    # tau = numpyro.sample("tau", dist.HalfCauchy(scale=jnp.ones((feature_dim, 1))))
    # lambdas = numpyro.sample("lambdas", dist.HalfCauchy(scale=1.0))
    scale = numpyro.deterministic("scale", tau * nu)

    W1 = numpyro.sample(
        "W1",
        dist.Normal(
            loc=jnp.zeros((feature_dim, layer1_dim)),
            scale=scale,
        ),
    )
    b1 = numpyro.sample("b1", dist.Normal(loc=0.0, scale=1.0))
    out1 = nonlin(jnp.matmul(X, W1)) + b1

    # layer 2
    W2 = numpyro.sample(
        "W2",
        dist.Normal(
            loc=jnp.zeros((layer1_dim, layer2_dim)),
            scale=jnp.ones((layer1_dim, layer2_dim)),
        ),
    )
    b2 = numpyro.sample("b2", dist.Normal(loc=0.0, scale=1.0))
    out2 = nonlin(jnp.matmul(out1, W2)) + b2

    # output layer
    W3 = numpyro.sample(
        "out_layer",
        dist.Normal(
            loc=jnp.zeros((layer2_dim, out_dim)), scale=jnp.ones((layer2_dim, out_dim))
        ),
    )
    b3 = numpyro.sample("b3", dist.Normal(loc=0.0, scale=1.0))

    mean = numpyro.deterministic("mean", jnp.matmul(out2, W3) + b3)
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    scale = 1.0 / jnp.sqrt(prec_obs)

    numpyro.sample(
        "y", dist.Normal(loc=jnp.squeeze(mean), scale=scale).to_event(1), obs=y
    )


def get_data(N=50, D_X=3, sigma_obs=0.05, response="continuous"):
    assert response in ["continuous", "binary"]
    assert D_X >= 3

    np.random.seed(0)
    X = np.random.randn(N, D_X)

    # the response only depends on X_0, X_1, and X_2
    W = np.array([2.0, -1.0, 0.50])
    y = jnp.dot(X[:, :3], W)
    y -= jnp.mean(y)

    if response == "continuous":
        y += sigma_obs * np.random.randn(N)
    elif response == "binary":
        y = np.random.binomial(1, expit(y))

    assert X.shape == (N, D_X)
    assert y.shape == (N,)

    return X, y


def make_plot(X, y, X_test, yhat, y_05, y_95):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(X.ravel(), y.ravel(), "x", color="tab:blue", markersize=5)
    ax.plot(X_test.ravel(), yhat.ravel(), "tab:orange")
    ax.fill_between(
        X_test.ravel(), y_05.ravel(), y_95.ravel(), alpha=0.5, color="green"
    )

    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig(f"plots/HorseshoeBNN_{datetime_str}.jpg")


def main(
    N: int = 100, num_warmup: int = 1000, num_samples: int = 4000, num_chains: int = 2
):

    X, y = get_data(N=N, D_X=5)

    rng_key, rng_key_predict = random.split(random.PRNGKey(915))
    kernel = NUTS(HorseshoeBNN)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
    )
    mcmc.run(X=X, y=y, rng_key=rng_key)
    samples = mcmc.get_samples()

    W1_post = jnp.squeeze(samples["W1"])
    d = W1_post.shape[1]

    W1_norm = jnp.sum(jnp.square(W1_post), axis=-1)
    fig, axes = plt.subplots(nrows=d, figsize=(24, 12), sharey=False, sharex=False)
    for i in range(d):
        ax = axes[i]
        ax.hist(W1_norm[:, i], bins=20)
    # plt.show()

    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig(f"plots/HorseshoeBNN_{datetime_str}.jpg")


if __name__ == "__main__":
    main()
