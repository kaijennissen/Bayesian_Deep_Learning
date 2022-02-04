import argparse
import warnings
from datetime import datetime
from re import A

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


# TODO: inculde priors in Docstring
def HorseshoeBNN(X, y=None):

    N, feature_dim = X.shape
    out_dim = 1
    layer1_dim = 4
    layer2_dim = 4

    # local shrinkage params
    lam = numpyro.sample(
        "lambdas",
        dist.InverseGamma(concentration=0.5 * jnp.ones((feature_dim, 1)), rate=0.25),
    )
    rate_tau = numpyro.deterministic("rate_tau", 1 / lam)
    tau = numpyro.sample(
        "tau",
        dist.InverseGamma(
            concentration=0.5 * jnp.ones((feature_dim, 1)), rate=rate_tau
        ),
    )

    # global shrinkage params
    mu = numpyro.sample("mu", dist.InverseGamma(concentration=0.5, rate=0.25))
    rate_nu = numpyro.deterministic("rate_nu", 1 / mu)
    nu = numpyro.sample("nu", dist.InverseGamma(concentration=0.5, rate=rate_nu))

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
    b1 = numpyro.sample("b1", dist.Normal(loc=jnp.zeros(layer1_dim), scale=1.0))
    out1 = nonlin(jnp.matmul(X, W1)) + b1

    # layer 2
    W2 = numpyro.sample(
        "W2",
        dist.Normal(
            loc=jnp.zeros((layer1_dim, layer2_dim)),
            scale=jnp.ones((layer1_dim, layer2_dim)),
        ),
    )
    b2 = numpyro.sample("b2", dist.Normal(loc=jnp.zeros(layer2_dim), scale=1.0))
    out2 = nonlin(jnp.matmul(out1, W2)) + b2

    # output layer
    W3 = numpyro.sample(
        "out_layer",
        dist.Normal(
            loc=jnp.zeros((layer2_dim, out_dim)), scale=jnp.ones((layer2_dim, out_dim))
        ),
    )
    b3 = numpyro.sample("b3", dist.Normal(loc=jnp.zeros(out_dim), scale=1.0))

    mean = numpyro.deterministic("mean", jnp.matmul(out2, W3) + b3)
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    scale = 1.0 / jnp.sqrt(prec_obs)
    assert mean.shape == (N, 1)
    assert scale.shape == ()
    if y is not None:
        assert y.shape == (N, 1)

    numpyro.sample("y", dist.Normal(loc=mean, scale=scale), obs=y)


def FinnishHorseshoeBNN(X, y=None):

    N, feature_dim = X.shape
    out_dim = 1
    layer1_dim = 4
    layer2_dim = 4

    # local shrinkage params
    b0_sq = numpyro.param("b0_sq", 4)
    alpha = numpyro.sample(
        "alpha",
        dist.InverseGamma(
            concentration=0.5 * jnp.ones((feature_dim, 1)), rate=1 / b0_sq
        ),
    )
    rate_tau = numpyro.deterministic("rate_tau", 1 / alpha)
    tau = numpyro.sample(
        "tau",
        dist.InverseGamma(
            concentration=0.5 * jnp.ones((feature_dim, 1)), rate=rate_tau
        ),
    )
    tau_sq = numpyro.deterministic("tau_sq", jnp.square(tau))

    # global shrinkage params
    bg_sq = numpyro.param("bg_sq", 1)
    mu = numpyro.sample("mu", dist.InverseGamma(concentration=0.5, rate=1 / bg_sq))
    rate_lam = numpyro.deterministic("rate_lam", 1 / mu)
    lamb = numpyro.sample("lamb", dist.InverseGamma(concentration=0.5, rate=rate_lam))
    lamb_sq = numpyro.deterministic("lamb_sq", jnp.square(lamb))

    df = numpyro.param("df", 100)
    s = numpyro.param("s", 10)
    a = numpyro.deterministic("a", 0.5 * df)
    b = numpyro.deterministic("b", 0.5 * (df * s**2))
    c_sq = numpyro.sample("c_sq", dist.InverseGamma(concentration=a, rate=b))

    scale = numpyro.deterministic(
        "scale", jnp.sqrt((c_sq * tau_sq) / (c_sq + lamb_sq * tau_sq))
    )

    W1 = numpyro.sample(
        "W1",
        dist.Normal(
            loc=jnp.zeros((feature_dim, layer1_dim)),
            scale=scale,
        ),
    )
    b1 = numpyro.sample("b1", dist.Normal(loc=jnp.zeros(layer1_dim), scale=1.0))
    out1 = nonlin(jnp.matmul(X, W1)) + b1

    # layer 2
    W2 = numpyro.sample(
        "W2",
        dist.Normal(
            loc=jnp.zeros((layer1_dim, layer2_dim)),
            scale=jnp.ones((layer1_dim, layer2_dim)),
        ),
    )
    b2 = numpyro.sample("b2", dist.Normal(loc=jnp.zeros(layer2_dim), scale=1.0))
    out2 = nonlin(jnp.matmul(out1, W2)) + b2

    # output layer
    W3 = numpyro.sample(
        "out_layer",
        dist.Normal(
            loc=jnp.zeros((layer2_dim, out_dim)), scale=jnp.ones((layer2_dim, out_dim))
        ),
    )
    b3 = numpyro.sample("b3", dist.Normal(loc=jnp.zeros(out_dim), scale=1.0))

    mean = numpyro.deterministic("mean", jnp.matmul(out2, W3) + b3)
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    scale = 1.0 / jnp.sqrt(prec_obs)
    assert mean.shape == (N, 1)
    assert scale.shape == ()
    if y is not None:
        assert y.shape == (N, 1)
    numpyro.sample("y", dist.Normal(loc=mean, scale=scale), obs=y)


def get_data(N=50, D_X=3, sigma_obs=0.05, N_test=500):
    D_Y = 1  # create 1d outputs
    np.random.seed(0)
    X = jnp.linspace(-1, 1, N)
    X = jnp.power(X[:, np.newaxis], jnp.arange(3))
    W = 0.5 * np.random.randn(3)
    if D_X > 3:
        W = np.append(W, np.zeros(D_X - 3))
        X = jnp.hstack([X, np.random.normal(size=(N, D_X - 3))])
    Y = jnp.dot(X, W) + 0.5 * jnp.power(0.5 + X[:, 1], 2.0) * jnp.sin(4.0 * X[:, 1])

    Y += 0.5 * sigma_obs * np.random.randn(N) + 2 * sigma_obs * np.abs(
        Y.ravel() - np.max(Y)
    ) * np.random.randn(N)
    Y = Y[:, np.newaxis]
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)
    # plt.plot(X[:, 1], Y.ravel(), "x")
    # plt.show()
    # raise ValueError
    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)

    X_test = jnp.linspace(-1.3, 1.3, N_test)
    X_test = jnp.power(X_test[:, np.newaxis], jnp.arange(D_X))

    return X, Y, X_test


def make_weights_plot(weights):

    d = weights.shape[1]

    weights_norm = jnp.sum(jnp.square(weights), axis=-1)
    fig, axes = plt.subplots(nrows=d, figsize=(24, 12), sharey=False, sharex=False)
    for i in range(d):
        ax = axes[i]
        ax.hist(weights_norm[:, i], bins=40, density=True)
    # plt.show()

    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig(f"plots/HorseshoeBNN_weights_{datetime_str}.jpg")


def make_predictive_plot(X, y, X_test, yhat, y_05, y_95):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(X[:, 1].ravel(), y.ravel(), "x", color="tab:blue", markersize=5)
    ax.plot(X_test[:, 1].ravel(), yhat.ravel(), "tab:orange")
    ax.fill_between(
        X_test[:, 1].ravel(), y_05.ravel(), y_95.ravel(), alpha=0.5, color="green"
    )

    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig(f"plots/HorseshoeBNN_{datetime_str}.jpg")


def main(
    N: int = 50, num_warmup: int = 1000, num_samples: int = 4000, num_chains: int = 1
):

    model = FinnishHorseshoeBNN
    X, y, X_test = get_data(N=N, D_X=10)

    rng_key, rng_key_predict = random.split(random.PRNGKey(915))
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
    )
    mcmc.run(X=X, y=y, rng_key=rng_key)
    samples = mcmc.get_samples()

    make_weights_plot(jnp.squeeze(samples["W1"]))
    predictive = Predictive(
        model,
        posterior_samples=samples,
        num_samples=500,
        parallel=True,
    )
    post_samples = predictive(rng_key=rng_key_predict, X=X_test)
    yhat = jnp.mean(post_samples["y"], axis=0)

    y_hpdi = hpdi(post_samples["y"], prob=0.9)
    y_05 = y_hpdi[0, :]
    y_95 = y_hpdi[1, :]

    make_predictive_plot(X=X, y=y, X_test=X_test, yhat=yhat, y_05=y_05, y_95=y_95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--sample-size", default=50, type=int)
    parser.add_argument("--num-warmup", default=1000, type=int)
    parser.add_argument("--num-samples", default=4000, type=int)
    parser.add_argument("--num-chains", default=1, type=int)

    args = parser.parse_args()
    main(
        N=args.sample_size,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
    )
