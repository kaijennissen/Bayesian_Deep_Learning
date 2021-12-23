import argparse
import warnings
from datetime import datetime

warnings.simplefilter("ignore", FutureWarning)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import jit, random
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, Predictive


@jit
def nonlin(x):
    return jnp.tanh(x)


# def BayesianDNN(X, y=None, layer_dims=[4, 4]):

#     N, feature_dim = X.shape
#     _, out_dim = 1, 1  # y.shape
#     layer_dims.insert(0, feature_dim)
#     layer_dims.append(out_dim)
#     activation = X
#     weights = [None]*3
#     bias = [None]*3

#     for i, (m, n) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
#         weights[i]=numpyro.sample(
#                 f"W{i}",
#                 dist.Normal(
#                     loc=np.zeros((m, n)),
#                     scale=np.ones((m, n)),
#                 ),
#             )

#         bias[i]=numpyro.sample(f"b{i}", dist.Normal(loc=0.0, scale=1.0))
#         activation = (jnp.matmul(activation, weights[i])) + bias[i]

#     prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
#     scale = 1.0 / jnp.sqrt(prec_obs)

#     numpyro.sample("y", dist.Normal(loc=activation, scale=scale), obs=y)


def GaussianBNN(X, y=None):

    N, feature_dim = X.shape
    out_dim = 1
    layer1_dim = 4
    layer2_dim = 4

    # layer 1
    W1 = numpyro.sample(
        "W1",
        dist.Normal(
            loc=jnp.zeros((feature_dim, layer1_dim)),
            scale=jnp.ones((feature_dim, layer1_dim)),
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

    numpyro.sample("y", dist.Normal(loc=mean, scale=scale), obs=y)


def get_data(N: int = 100):

    # X = jnp.asarray(np.random.uniform(-np.pi * 3 / 2, np.pi * 3 / 2, size=(N, 1)))
    X1 = jnp.asarray(
        np.random.uniform(-np.pi * 3 / 2, -np.pi * 1 / 2, size=(N // 2, 1))
    )
    X2 = jnp.asarray(np.random.uniform(np.pi * 1 / 2, np.pi * 3 / 2, size=(N // 2, 1)))
    X = jnp.vstack([X1, X2])
    y = 0.125 * X + jnp.asarray(
        2 * np.sin(X * 4) + np.random.normal(loc=0, scale=0.2, size=(N, 1))
    )
    X_test = jnp.linspace(-np.pi * 2, 2 * np.pi, num=1000).reshape((-1, 1))
    # plt.plot(X.ravel(), y.ravel(), "x", color="tab:blue", markersize=5)
    # plt.show()


def make_plot(X, y, X_test, yhat, y_05, y_95):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(X.ravel(), y.ravel(), "x", color="tab:blue", markersize=5)
    ax.plot(X_test.ravel(), yhat.ravel(), "tab:orange")
    ax.fill_between(
        X_test.ravel(), y_05.ravel(), y_95.ravel(), alpha=0.5, color="green"
    )

    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig(f"plots/BayesianDNN_{datetime_str}.jpg")


def main(
    N: int = 1000, num_warmup: int = 1000, num_samples: int = 4000, num_chains: int = 2
):

    X, y, X_test = get_data(N=N)

    rng_key, rng_key_predict = random.split(random.PRNGKey(915))
    kernel = NUTS(GaussianBNN)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
    )
    mcmc.run(X=X, y=y, rng_key=rng_key)

    predictive = Predictive(
        GaussianBNN,
        posterior_samples=mcmc.get_samples(),
        num_samples=500,
        parallel=True,
    )
    post_samples = predictive(rng_key=rng_key_predict, X=X_test)
    yhat = jnp.mean(post_samples["y"], axis=0)

    y_hpdi = hpdi(post_samples["y"], prob=0.9)
    y_05 = y_hpdi[0, :, :]
    y_95 = y_hpdi[1, :, :]

    make_plot(X=X, y=y, X_test=X_test, yhat=yhat, y_05=y_05, y_95=y_95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument()
