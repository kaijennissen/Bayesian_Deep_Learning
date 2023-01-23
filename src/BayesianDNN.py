import argparse
import warnings
from datetime import datetime

warnings.simplefilter("ignore", FutureWarning)
import matplotlib.pyplot as plt
import numpy as np
import numpyro

numpyro.set_host_device_count(2)
import jax.numpy as jnp
import numpyro.distributions as dist
from flax import linen as nn
from jax import jit, random
from numpyro.contrib.module import random_flax_module
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, Predictive


@jit
def nonlin(x):
    return jnp.tanh(x)


def MNDNN(X, y=None):

    N, feature_dim = X.shape
    out_dim = 1
    layer1_dim = 4
    layer2_dim = 4

    # layer 1
    W1 = numpyro.sample(
        "W1",
        dist.MatrixNormal(
            loc=jnp.zeros((feature_dim, layer1_dim)),
            scale_tril_row=jnp.eye(feature_dim),
            scale_tril_column=jnp.eye(layer1_dim),
        ),
    )
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(layer1_dim), 1.0))
    out1 = nonlin(jnp.matmul(X, W1) + b1)

    # layer 2
    W2 = numpyro.sample("W2", dist.Normal(jnp.zeros((layer1_dim, layer2_dim)), 1.0))
    b2 = numpyro.sample("b2", dist.Normal(jnp.zeros(layer1_dim), 1.0))
    out2 = nonlin(jnp.matmul(out1, W2) + b2)

    # output layer
    W3 = numpyro.sample("out_layer", dist.Normal(jnp.zeros((layer2_dim, out_dim)), 1.0))
    b3 = numpyro.sample("b3", dist.Normal(jnp.zeros(out_dim), 1.0))

    mean = numpyro.deterministic("mean", jnp.matmul(out2, W3) + b3)
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    scale = 1.0 / jnp.sqrt(prec_obs)

    assert mean.shape == (N, 1)
    assert scale.shape == ()
    if y is not None:
        assert y.shape == (N,)

    numpyro.sample("y", dist.Normal(loc=jnp.squeeze(mean), scale=scale), obs=y)


def GaussianBNN(X, y=None):

    N, feature_dim = X.shape
    out_dim = 1
    layer1_dim = 4
    layer2_dim = 4

    # layer 1
    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((feature_dim, layer1_dim)), 1.0))
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(layer1_dim), 1.0))
    out1 = nonlin(jnp.matmul(X, W1) + b1)

    # layer 2
    W2 = numpyro.sample("W2", dist.Normal(jnp.zeros((layer1_dim, layer2_dim)), 1.0))
    b2 = numpyro.sample("b2", dist.Normal(jnp.zeros(layer1_dim), 1.0))
    out2 = nonlin(jnp.matmul(out1, W2) + b2)

    # output layer
    W3 = numpyro.sample("out_layer", dist.Normal(jnp.zeros((layer2_dim, out_dim)), 1.0))
    b3 = numpyro.sample("b3", dist.Normal(jnp.zeros(out_dim), 1.0))

    mean = numpyro.deterministic("mean", jnp.matmul(out2, W3) + b3)
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    scale = 1.0 / jnp.sqrt(prec_obs)

    assert mean.shape == (N, 1)
    assert scale.shape == ()
    if y is not None:
        assert y.shape == (N,)

    numpyro.sample("y", dist.Normal(loc=jnp.squeeze(mean), scale=scale), obs=y)


class DNN(nn.Module):

    n_units: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_units)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.tanh(x)
        mean = nn.Dense(1)(x)
        return mean.squeeze()


def FlaxDNN(X, y=None):
    N, k = X.shape
    module = DNN(4)
    net = random_flax_module("nn", module, dist.Normal(0.0, 1.0), input_shape=(1, k))

    mean = net(X)

    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    scale = 1.0 / jnp.sqrt(prec_obs)

    assert mean.shape == (N,)
    if y is not None:
        assert y.shape == (N,)

    numpyro.sample("y", dist.Normal(mean, scale), obs=y)


def get_data(N: int = 50, N_test: int = 1000):
    np.random.seed(333)
    X = jnp.asarray(np.random.uniform(-np.pi * 3 / 2, np.pi, size=(N,)))
    y = jnp.asarray(np.sin(X) + np.random.normal(loc=0, scale=0.2, size=(N,)))
    X_test = jnp.linspace(-np.pi * 2, 2 * np.pi, num=N_test)
    return X[:, np.newaxis], y, X_test[:, np.newaxis]


def make_plot(X, y, X_test, yhat, y_05, y_95):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(X.ravel(), y.ravel(), ".", color="tab:blue", markersize=2, label="Observed")
    ax.plot(X_test.ravel(), yhat.ravel(), "tab:orange", label="Mean Prediction")
    ax.fill_between(
        X_test.ravel(),
        y_05.ravel(),
        y_95.ravel(),
        alpha=0.5,
        color="green",
        label="90%-HPDI",
    )
    plt.legend(loc="upper right")

    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig(f"plots/BayesianDNN_{datetime_str}.jpg")


def main(
    N: int = 50, num_warmup: int = 1000, num_samples: int = 4000, num_chains: int = 2
):

    X, y, X_test = get_data(N=N)

    model = MNDNN
    rng_key, rng_key_predict = random.split(random.PRNGKey(915))
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="parallel",
    )
    mcmc.run(X=X, y=y, rng_key=rng_key)
    mcmc.print_summary()

    predictive = Predictive(
        model,
        posterior_samples=mcmc.get_samples(),
        num_samples=500,
        parallel=True,
    )
    post_samples = predictive(rng_key=rng_key_predict, X=X_test)
    yhat = jnp.mean(post_samples["y"], axis=0)

    y_hpdi = hpdi(post_samples["y"], prob=0.9)
    y_05 = y_hpdi[0, ...]
    y_95 = y_hpdi[1, ...]

    make_plot(X=X, y=y, X_test=X_test, yhat=yhat, y_05=y_05, y_95=y_95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num-samples", default=50, type=int)
    args = parser.parse_args()
    main(N=args.num_samples)
