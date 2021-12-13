import warnings

warnings.simplefilter("ignore", FutureWarning)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, Predictive

LAYER_DIMS = [5, 5]


def nonlin(x):
    return jnp.tanh(x)


def BayesianDNN(X, y=None):

    N, feature_dim = X.shape
    _, out_dim = 1, 1  # y.shape
    layer1_dim = 5
    layer2_dim = 5

    # layer 1
    W1 = numpyro.sample(
        "layer1",
        dist.Normal(
            loc=jnp.zeros((feature_dim, layer1_dim)),
            scale=jnp.ones((feature_dim, layer1_dim)),
        ),
    )
    out1 = nonlin(jnp.matmul(X, W1))

    # layer 2
    W2 = numpyro.sample(
        "layer2",
        dist.Normal(
            loc=jnp.zeros((layer1_dim, layer2_dim)),
            scale=jnp.ones((layer1_dim, layer2_dim)),
        ),
    )
    out2 = nonlin(jnp.matmul(out1, W2))
    # output layer
    W3 = numpyro.sample(
        "out_layer",
        dist.Normal(
            loc=jnp.zeros((layer2_dim, out_dim)), scale=jnp.ones((layer2_dim, out_dim))
        ),
    )

    mean = numpyro.deterministic("mean", jnp.matmul(out2, W3))
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    scale = 1.0 / jnp.sqrt(prec_obs)

    numpyro.sample("y", dist.Normal(loc=mean, scale=scale), obs=y)


N = 30
X = jnp.asarray(np.random.uniform(-np.pi * 3 / 2, np.pi, size=(N, 1)))
y = jnp.asarray(2 * np.sin(X) + np.random.normal(loc=0, scale=0.2, size=(N, 1)))

rng_key = random.PRNGKey(125)
kernel = NUTS(BayesianDNN)
mcmc = MCMC(
    kernel, num_warmup=1000, num_samples=4000, num_chains=2, chain_method="vectorized"
)
mcmc.run(X=X, y=y, rng_key=rng_key)
mcmc.print_summary()

X_test = jnp.linspace(-6, 8, num=1000).reshape((-1, 1))
predictive = Predictive(
    BayesianDNN, posterior_samples=mcmc.get_samples(), num_samples=500, parallel=True
)
post_samples = predictive(rng_key=rng_key, X=X_test)
yhat = jnp.mean(post_samples["y"], axis=0)
print(yhat.shape)
y_hpdi = hpdi(post_samples["y"], prob=0.9)
y_05 = y_hpdi[0, :, :]
y_95 = y_hpdi[1, :, :]

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(X.ravel(), y.ravel(), "x", color="tab:blue", markersize=5)
ax.plot(X_test.ravel(), yhat.ravel(), "tab:orange")
ax.fill_between(X_test.ravel(), y_05.ravel(), y_95.ravel(), alpha=0.5, color="green")
plt.show()
