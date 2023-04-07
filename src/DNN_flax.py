import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from flax import linen as nn
from jax import random
from numpyro.contrib.module import random_flax_module
from numpyro.diagnostics import hpdi
from numpyro.infer import (
    MCMC,
    NUTS,
    SVI,
    Predictive,
    TraceMeanField_ELBO,
    autoguide,
    init_to_feasible,
)

np.random.seed(0)


class Net(nn.Module):
    n_units: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        mean = nn.Dense(1)(x)
        rho = nn.Dense(1)(x)
        return mean.squeeze(), rho.squeeze()


def get_data2(N=50, D_X=3, sigma_obs=0.05, N_test=500):
    D_Y = 1  # create 1d outputs
    np.random.seed(0)
    X = jnp.linspace(-1, 1, N)
    X = jnp.power(X[:, np.newaxis], jnp.arange(D_X))
    W = 0.5 * np.random.randn(D_X)
    Y = jnp.dot(X, W) + 0.5 * jnp.power(0.5 + X[:, 1], 2.0) * jnp.sin(4.0 * X[:, 1])
    Y += sigma_obs * np.random.randn(N)
    Y = Y[:, np.newaxis]
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)

    X_test = jnp.linspace(-1.3, 1.3, N_test)
    X_test = jnp.power(X_test[:, np.newaxis], jnp.arange(D_X))

    return X, Y.ravel(), X_test


def get_data(N: int = 30, N_test: int = 1000):
    X = jnp.asarray(np.random.uniform(-np.pi * 3 / 2, np.pi, size=(N, 1)))
    y = jnp.asarray(np.sin(X) + np.random.normal(loc=0, scale=0.2, size=(N, 1)))
    X_test = jnp.linspace(-np.pi * 2, 2 * np.pi, num=N_test).reshape((-1, 1))
    return X.ravel(), y.ravel(), X_test.ravel()

    # x = np.random.normal(size=N)
    # y = np.cos(x * 3) + np.random.normal(size=N) * np.abs(x) / 2
    # x_test = jnp.linspace(-np.pi * 2, 2 * np.pi, num=N_test)
    # return x[:, np.newaxis], y, x_test[:, np.newaxis]


def model(x, y=None, batch_size=None):
    module = Net(n_units=16)
    net = random_flax_module("nn", module, dist.Normal(0, 0.1), input_shape=x.shape)
    with numpyro.plate("batch", x.shape[0], subsample_size=batch_size):
        batch_x = numpyro.subsample(x, event_dim=1)
        batch_y = numpyro.subsample(y, event_dim=0) if y is not None else None
        mean, rho = net(batch_x)
        sigma = nn.softplus(rho)
        # assert mean.shape == (256,)
        # assert sigma.shape == (256,)
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=batch_y)


n_train_data = 5000
X, y, X_test = get_data(N=n_train_data)
guide = autoguide.AutoNormal(model, init_loc_fn=init_to_feasible)
svi = SVI(model, guide, numpyro.optim.Adam(5e-3), TraceMeanField_ELBO())

n_iterations = 3000
# svi_result = svi.run(random.PRNGKey(234), n_iterations, x=X, y=y, batch_size=512)
# params, losses = svi_result.params, svi_result.losses
# predictive = Predictive(model, guide=guide, params=params, num_samples=1000)


kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=4000, num_chains=1)
mcmc.run(x=X, y=y, batch_size=256, rng_key=random.PRNGKey(63547901))
predictive = Predictive(model, posterior_samples=mcmc.get_samples(), num_samples=50)


post_samples = predictive(rng_key=random.PRNGKey(1), x=X_test)
yhat = jnp.mean(post_samples["obs"], axis=0)
y_hpdi = hpdi(post_samples["obs"], prob=0.9)
y_05 = y_hpdi[0, ...]
y_95 = y_hpdi[1, ...]
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(
    X[:, 1].ravel(),
    y.ravel(),
    ".",
    color="tab:blue",
    markersize=2,
    label="Observed",
)
ax.plot(X_test[:, 1].ravel(), yhat.ravel(), "tab:orange", label="Mean Prediction")
ax.fill_between(
    X_test[:, 1].ravel(),
    y_05.ravel(),
    y_95.ravel(),
    alpha=0.5,
    color="green",
    label="90%-HPDI",
)
plt.legend(loc="upper right")
plt.show()
