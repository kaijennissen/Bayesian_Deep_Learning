import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import tqdm
from flax import linen as nn
from jax import jit, random
from numpyro.contrib.module import random_flax_module
from numpyro.diagnostics import hpdi
from numpyro.infer import (
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
        x = nn.Dense(self.n_units)(x[..., None])
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        mean = nn.Dense(1)(x)
        rho = nn.Dense(1)(x)
        return mean.squeeze(), rho.squeeze()


def get_data(N: int = 30, N_test: int = 1000):

    X = jnp.asarray(np.random.uniform(-np.pi * 3 / 2, np.pi, size=(N, 1)))
    y = jnp.asarray(np.sin(X) + np.random.normal(loc=0, scale=0.2, size=(N, 1)))
    X_test = jnp.linspace(-np.pi * 2, 2 * np.pi, num=N_test).reshape((-1, 1))
    return X.ravel(), y.ravel(), X_test.ravel()


def model(x, y=None, batch_size=None):
    module = Net(n_units=32)
    net = random_flax_module("nn", module, dist.Normal(0, 0.1), input_shape=())
    with numpyro.plate("batch", x.shape[0], subsample_size=batch_size):
        batch_x = numpyro.subsample(x, event_dim=0)
        batch_y = numpyro.subsample(y, event_dim=0) if y is not None else None
        mean, rho = net(batch_x)
        sigma = nn.softplus(rho)
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=batch_y)


n_train_data = 5000
X, y, X_test = get_data(N=n_train_data)
guide = autoguide.AutoNormal(model, init_loc_fn=init_to_feasible)
svi = SVI(model, guide, numpyro.optim.Adam(5e-3), TraceMeanField_ELBO())

n_iterations = 3000
svi_result = svi.run(random.PRNGKey(0), n_iterations, X, y, batch_size=256)
params, losses = svi_result.params, svi_result.losses

predictive = Predictive(model, guide=guide, params=params, num_samples=1000)


post_samples = predictive(rng_key=random.PRNGKey(1), x=X_test)
yhat = jnp.mean(post_samples["obs"], axis=0)
y_hpdi = hpdi(post_samples["obs"], prob=0.9)
y_05 = y_hpdi[0, :]
y_95 = y_hpdi[1, :]

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(
    X.ravel(),
    y.ravel(),
    ".",
    color="tab:blue",
    markersize=2,
    label="Observed",
)
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
plt.show()
