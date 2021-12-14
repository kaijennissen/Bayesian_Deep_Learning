import warnings
from datetime import datetime

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

# squared exponential kernel with diagonal noise


def rbf_kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):

    deltaXsq = jnp.power((X - Z) / length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)

    if include_noise:
        k += (noise + jitter) * np.eye(X.shape[0])

    return k


def periodic_kernel(
    X, Z, var, period, length, noise, jitter=1.0e-6, include_noise=True
):
    deltaX = jnp.pi * jnp.abs(X - Z) / period
    sin_deltaX = jnp.power(jnp.sin(deltaX) / length, 2.0)
    k = var * jnp.exp(-0.5 * sin_deltaX)
    if include_noise:
        k += (noise + jitter) * np.eye(X.shape[0])

    return k


def local_periodic(X, Z, var, period, length, noise, jitter=1.0e-6, include_noise=True):
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


kernel = local_periodic


def GaussianProcess(X, y=None):
    N, _ = X.shape
    var = numpyro.sample("kernel_var", dist.LogNormal(loc=0.0, scale=2.0))
    length = numpyro.sample("kernel_length", dist.LogNormal(loc=0.0, scale=2.0))
    noise = numpyro.sample("kernel_noise", dist.LogNormal(loc=0.0, scale=2.0))
    period = numpyro.sample("kernel_period", dist.InverseGamma(2, 1))
    k = numpyro.deterministic(
        "k", kernel(X=X, Z=X.T, var=var, period=period, length=length, noise=noise)
    )
    numpyro.sample(
        "y", dist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=k), obs=y
    )


def predict(rng_key, X, Y, X_test, var, period, length, noise):
    # compute kernels between train and test data, etc.

    k_pp = kernel(X_test, X_test.T, var, period, length, noise, include_noise=True)
    k_pX = kernel(X_test, X.T, var, period, length, noise, include_noise=False)
    k_XX = kernel(X, X.T, var, period, length, noise, include_noise=True)
    K_xx_inv = jnp.linalg.inv(k_XX)
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * random.normal(
        rng_key, X_test.shape[:1]
    )

    mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean, mean + sigma_noise


N = 100
N_test = 400
# X = jnp.asarray(np.random.uniform(-np.pi * 3 / 2, np.pi * 3 / 2, size=(N, 1)))
X1 = jnp.asarray(np.random.uniform(-np.pi * 3 / 2, -np.pi * 1 / 2, size=(N // 2, 1)))
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


# plt.plot(X.ravel(), y.ravel(), "x", color="tab:blue", markersize=5)
# plt.show()


rng_key, rng_key_predict = random.split(random.PRNGKey(532))
nuts_kernel = NUTS(
    GaussianProcess,
    init_strategy=init_to_value(
        values={
            "kernel_var": 0.5,
            "kernel_noise": 0.05,
            "kernel_length": 0.5,
            "kernel_period": jnp.pi,
        }
    ),
)
mcmc = MCMC(
    nuts_kernel,
    num_warmup=1000,
    num_samples=4000,
    num_chains=1,
    chain_method="vectorized",
)
mcmc.run(X=X, y=y, rng_key=rng_key)
mcmc.print_summary()
samples = mcmc.get_samples()


vmap_args = (
    random.split(rng_key_predict, samples["kernel_var"].shape[0]),
    samples["kernel_var"],
    samples["kernel_period"],
    samples["kernel_length"],
    samples["kernel_noise"],
)
means, predictions = vmap(
    lambda rng_key, var, period, length, noise: predict(
        rng_key, X, y, X_test, var, period, length, noise
    )
)(*vmap_args)


yhat = jnp.mean(means, axis=0)
y_hpdi = hpdi(predictions, prob=0.9)
y_05 = y_hpdi[0, :]
y_95 = y_hpdi[1, :]

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(X.ravel(), y.ravel(), "x", color="tab:blue", markersize=5)
ax.plot(X_test.ravel(), yhat.ravel(), "tab:orange")
ax.fill_between(X_test.ravel(), y_05.ravel(), y_95.ravel(), alpha=0.5, color="green")

datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
plt.savefig(f"plots/GaussianProcess_{datetime_str}.jpg")
