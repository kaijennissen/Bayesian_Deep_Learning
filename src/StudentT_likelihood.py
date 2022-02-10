import argparse
import warnings
from datetime import datetime

warnings.simplefilter("ignore", FutureWarning)
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.nn import relu, tanh
from numpyro.diagnostics import hpdi
from numpyro.distributions import constraints
from numpyro.infer import (
    MCMC,
    NUTS,
    SVI,
    Predictive,
    RenyiELBO,
    Trace_ELBO,
    TraceGraph_ELBO,
    TraceMeanField_ELBO,
)
from numpyro.infer.autoguide import AutoDAIS

numpyro.set_host_device_count(4)

# https://github.com/microsoft/horseshoe-bnn
# https://arxiv.org/pdf/1905.02599.pdf
# https://jmlr.org/papers/volume20/19-236/19-236.pdf


def GaussianBNN(X, y=None, D_H=10):

    N, k = X.shape
    D_Y = 1

    scale = 1

    # layer 1
    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((k, D_H)), scale))
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(D_H), scale))
    out1 = tanh(jnp.dot(X, W1) + b1)

    # output layer
    W2 = numpyro.sample("W2", dist.Normal(jnp.zeros((D_H, D_Y)), scale))
    b2 = numpyro.sample("b2", dist.Normal(jnp.zeros(D_Y), scale))
    mean = numpyro.deterministic("mean", jnp.dot(out1, W2) + b2)

    # shape, rate parametrization (see Wikipedia)
    #  -> expected precision = alpha/beta, variance precision = alpha/beta^2
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(1, 1))
    scale = numpyro.deterministic("obs_scale", 1.0 / jnp.sqrt(prec_obs))

    assert mean.shape == (N, 1)
    assert scale.shape == ()
    if y is not None:
        assert y.shape == (N,)

    with numpyro.plate("data", size=N):
        numpyro.sample("y", dist.Normal(loc=jnp.squeeze(mean), scale=scale), obs=y)


def StudentTBNN(X, y=None, D_H=10):

    N, k = X.shape
    D_Y = 1

    scale = 1

    # layer 1
    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((k, D_H)), scale))
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(D_H), scale))
    out1 = tanh(jnp.dot(X, W1) + b1)

    # output layer
    W2 = numpyro.sample("W2", dist.Normal(jnp.zeros((D_H, D_Y)), scale))
    b2 = numpyro.sample("b2", dist.Normal(jnp.zeros(D_Y), scale))
    mean = numpyro.deterministic("mean", jnp.dot(out1, W2) + b2)

    # shape, rate parametrization (see Wikipedia)
    #  -> expected precision = alpha/beta, variance precision = alpha/beta^2
    # Direct paramtrization
    # prec_obs = numpyro.sample("prec_obs", dist.Gamma(0.1, 0.1))
    # scale = numpyro.deterministic("obs_scale", 1.0 / jnp.sqrt(prec_obs))
    # nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))

    # StudentT as Infinite Normal-Gamma Mixture parametrizaction
    alpha = numpyro.sample("alpha", dist.Uniform(0, 100))
    beta = numpyro.sample("beta", dist.Uniform(0, 100))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    lam = numpyro.sample("lam", dist.Gamma(alpha**2 / 2, beta / 2))
    omega = numpyro.sample("omega", dist.Gamma(nu / 2, nu / 2))
    scale = numpyro.deterministic("scale", 1 / jnp.sqrt(lam * omega))

    assert mean.shape == (N, 1)
    assert scale.shape == ()
    if y is not None:
        assert y.shape == (N,)
    # with numpyro.plate("data", size=N):
    #     numpyro.sample(
    #         "y", dist.StudentT(df=nu, loc=jnp.squeeze(mean), scale=scale), obs=y
    #     )
    with numpyro.plate("data", size=N):
        numpyro.sample("y", dist.Normal(loc=jnp.squeeze(mean), scale=scale), obs=y)


def guideBNN(X, y=None, D_H=10):

    N, k = X.shape
    D_Y = 1

    # layer 1
    W1_loc = numpyro.param("W1_loc", init_value=jnp.ones((k, D_H)))
    W1_scale = numpyro.param(
        "W1_scale",
        init_value=jnp.ones((k, D_H)),
        constraint=constraints.positive,
    )
    b1_loc = numpyro.param("b1_loc", init_value=jnp.ones(D_H))
    b1_scale = numpyro.param(
        "b1_scale", init_value=jnp.ones(D_H), constraint=constraints.positive
    )

    W1 = numpyro.sample("W1", dist.Normal(W1_loc, W1_scale))
    b1 = numpyro.sample("b1", dist.Normal(b1_loc, b1_scale))

    # layer 2
    W2_loc = numpyro.param("W2_loc", init_value=jnp.ones((D_H, D_Y)))
    W2_scale = numpyro.param(
        "W2_scale",
        init_value=jnp.ones((D_H, D_Y)),
        constraint=constraints.positive,
    )
    b2_loc = numpyro.param("b2_loc", init_value=jnp.ones(D_Y))
    b2_scale = numpyro.param(
        "b2_scale", init_value=jnp.ones(D_Y), constraint=constraints.positive
    )

    W2 = numpyro.sample("W2", dist.Normal(W2_loc, W2_scale))
    b2 = numpyro.sample("b2", dist.Normal(b2_loc, b2_scale))

    # shape, rate parametrization (see Wikipedia) -> mean = 1, var = 1/6
    alpha = numpyro.param("alpha", init_value=6, constraint=constraints.positive)
    beta = numpyro.param("beta", init_value=6, constraint=constraints.positive)

    prec_obs = numpyro.sample("prec_obs", dist.Gamma(alpha, beta))
    scale = numpyro.deterministic("obs_scale", 1.0 / jnp.sqrt(prec_obs))


def sin_data(N=10, N_test=1_000):

    np.random.seed(89)
    x = np.random.uniform(-np.pi / 2, np.pi / 2, N)
    I = np.random.binomial(n=1, p=0.8, size=N)
    y = (
        np.sin(x * 2)
        + I * np.random.normal(0, 0.1, size=(N))
        + (1 - I) * np.random.normal(0, 3, size=(N))
    )

    x_test = jnp.linspace(-1.3 * (np.pi / 2), (np.pi / 2) * 1.3, N_test)

    # plt.plot(x, y, ".")
    # plt.plot(x_test, np.sin(2 * x_test), "r--")
    # plt.show()
    # raise ValueError

    return x[:, np.newaxis], y, x_test[:, np.newaxis]


def main(
    N: int = 50,
    num_warmup: int = 5000,
    num_samples: int = 4000,
    num_chains: int = 2,
    D_H: int = 10,
    inference: str = "MCMC",
    model_str: str = "Gaussian",
):

    if model_str == "Gaussian":
        model = GaussianBNN
    elif model_str == "StudentT":
        model = StudentTBNN

    X, y, X_test = sin_data(N=N)
    rng_key = jax.random.PRNGKey(2309)

    if inference == "SVI":
        # guide = guideBNN
        guide = AutoDAIS(model, K=128)
        svi = SVI(model, guide, numpyro.optim.Adam(5e-3), Trace_ELBO())
        svi_result = svi.run(rng_key, num_samples, X=X, y=y, D_H=D_H)
        params, losses = svi_result.params, svi_result.losses
        predictive = Predictive(model, guide=guide, params=params, num_samples=1000)

    elif inference == "MCMC":
        kernel = NUTS(model)
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method="parallel",
        )
        mcmc.run(X=X, y=y, D_H=D_H, rng_key=rng_key)
        mcmc.print_summary()

        samples = mcmc.get_samples()
        predictive = Predictive(model, posterior_samples=samples, parallel=True)

    post_samples = predictive(rng_key=jax.random.PRNGKey(2), X=X_test, D_H=D_H)

    yhat = jnp.mean(post_samples["y"], axis=0)

    y_hpdi = hpdi(post_samples["y"], prob=0.9)
    y_05 = y_hpdi[0, :].ravel()
    y_95 = y_hpdi[1, :].ravel()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    ax.plot(X, y, ".", color="tab:blue", markersize=5, label="Obs.")
    ax.plot(X_test.ravel(), yhat, "tab:orange", label="Mean Prediction")
    ax.plot(
        X_test.ravel(),
        np.sin(2 * X_test.ravel()),
        "tab:red",
        ls="--",
        label="Ground Truth",
    )
    ax.fill_between(
        X_test.ravel(), y_05, y_95, alpha=0.3, color="green", label="90%-HPDI"
    )
    ax.set_xlabel("x")
    ax.set_xlabel("y")
    ax.set_title(f"BNN with HS-prior and {D_H} hidden units")
    plt.legend(loc="upper left")

    datetime_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig(f"plots/experiment2/{model_str}_{N}_{D_H}_{inference}.jpg", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", default=50, type=int)
    parser.add_argument("--num-warmup", default=1_000, type=int)
    parser.add_argument("--num-samples", default=4_000, type=int)
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("--hidden-units", default=10, type=int)
    parser.add_argument("--inference", default="MCMC", type=str)
    parser.add_argument("--model", default="Gaussian", type=str)

    args = parser.parse_args()
    main(
        N=args.sample_size,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        D_H=args.hidden_units,
        inference=args.inference,
        model_str=args.model,
    )
