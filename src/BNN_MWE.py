import jax.numpy as jnp
from numpyro import deterministic, plate, sample
from numpyro.distributions import Gamma, Normal


def BNN(X, y=None):
    N, k = X.shape
    D_H1 = 4
    D_H2 = 4

    # layer 1
    W1 = sample("W1", Normal(loc=jnp.zeros((k, D_H1)), scale=jnp.ones((k, D_H1))))
    b1 = sample("b1", Normal(loc=jnp.zeros(D_H1), scale=1.0))
    out1 = jnp.tanh(jnp.matmul(X, W1) + b1)

    # layer 2
    W2 = sample("W2", Normal(loc=jnp.zeros((D_H1, D_H2)), scale=jnp.ones((D_H1, D_H2))))
    b2 = sample("b2", Normal(loc=jnp.zeros(D_H2), scale=jnp.ones(D_H2)))
    out2 = jnp.tanh(jnp.matmul(out1, W2) + b2)

    # output layer
    W3 = sample("W3", Normal(loc=jnp.zeros((D_H2, 1)), scale=jnp.ones((D_H2, 1))))
    b3 = sample("b3", Normal(loc=jnp.zeros(1), scale=jnp.ones(1)))
    out3 = jnp.matmul(out2, W3) + b3

    mean = deterministic("mean", out3)

    prec_obs = sample("prec_obs", Gamma(3.0, 1.0))
    scale = 1.0 / jnp.sqrt(prec_obs)

    with plate("data", size=N, dim=-2):
        sample("y", Normal(loc=mean, scale=scale), obs=y)
