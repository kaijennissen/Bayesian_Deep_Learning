from math import prod

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpyro.distributions as dist
from jax import random
from numpyro.distributions import Distribution, constraints

independent = constraints._IndependentConstraint
real_matrix = independent(constraints.real, 2)


@jax.jit
def my_kron(A, B):
    D = A[..., :, None, :, None] * B[..., None, :, None, :]
    ds = D.shape
    newshape = (*ds[:-4], ds[-4] * ds[-3], ds[-2] * ds[-1])
    return D.reshape(newshape)


class MatrixNormal(Distribution):
    arg_constraints = {
        "loc": constraints.real_vector,
        "scale_columns": constraints.positive_definite,
        "scale_rows": constraints.positive_definite,
    }
    support = real_matrix
    reparametrized_params = ["loc", "scale_columns", "scale_rows"]

    def __init__(self, loc, scale_columns, scale_rows, validate_args=None):
        if not (loc.ndim == scale_columns.ndim == scale_rows.ndim):
            raise ValueError(
                "ndim of loc, scale_columns and scale_rows are reuqired to be equal."
            )

        event_shape = loc.shape[-2:]
        n, p = event_shape
        batch_shape = loc.shape[:-2]

        assert scale_rows.shape == batch_shape + (
            n,
            n,
        ), "scale_rows.shape does not match"
        assert scale_columns.shape == batch_shape + (
            p,
            p,
        ), "scale_rows.shape does not match"

        super(MatrixNormal, self).__init__(
            batch_shape=batch_shape, event_shape=event_shape
        )

        self.loc = loc
        self.scale_columns = scale_columns
        self.scale_rows = scale_rows

    def sample(self, key, sample_shape=()):

        mvn_dim = prod(self.event_shape)
        loc = jnp.reshape(self.loc, newshape=self.batch_shape + (mvn_dim,), order="F")
        assert loc.shape == self.batch_shape + (mvn_dim,), "vec shape does not match"

        cov = my_kron(self.scale_rows, self.scale_columns)
        assert cov.shape == self.batch_shape + (
            mvn_dim,
            mvn_dim,
        ), "cov shape does not match"

        mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)

        assert mvn.batch_shape == self.batch_shape, "batch shape of MVN does not match"
        assert mvn.event_shape == (
            prod(self.event_shape),
        ), "event shape of MVN does not match"

        samples_mvn = mvn.sample(key, sample_shape=sample_shape)
        assert samples_mvn.shape == sample_shape + self.batch_shape + (
            mvn_dim,
        ), "MVN sample shape does not match"
        samples_mn = samples_mvn.reshape(
            sample_shape + self.batch_shape + self.event_shape, order="F"
        )
        assert (
            samples_mn.shape == sample_shape + self.batch_shape + self.event_shape
        ), "MN sample shape does not match"

        return samples_mn

    def log_prob(self, value):
        mvn_dim = prod(self.event_shape)
        loc = jnp.reshape(self.loc, newshape=self.batch_shape + (mvn_dim,), order="F")
        assert loc.shape == self.batch_shape + (mvn_dim,)

        cov = my_kron(self.scale_rows, self.scale_columns)
        assert cov.shape == self.batch_shape + (
            mvn_dim,
            mvn_dim,
        ), "cov shape does not match"

        mvn = dist.MultivariateNormal(loc=loc, covariance_matrix=cov)
        values = jnp.reshape(
            self.loc, newshape=self.batch_shape + (mvn_dim,), order="F"
        )
        log_prob_mvn = mvn.log_prob(values)
        return log_prob_mvn
