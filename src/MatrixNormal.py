from math import prod

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro.distributions as dist
from jax import lax, random
from jax.scipy.linalg import solve_triangular
from numpyro.distributions import Distribution, constraints

independent = constraints._IndependentConstraint
real_matrix = independent(constraints.real, 2)


@jax.jit
def my_kron(A, B):
    D = A[..., :, None, :, None] * B[..., None, :, None, :]
    ds = D.shape
    newshape = (*ds[:-4], ds[-4] * ds[-3], ds[-2] * ds[-1])
    return D.reshape(newshape)


def _reshape(x, shape):
    if isinstance(x, (int, float, np.ndarray, np.generic)):
        return np.reshape(x, shape)
    else:
        return jnp.reshape(x, shape)


def promote_shapes(*args, shape=()):
    # adapted from lax.lax_numpy
    if len(args) < 2 and not shape:
        return args
    else:
        shapes = [jnp.shape(arg) for arg in args]
        num_dims = len(lax.broadcast_shapes(shape, *shapes))
        return [
            _reshape(arg, (1,) * (num_dims - len(s)) + s) if len(s) < num_dims else arg
            for arg, s in zip(args, shapes)
        ]


def _batch_solve_triangular(A, B):
    """
    Extende solve_triangular for the case that B.ndim > A.ndim.
    This is achived by first flattening the leading B.ndim - A.ndim dimensions of B and then
    moving the first dimension to the end.


    :param jnp.ndarray (...,M,M) A: An array with lower triangular structure in the last two dimensions.
    :param jnp.ndarray (...,M,N) B: Right-hand side matrix in A x = B.

    :return: Solution of A x = B.
    """
    event_shape = B.shape[-2:]
    batch_shape = lax.broadcast_shapes(A.shape[:-2], B.shape[-A.ndim : -2])
    sample_shape = B.shape[: -A.ndim]
    n, p = event_shape

    A = jnp.broadcast_to(A, batch_shape + A.shape[-2:])
    B = jnp.broadcast_to(B, sample_shape + batch_shape + event_shape)

    B_flat = jnp.moveaxis(B.reshape((-1,) + batch_shape + event_shape), 0, -2).reshape(
        batch_shape + (n,) + (-1,)
    )

    X_flat = solve_triangular(A, B_flat, lower=True)

    sample_shape_dim = len(sample_shape)
    src_axes = tuple([-2 - i for i in range(sample_shape_dim)])
    src_axes = src_axes[::-1]
    dest_axes = tuple([i for i in range(sample_shape_dim)])

    X = jnp.moveaxis(
        X_flat.reshape(batch_shape + (n,) + sample_shape + (p,)),
        src_axes,
        dest_axes,
    )
    return X


class MatrixNormal(Distribution):
    arg_constraints = {
        "loc": constraints.real_vector,
        "scale_tril_row": constraints.lower_cholesky,
        "scale_tril_column": constraints.lower_cholesky,
    }
    support = real_matrix
    reparametrized_params = ["loc", "scale_tril_column", "scale_tril_row"]

    def __init__(self, loc, scale_tril_row, scale_tril_column, validate_args=None):
        """_summary_

        Args:
            loc (_type_): _description_
            scale_tril_column (_type_): _description_
            scale_tril_row (_type_): _description_
            validate_args (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """

        event_shape = loc.shape[-2:]
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(loc)[:-2],
            jnp.shape(scale_tril_row)[:-2],
            jnp.shape(scale_tril_column)[:-2],
        )
        (self.loc,) = promote_shapes(loc, shape=batch_shape + loc.shape[-2:])
        (self.scale_tril_row,) = promote_shapes(
            scale_tril_row, shape=batch_shape + scale_tril_row.shape[-2:]
        )
        (self.scale_tril_column,) = promote_shapes(
            scale_tril_column, shape=batch_shape + scale_tril_column.shape[-2:]
        )
        super(MatrixNormal, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    @property
    def mean(self):
        return jnp.broadcast_to(self.loc, self.shape())

    def sample(self, key, sample_shape=()):

        eps = random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )
        samples = self.loc + self.scale_tril_row @ eps @ jnp.swapaxes(
            self.scale_tril_column, -2, -1
        )

        return samples

    def log_prob(self, values):
        n, p = self.event_shape
        row_log_det = jnp.log(
            jnp.diagonal(self.scale_tril_row, axis1=-2, axis2=-1)
        ).sum(-1)
        col_log_det = jnp.log(
            jnp.diagonal(self.scale_tril_column, axis1=-2, axis2=-1)
        ).sum(-1)
        log_det_term = (
            p * row_log_det + n * col_log_det + 0.5 * n * p * jnp.log(2 * jnp.pi)
        )

        # compute the trace term
        diff = values - self.loc
        diff_row_solve = _batch_solve_triangular(A=self.scale_tril_row, B=diff)
        diff_col_solve = _batch_solve_triangular(
            A=self.scale_tril_column, B=jnp.swapaxes(diff_row_solve, -2, -1)
        )
        batched_trace_term = jnp.square(
            diff_col_solve.reshape(diff_col_solve.shape[:-2] + (-1,))
        ).sum(-1)

        log_prob = -0.5 * batched_trace_term - log_det_term

        return log_prob
