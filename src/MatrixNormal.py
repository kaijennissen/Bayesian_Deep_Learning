import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpyro.distributions as dist
from jax import random
from numpyro.distributions import Distribution
from optax import scale


class MatrixNormal(Distribution):
    def __init__(self, loc, scale_columns_tril, scale_rows_tril):
        super(MatrixNormal, self).__init__()

    def sample(self, key, sample_shape=()):
        pass

    def log_prob(self, value):
        pass


# Case 1: sample_shape=(s,), batch_shape=()
def sample1(rng_key, loc, scale_rows, scale_columns, sample_shape=()):
    event_shape = loc.shape[-2:]
    batch_shape = ()
    n, p = event_shape
    assert loc.ndim == scale_columns.ndim == scale_rows.ndim
    assert scale_columns.shape == (p, p)
    assert scale_rows.shape == (n, n)

    tril_V = jsp.linalg.cholesky(scale_rows, lower=True)
    assert tril_V.shape == batch_shape + (n, n)

    triu_U = jsp.linalg.cholesky(scale_columns, lower=False)
    assert triu_U.shape == batch_shape + (p, p)

    X = random.normal(rng_key, shape=sample_shape + batch_shape + event_shape)
    assert X.shape == sample_shape + batch_shape + event_shape

    # X.shape = (n,p) & x_{i,j} ~ N(0,1) -> Y ~ MN(loc,tril_V@tril_V' , triu_U'@triu_U)
    # with Y = loc + tril_V @ X @ triu_U (https://en.wikipedia.org/wiki/Matrix_normal_distribution)
    Y = loc + jax.vmap(lambda x: tril_V @ x @ triu_U)(X)
    assert Y.shape == sample_shape + batch_shape + event_shape

    return Y


# Test 1:
sample_shape = (10,)
batch_shape = ()
event_shape = (2, 3)

rng_key = random.PRNGKey(435)

loc = jnp.arange(6).reshape(event_shape)
assert loc.shape == batch_shape + event_shape

tril_U = jnp.array([[1.0, 0, 0], [4.0, 1.0, 0], [0.4, 2.25, 1.0]])
scale_columns = jnp.matmul(tril_U, tril_U.T)
assert scale_columns.shape == batch_shape + (event_shape[1], event_shape[1])


tril_V = jnp.array([[4.0, 0.0], [1, 0.25]])
scale_rows = jnp.matmul(tril_V, tril_V.T)
assert scale_rows.shape == batch_shape + (event_shape[0], event_shape[0])

Y = sample1(rng_key, loc, scale_rows, scale_columns, sample_shape=sample_shape)
assert Y.shape == sample_shape + batch_shape + event_shape


# Case 2: sample_shape=(s,), batch_shape=(b,)
def sample2(rng_key, loc, scale_rows, scale_columns, sample_shape=()):
    event_shape = loc.shape[-2:]
    n, p = event_shape
    batch_shape = loc.shape[:-2]
    assert loc.ndim == scale_columns.ndim == scale_rows.ndim

    triu_U = jsp.linalg.cholesky(scale_columns, lower=False)
    assert triu_U.shape == batch_shape + (p, p)
    tril_V = jsp.linalg.cholesky(scale_rows, lower=True)
    assert tril_V.shape == batch_shape + (n, n)

    X = random.normal(rng_key, shape=sample_shape + batch_shape + event_shape)
    assert X.shape == sample_shape + batch_shape + event_shape

    # X.shape = (n,p) & x_{i,j} ~ N(0,1) -> Y ~ MN(loc,tril_V@tril_V' , triu_U'@triu_U)
    # with Y = loc + tril_V @ X @ triu_U (https://en.wikipedia.org/wiki/Matrix_normal_distribution)
    # X.ndim=4 > tril_U.ndim=triu_U.ndim=3
    # Step 1: use vmap to map matrix multiplicaton along batch and then along sampels
    batch_map = jax.vmap(lambda x, y, z: x @ y @ z, in_axes=0, out_axes=0)
    assert batch_map(tril_V, X[0, ...], triu_U).shape == batch_shape + event_shape
    sample_map = jax.vmap(lambda x: batch_map(tril_V, x, triu_U))
    Y = sample_map(X)
    assert Y.shape == sample_shape + batch_shape + event_shape

    return Y


# Test 2:
sample_shape = (10,)
batch_shape = (2,)
event_shape = (2, 3)

rng_key = random.PRNGKey(435)


loc = jnp.arange(6).reshape(event_shape) * jnp.ones(batch_shape + (1, 1))
assert loc.shape == batch_shape + event_shape

tril_U = jnp.array([[1.0, 0, 0], [4.0, 1.0, 0], [0.4, 2.25, 1.0]])
scale_columns = jnp.matmul(tril_U, tril_U.T) * jnp.ones(batch_shape + (1, 1))
assert scale_columns.shape == batch_shape + (event_shape[1], event_shape[1])

tril_V = jnp.array([[4.0, 0.0], [1, 0.25]])
scale_rows = jnp.matmul(tril_V, tril_V.T) * jnp.ones(batch_shape + (1, 1))
assert scale_rows.shape == batch_shape + (event_shape[0], event_shape[0])

Y = sample2(rng_key, loc, scale_rows, scale_columns, sample_shape=sample_shape)
assert Y.shape == sample_shape + batch_shape + event_shape


for r in range(Y.shape[1]):
    for c in range(Y.shape[2]):
        true_cov = scale_rows[r, r] * scale_columns[c, c]
        print(f"Sample cov: {jnp.cov(Y[:,r,c])}; True cov: {true_cov}")

# sample_shape=(s,), batch_shape=(),
def log_prob1(x, loc, scale_rows, scale_columns):
    event_shape = loc.shape[-2:]
    n, p = event_shape
    k = n * p
    batch_shape = loc.shape[:-2]

    new_shape = (-1,) + batch_shape + (k,)
    loc_mvn = loc.reshape(new_shape)

    tril_U = jsp.linalg.cholesky(scale_columns, lower=True)
    tril_V = jsp.linalg.cholesky(scale_rows, lower=True)
    assert tril_U.shape == batch_shape + (p, p)
    assert tril_V.shape == batch_shape + (n, n)

    # If (x) is the Kronecker-Product then it holds that
    # (A (x) B) @ (C (x) D) =(A @ C) (x) (B @ D)
    # see (KRON 13 - https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/kronthesisschaecke04.pdf)
    tril_scale = jax.vmap(lambda x, y: jnp.kron(x, y))(tril_U, tril_V)
    assert tril_scale.shape == batch_shape + (k, k)
    # mvn = dist.MultivariateNormal(loc=loc_mvn, scale_tril=tril_scale)
    mvn = dist.MultivariateNormal(
        loc=jnp.squeeze(loc_mvn), scale_tril=jnp.squeeze(tril_scale)
    )
    assert mvn.event_shape == (k,)
    assert mvn.batch_shape == batch_shape
    log_prob_ = mvn.log_prob(x.reshape(new_shape))
    assert log_prob_.shape == sample_shape + batch_shape
    return log_prob_


sample_shape = (10,)
batch_shape = ()
event_shape = (2, 3)

rng_key = random.PRNGKey(435)

loc = jnp.arange(6).reshape(event_shape)
assert loc.shape == batch_shape + event_shape

tril_U = jnp.array([[1.0, 0, 0], [4.0, 1.0, 0], [0.4, 2.25, 1.0]])
scale_columns = jnp.matmul(tril_U, tril_U.T)
assert scale_columns.shape == batch_shape + (event_shape[1], event_shape[1])

tril_V = jnp.array([[4.0, 0.0], [1, 0.25]])
scale_rows = jnp.matmul(tril_V, tril_V.T)
assert scale_rows.shape == batch_shape + (event_shape[0], event_shape[0])

Y = sample1(rng_key, loc, scale_rows, scale_columns, sample_shape=sample_shape)
assert Y.shape == sample_shape + batch_shape + event_shape

x_log_prob = log_prob1(Y, loc, scale_rows, scale_columns)
assert x_log_prob.shape == sample_shape + batch_shape

# Test 4:
sample_shape = (10,)
batch_shape = (2,)
event_shape = (2, 3)

rng_key = random.PRNGKey(435)


loc = jnp.arange(6).reshape(event_shape) * jnp.ones(batch_shape + (1, 1))
assert loc.shape == batch_shape + event_shape

tril_U = jnp.array([[1.0, 0, 0], [4.0, 1.0, 0], [0.4, 2.25, 1.0]])
scale_columns = jnp.matmul(tril_U, tril_U.T) * jnp.ones(batch_shape + (1, 1))
assert scale_columns.shape == batch_shape + (event_shape[1], event_shape[1])

tril_V = jnp.array([[4.0, 0.0], [1, 0.25]])
scale_rows = jnp.matmul(tril_V, tril_V.T) * jnp.ones(batch_shape + (1, 1))
assert scale_rows.shape == batch_shape + (event_shape[0], event_shape[0])

Y = sample2(rng_key, loc, scale_rows, scale_columns, sample_shape=sample_shape)
assert Y.shape == sample_shape + batch_shape + event_shape

x_log_prob = log_prob1(Y, loc, scale_rows, scale_columns)
assert x_log_prob.shape == sample_shape + batch_shape
