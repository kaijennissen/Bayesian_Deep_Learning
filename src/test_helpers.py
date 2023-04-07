from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit

from CNN_helpers import pool_fn

# def test_convolution_fn():
#     img = jnp.arange(36).reshape(1, 6, 6, 1)
#     W = jnp.ones((3, 3))
#     assert jnp.allclose(convolution_fn(img, W), img)


def test_max_pool_fn():
    img = jnp.arange(36).reshape(1, 6, 6, 1) + 1
    expected = jnp.array([[8.0, 10.0, 12.0], [20.0, 22.0, 24.0], [32.0, 34.0, 36.0]])
    max_pool_fn = jit(partial(pool_fn, pool_size=2, reducer_fn=jnp.max))

    assert jnp.allclose(max_pool_fn(img)[0, :, :, 0], expected)

    img = jnp.arange(16).reshape(1, 4, 4, 1)
    expected = jnp.array([[5.0, 7.0], [13.0, 15.0]])
    max_pool_fn = jit(partial(pool_fn, pool_size=2, reducer_fn=jnp.max))
    assert jnp.allclose(max_pool_fn(img)[0, :, :, 0], expected)

    img = jnp.arange(9).reshape(1, 3, 3, 1) + 1
    expected = jnp.array([[9.0]])
    max_pool_fn = jit(partial(pool_fn, pool_size=3, reducer_fn=jnp.max))
    assert jnp.allclose(max_pool_fn(img)[0, :, :, 0], expected)

    img = jnp.arange(36).reshape(1, 6, 6, 1) + 1
    expected = jnp.array([[15.0, 18.0], [33.0, 36.0]])
    max_pool_fn = jit(partial(pool_fn, pool_size=3, reducer_fn=jnp.max))
    assert jnp.allclose(max_pool_fn(img)[0, :, :, 0], expected)

    img = np.zeros((4, 2, 2, 1))
    img[0, 0, 0, :] = 0
    img[1, 0, 1, :] = 1
    img[2, 1, 0, :] = 2
    img[3, 1, 1, :] = 3
    max_pool_fn = jit(partial(pool_fn, pool_size=1, reducer_fn=jnp.max))
    assert jnp.allclose(max_pool_fn(img), img)
