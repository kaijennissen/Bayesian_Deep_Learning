from typing import Callable

import jax.numpy as jnp
from jax import jit


@jit
def convolution_fn(inp, W):
    batch, width_in, height_in, *_ = inp.shape
    height_filter, width_filter, num_filters = W.shape
    width_out = width_in
    height_out = height_in

    input_padded = jnp.pad(
        inp, [(0, 0), (2, 2), (2, 2), (0, 0)], mode="constant", constant_values=0
    )
    output = jnp.zeros((batch, height_out, width_out, num_filters))

    for i in range(height_out):
        for j in range(width_out):
            h_start = i
            h_end = h_start + height_filter
            w_start = j
            w_end = w_start + width_filter
            # batch_dim x height x width x channels x num_filters
            # x_slice = x[vert_start: vert_end, horiz_start: horiz_end, :]
            # np.sum(np.multiply(input, W)) + float(b)
            val = jnp.sum(
                input_padded[:, h_start:h_end, w_start:w_end, :, jnp.newaxis]
                * W[:, :, jnp.newaxis, :],
                axis=(1, 2, 3),
            )
            output.at[:, i, j, :].set(val)
    return output


def pool_fn(inp, pool_size: int = 2, reducer_fn: Callable = jnp.max):
    # pool_size = (2, 2)
    batch, height_in, width_in, num_filters = inp.shape

    height_out = height_in // pool_size
    width_out = width_in // pool_size
    # output = jnp.zeros((batch, height_out, width_out, num_filters))
    h_step = pool_size
    w_step = pool_size
    vals = []
    for i in range(height_out):
        for j in range(width_out):
            h_start = i * h_step
            h_end = h_start + h_step
            w_start = j * w_step
            w_end = w_start + w_step
            # print(f"h_start: {h_start}; h_end: {h_end}; w_start: {w_start}; w_end: {w_end}")

            val = reducer_fn(inp[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
            vals.append(val)
            # output.at[:, i, j, :].set(0.5)  # = val
            # breakpoint()
    output = jnp.concatenate(vals).reshape(batch, height_out, width_out, num_filters)
    return output
