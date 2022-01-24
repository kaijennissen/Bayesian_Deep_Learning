import time
from functools import partial

import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup
import jax
import jax.numpy as jnp
from numpyro.infer.util import initialize_model

from BayesianDNN import GaussianBNN, get_data

X, y, X_test = get_data()
rng_key = jax.random.PRNGKey(0)
# Step 1: get the log-probability of the model
init_params, potential_fn_gen, *_ = initialize_model(
    rng_key,
    GaussianBNN,
    model_args=(X, y),
    dynamic_args=True,
)

# Step 2:
logprob = lambda position: -potential_fn_gen(X, y)(position)
initial_position = init_params.z
initial_state = nuts.new_state(initial_position, logprob)

# Step 3: window adaption
kernel_factory = lambda step_size, inverse_mass_matrix: nuts.kernel(
    logprob, step_size, inverse_mass_matrix
)

last_state, (step_size, inverse_mass_matrix), _ = stan_warmup.run(
    rng_key, kernel_factory, initial_state, 1000
)


@partial(jax.jit, static_argnums=(1, 3))
def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, infos


# Build the kernel using the step size and inverse mass matrix returned from the window adaptation
kernel = kernel_factory(step_size, inverse_mass_matrix)

# Sample from the posterior distribution
start_time = time.time()
states, infos = inference_loop(rng_key, kernel, last_state, 1_000)
W1 = states.position["W1"].block_until_ready()
inference_time = time.time() - start_time
print(f"time: {inference_time}")
