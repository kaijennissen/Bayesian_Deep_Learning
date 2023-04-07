import pickle
import time
from functools import partial

import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model

with open("data/mnist_train.pickle", "rb") as file:
    train = pickle.load(file)
with open("data/mnist_test.pickle", "rb") as file:
    test = pickle.load(file)

x_train, y_train = train["image"] / 255, train["label"] * 1.0
x_test, y_test = test["image"] / 255, test["label"] * 1.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)


@jax.jit
def nonlin(x):
    return jax.nn.relu(x)


def GaussianBNN(X, y=None):
    N, feature_dim = X.shape
    out_dim = 10
    layer1_dim = 64
    layer2_dim = 64

    # layer 1
    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((feature_dim, layer1_dim)), 1.0))
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(layer1_dim), 1.0))
    out1 = nonlin(jnp.matmul(X, W1)) + b1

    # layer 2
    W2 = numpyro.sample("W2", dist.Normal(jnp.zeros((layer1_dim, layer2_dim)), 1.0))
    b2 = numpyro.sample("b2", dist.Normal(jnp.zeros(layer1_dim), 1.0))
    out2 = nonlin(jnp.matmul(out1, W2)) + b2

    # output layer
    W3 = numpyro.sample("out_layer", dist.Normal(jnp.zeros((layer2_dim, out_dim)), 1.0))
    b3 = numpyro.sample("b3", dist.Normal(jnp.zeros(out_dim), 1.0))

    logits = numpyro.deterministic("mean", jnp.dot(out2, W3) + b3)

    assert logits.shape == (N, 10)
    if y is not None:
        assert y.shape == (N,)

    with numpyro.plate("data", size=N, dim=-2):
        numpyro.sample("y", dist.Categorical(logits=logits), obs=y)


rng_key = jax.random.PRNGKey(0)
# Step 1: get the log-probability of the model
init_params, potential_fn_gen, *_ = initialize_model(
    rng_key,
    GaussianBNN,
    model_args=(x_test, y_test),
    dynamic_args=True,
)


# Step 2:
def logprob(position):
    return -potential_fn_gen(x_test, y_test)(position)


initial_position = init_params.z
initial_state = nuts.new_state(initial_position, logprob)


# Step 3: window adaption
def kernel_factory(step_size, inverse_mass_matrix):
    return nuts.kernel(logprob, step_size, inverse_mass_matrix)


print("start warmup")
start_time = time.time()
last_state, (step_size, inverse_mass_matrix), _ = stan_warmup.run(
    rng_key, kernel_factory, initial_state, 1000
)
end_time = time.time() - start_time
print(f"finished warmup in : {end_time}")


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
print("Started sampling!")
start_time = time.time()
states, infos = inference_loop(rng_key, kernel, last_state, 1_000)
W1 = states.position["W1"].block_until_ready()
inference_time = time.time() - start_time
print(f"Finished sampling in time: {inference_time}")
