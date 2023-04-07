import pickle
import time
import warnings
from functools import partial

import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup
import flax.linen as nn
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module
from numpyro.infer.util import initialize_model

warnings.simplefilter("ignore", FutureWarning)
with open("data/mnist_train.pickle", "rb") as file:
    train = pickle.load(file)
with open("data/mnist_test.pickle", "rb") as file:
    test = pickle.load(file)

x_train, y_train = train["image"] / 255, train["label"] * 1.0
x_test, y_test = test["image"] / 255, test["label"] * 1.0


class CNN(nn.Module):
    """LeNet"""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=4, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(6, 6), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x


def BayesianCNN(X, y=None):
    N, *_ = X.shape
    module = CNN()
    net = random_flax_module(
        "nn", module, dist.Normal(0.0, 10.0), input_shape=(1, 28, 28, 1)
    )
    logits = net(X)
    if y is not None:
        assert y.shape == (N,)
    assert logits.shape == (N, 10)
    numpyro.sample("y", dist.Categorical(logits=logits), obs=y)


rng_key = jax.random.PRNGKey(2643)
# Step 1: get the log-probability of the model
init_params, potential_fn_gen, *_ = initialize_model(
    rng_key,
    BayesianCNN,
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
    rng_key, kernel_factory, initial_state, 2
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
