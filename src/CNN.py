import warnings

from numpy.random.mtrand import sample

warnings.simplefilter("ignore", FutureWarning)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import jit, random
from jax.nn import sigmoid
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
from scipy.stats import mode

LAYER_DIMS = [5, 5]
# https://num.pyro.ai/en/stable/primitives.html#numpyro.contrib.module.random_haiku_module


def stratified_samples(X, y, size=200):
    unique_values = np.unique(y)
    samples_idx = []
    for v in unique_values:
        samples_idx.append(
            np.random.choice(np.where(y == v)[0].ravel(), size, replace=False)
        )

    samples_idx = np.concatenate(samples_idx)
    np.random.shuffle(samples_idx)

    return X[samples_idx], y[samples_idx]


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
y = train.pop("label").values
y_train_onehot = (y[:, np.newaxis] == np.arange(10)) * 1
X = train.values.reshape(-1, 28, 28, 1) / 255
X_train, y_train = stratified_samples(X, y, size=1000)

X_train = jnp.asarray(X_train)  # type:ignore
y_train = jnp.asarray(y_train)


@jit
def nonlin(x):
    return sigmoid(x)


# TODO change axis where the num_filters are calculated
# TODO keep num_filters


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


@jit
def pooling_fn(inp):
    pool_size = (2, 2)
    batch, height_in, width_in, num_filters = inp.shape

    height_out = height_in // pool_size[0]
    width_out = width_in // pool_size[1]
    output = jnp.zeros((batch, height_out, width_out, num_filters))
    h_step = pool_size[0]
    w_step = pool_size[1]
    for i in range(height_out):
        for j in range(width_out):
            h_start = i * h_step
            h_end = h_start + (i + 1) * h_step
            w_start = j * w_step
            w_end = w_start + (i + 1) * w_step
            val = jnp.max(inp[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
            output.at[:, i, j, :].set(val)
    return output


@jit
def softmax(x):
    return jnp.exp(x) / jnp.sum(jnp.exp(x), axis=-1, keepdims=True)


# fixed architecture
def BayesianCNN(X, y=None, num_categories=10):

    B, H, W, C = X.shape
    num_filters = [6, 16]
    # Convolution + Max. Pooling
    W1_shape = (5, 5, num_filters[0])
    W1 = numpyro.sample(
        "W1", dist.Normal(-9.0 * jnp.ones(W1_shape), 0.1 * jnp.ones(W1_shape))
    )
    b1 = numpyro.sample(
        "b1",
        dist.Normal(jnp.zeros(W1_shape[-1]), 0.1 * jnp.ones(W1_shape[-1])),
    )
    out_conv1 = nonlin(convolution_fn(X, W1) + b1)
    assert out_conv1.shape == (B, H, W, num_filters[0])
    out_pool1 = pooling_fn(out_conv1)
    assert out_pool1.shape == (B, H // 2, W // 2, num_filters[0])

    # Convolution + Max. Pooling
    W2_shape = (5, 5, num_filters[1])
    W2 = numpyro.sample(
        "W2",
        dist.Normal(-9.0 * jnp.ones(W2_shape), 0.1 * jnp.ones(W2_shape)),
    )
    b2 = numpyro.sample(
        "b2",
        dist.Normal(jnp.zeros(W2_shape[-1]), 0.1 * jnp.ones(W2_shape[-1])),
    )

    out_conv2 = nonlin(convolution_fn(out_pool1, W2) + b2)
    assert out_conv2.shape == (B, H // 2, W // 2, num_filters[1])
    out_pool2 = pooling_fn(out_conv2)
    assert out_pool2.shape == (B, H // 4, W // 4, num_filters[1])

    # Flatten
    b, *_ = out_pool2.shape
    out_flatten = jnp.ravel(out_pool2).reshape(b, -1)
    assert out_flatten.shape == (B, (H // 4) * (W // 4) * num_filters[-1])

    # Dense
    W4_shape = (784, 120)
    W4 = numpyro.sample(
        "W4", dist.Normal(jnp.zeros(W4_shape), 0.1 * jnp.ones(W4_shape))
    )
    b4 = numpyro.sample(
        "b4", dist.Normal(jnp.zeros(W4_shape[-1]), 0.1 * jnp.ones(W4_shape[-1]))
    )
    out4 = nonlin(jnp.matmul(out_flatten, W4) + b4)
    assert out4.shape == (B, W4_shape[-1])

    W5_shape = (120, 84)
    W5 = numpyro.sample(
        "W5",
        dist.Normal(jnp.zeros(W5_shape), 0.1 * jnp.ones(W5_shape)),
    )
    b5 = numpyro.sample(
        "b5", dist.Normal(jnp.zeros(W5_shape[-1]), 0.1 * jnp.ones(W5_shape[-1]))
    )
    out5 = nonlin(jnp.matmul(out4, W5) + b5)
    assert out5.shape == (B, W5_shape[-1])

    W6_shape = (84, num_categories)
    W6 = numpyro.sample(
        "W6", dist.Normal(jnp.zeros(W6_shape), 0.1 * jnp.ones(W6_shape))
    )
    b6 = numpyro.sample(
        "b6", dist.Normal(jnp.zeros(num_categories), 0.1 * jnp.ones(num_categories))
    )
    out6 = jnp.matmul(out5, W6) + b6
    assert out6.shape == (B, num_categories)

    # probs = softmax(out6)
    logits = out6

    numpyro.sample("y", dist.Categorical(logits=logits), obs=y)


rng_key = random.PRNGKey(125)
guide = AutoNormal(BayesianCNN)
svi = SVI(BayesianCNN, guide, Adam(0.01), loss=Trace_ELBO())
svi_result, _, _ = svi.run(random.PRNGKey(0), 12000, X=X_train, y=y_train)
predictive = Predictive(BayesianCNN, guide=guide, num_samples=100, parallel=True)

# rng_key = random.PRNGKey(125)
# kernel = NUTS(BayesianCNN)
# mcmc = MCMC(
#     kernel,
#     num_warmup=1000,
#     num_samples=4000,
#     num_chains=1,
#     chain_method="vectorized",
#     jit_model_args=True,
# )
# mcmc.run(X=X_train, y=y_train, rng_key=rng_key)
# mcmc.print_summary()

# predictive = Predictive(
#     BayesianCNN, posterior_samples=mcmc.get_samples(), num_samples=100, parallel=True
# )

post_samples = predictive(rng_key=rng_key, X=X_train)


yhat, count = mode(post_samples["y"], axis=0)
yhat = yhat.ravel()
acc = np.mean(yhat == y_train)
print(f"Accuraccy: {acc}")

# idx = 77
# fig, ax = plt.subplots()
# ax.imshow(X_train[idx])
# ax.set_title(f"True: {y_train[idx]}, Pred: {yhat[idx]}")
# plt.show()

i = 0
imgs = [124, 12, 314, 718, 2, 22, 917, 513, 15]
fix, axes = plt.subplots(nrows=3, ncols=3)
for i, idx in enumerate(imgs):
    r = i // 3
    c = i % 3
    # print(f"row:{r}")
    # print(f"col:{c}")
    ax = axes[r, c]
    ax.imshow(X_train[idx])
    # ax.set_title(f"{y_train[imgs[i]]}")
    ax.set_title(f"True: {y_train[idx]}, Pred: {yhat[idx]}")
    i += 1
plt.show()


# y_hpdi = hpdi(post_samples["y"], prob=0.9)
# y_05 = y_hpdi[0, :, :]
# y_95 = y_hpdi[1, :, :]
