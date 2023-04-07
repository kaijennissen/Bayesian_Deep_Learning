import warnings
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import jit, random
from jax.nn import log_softmax, relu
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
from scipy.stats import mode

from CNN_helpers import convolution_fn, pool_fn

warnings.simplefilter("ignore", FutureWarning)


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
X_train, y_train = stratified_samples(X, y, size=200)

X_train = jnp.asarray(X_train)  # type:ignore
y_train = jnp.asarray(y_train)


@jit
def nonlin(x):
    return relu(x)


@jit
def softmax(x):
    return jnp.exp(x) / jnp.sum(jnp.exp(x), axis=-1, keepdims=True)


# fixed architecture
max_pool_fn = jit(partial(pool_fn, pool_size=7, reducer_fn=jnp.max))
# TODO: try deterministic CNN with


def BayesianCNN(X, y=None, num_cats=10):
    B, H, W, C = X.shape
    num_filters = [8]

    # Convolution + Max. Pooling
    units = (5, 5, num_filters[0])
    W1 = numpyro.sample("W1", dist.Normal(jnp.ones(units), jnp.ones(units)))
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(units[-1]), jnp.ones(units[-1])))
    out_conv1 = nonlin(convolution_fn(X, W1) + b1)
    assert out_conv1.shape == (B, H, W, num_filters[0])

    out_pool1 = max_pool_fn(out_conv1)
    assert out_pool1.shape == (B, H // 7, W // 7, num_filters[0])

    # Flatten
    b, *_ = out_pool1.shape
    out_flatten = jnp.ravel(out_pool1).reshape(b, -1)
    assert out_flatten.shape == (B, (H // 7) * (W // 7) * num_filters[-1])

    # Dense
    cells = (128, num_cats)
    W6 = numpyro.sample("W6", dist.Normal(jnp.zeros(cells), jnp.ones(cells)))
    b6 = numpyro.sample("b6", dist.Normal(jnp.zeros(num_cats), jnp.ones(num_cats)))
    out6 = jnp.matmul(out_flatten, W6) + b6
    assert out6.shape == (B, num_cats)

    logits = log_softmax(out6)
    numpyro.sample("y", dist.Categorical(logits=logits), obs=y)


rng_key = random.PRNGKey(876)
guide = AutoNormal(BayesianCNN)
svi = SVI(BayesianCNN, guide, Adam(0.05), loss=Trace_ELBO())
svi_result, _, elbo = svi.run(random.PRNGKey(0), 12000, X=X_train, y=y_train)
predictive = Predictive(BayesianCNN, params=svi_result, num_samples=50, parallel=True)

plt.plot(np.log(elbo))
plt.show()

# rng_key = random.PRNGKey(63547901)
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
#     BayesianCNN, posterior_samples=mcmc.get_samples(), num_samples=50, parallel=True
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
