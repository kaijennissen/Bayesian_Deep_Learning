import pickle
import time
from pickletools import optimize
from typing import Callable, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random
from jax.random import PRNGKey
from matplotlib.transforms import Transform
from numpy import ndarray

Dist = Dict[str, jnp.ndarray]
NUM_CLASSES = 10
BETA = 0.1

# https://neptune.ai/blog/bayesian-neural-networks-with-jax
# https://gitlab.com/awarelab/spin-up-with-variational-bayes/-/blob/master/bayes.py


@jax.jit
def gaussian_kl(mu: jnp.ndarray, logstd: jnp.ndarray) -> jnp.ndarray:
    """Computes mean KL between parameterized Gaussian and Normal distributions.

    Gaussian parameterized by mu and logvar. Mean over the batch.

    NOTE: See Appendix B from VAE paper (Kingma 2014):
          https://arxiv.org/abs/1312.6114
    """

    var = jnp.exp(logstd) ** 2
    logvar = 2 * logstd
    kl_divergence = jnp.sum(var + mu ** 2 - 1 - logvar) / 2
    kl_divergence /= mu.shape[0]

    return kl_divergence


@jax.jit
def softmax_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


@jax.jit
def accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    return jnp.argmax(logits, axis=1) == labels


def sample_params(dist: Dist, key: jax.random.KeyArray) -> jnp.ndarray:
    def sample_gaussian(mu: jnp.ndarray, logstd: jnp.ndarray) -> jnp.ndarray:
        eps = jax.random.normal(key, shape=mu.shape)
        return eps * jnp.exp(logstd) + mu

    sample = jax.tree_multimap(sample_gaussian, dist["mu"], dist["logstd"])
    return sample


def calc_matrics(
    dist: Dist,
    # net_fn: hk.Transformed,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    key: jax.random.KeyArray,
    num_samples: int = 100,
):
    # calc metrics like accuracy etc
    """Calculates metrics."""
    probs = predict(
        dist=dist, images=images, key=key, num_samples=num_samples  # net_fn=net_fn,
    )[0]
    elbo_, log_likelihood, kl_divergence = elbo(
        dist=dist, images=images, labels=labels, key=key  # net_fn=net_fn,
    )
    mean_aprx_evidence = jnp.exp(elbo_ / NUM_CLASSES)
    return {
        "accuracy": accuracy(probs, labels),
        "elbo": elbo_,
        "log_likelihood": log_likelihood,
        "kl_divergence": kl_divergence,
        "mean_approximate_evidence": mean_aprx_evidence,
    }


# Functions above are not model dependent, i.e. do not rely on global variables


def net(images: jnp.ndarray) -> jnp.ndarray:
    mlp = hk.Sequential(
        [
            hk.Flatten(preserve_dims=1),
            hk.Linear(16, name="linear_1"),
            jax.nn.relu,
            hk.Linear(10, name="linear_2"),
        ]
    )
    return mlp(images)


def predict(
    dist: Dict,
    # net_fn: hk.Transformed,
    key: jax.random.KeyArray,
    num_samples: int,
    images: jnp.ndarray,
):
    predicts = []

    for _ in range(num_samples):
        key, subkey = random.split(key)
        param_sample = sample_params(dist, subkey)
        predicts.append(net_fn_t.apply(param_sample, images))  # type:ignore
    stack_probs = jnp.stack(predicts)
    mean, var = jnp.mean(stack_probs, axis=0), jnp.var(stack_probs, axis=0)
    return mean, var


def elbo(
    dist: Dist,
    # net_fn: hk.Transformed,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    key: jax.random.KeyArray,
) -> Tuple:
    # sample params from approx. posterior
    params = sample_params(dist=dist, key=key)
    # get predictions from network with sampled params
    logits = net_fn_t.apply(dist, images)  # type:ignore
    # compute log-likelihood
    log_likelihood = softmax_cross_entropy(logits, labels)
    kl_divergence = jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_util.tree_multimap(gaussian_kl, params["mu"], params["logstd"]),
    )
    elbo_ = log_likelihood - BETA * kl_divergence
    return elbo_, log_likelihood, kl_divergence


def loss_fn(
    dist: Dist,
    # net_fn: hk.Transformed,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    key: jax.random.KeyArray,
):
    return -elbo(dist=dist, images=images, labels=labels, key=key)[0]  # net_fn=net_fn,


def sgd_update(
    dist: Dist,
    opt_state: optax.OptState,
    # net_fn: hk.Transformed,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    key: jax.random.KeyArray,
) -> Tuple[Dist, optax.OptState]:
    # calc loss and update params
    grads = jax.grad(loss_fn)(dist, images, labels, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    # Apply updates to parameters
    posterior = optax.apply_updates(dist, updates)
    return posterior, opt_state


with open("data/mnist_train.pickle", "rb") as file:
    train = pickle.load(file)
with open("data/mnist_test.pickle", "rb") as file:
    test = pickle.load(file)

x_train, y_train = train["image"] / 255, train["label"]
x_test, y_test = test["image"] / 255, test["label"]


epochs = 100
batch_size = 100
learning_rate = 1e-4


def get_batches(batch_size: int = 100):
    X = x_train.copy()
    y = y_train.copy()
    N = X.shape[0]

    idx = np.arange(N)
    np.random.shuffle(idx)
    splits = np.split(idx, N // batch_size)
    for split in splits:
        yield X[split, ...], y[split, ...]


# Initialize Network
net_fn_t = hk.transform(net)
net_fn_t = hk.without_apply_rng(net_fn_t)

key = jax.random.PRNGKey(42)
images = jnp.ones((8, 28, 28, 1))
params = net_fn_t.init(key, images)

aprx_posterior = dict(
    mu=params,
    logstd=jax.tree_map(lambda x: -7.0 * jnp.ones_like(x), params),
)

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(aprx_posterior)


# sanity checks
key = jax.random.PRNGKey(345)
images, labels = next(get_batches(20))
mean, var = predict(
    dist=aprx_posterior, key=key, num_samples=2, images=images  # net_fn=net_fn_t,
)
assert mean.shape == (20, NUM_CLASSES)
assert var.shape == (20, NUM_CLASSES)

params = sample_params(dist=aprx_posterior, key=key)
# assert params.shape == (20, NUM_CLASSES)
grads = jax.grad(loss_fn)(aprx_posterior, images, labels, key)

# rng_seq = hk.PRNGSequence(2134)
# for epoch in range(1, epochs + 1):
#     start_time = time.time()
#     for labels, images in get_batches(batch_size):

#         aprx_posterior, opt_state = sgd_update(
#             dist=aprx_posterior,
#             opt_state=opt_state,
#             net_fn=net_fn_t,
#             images=images,
#             labels=labels,
#             key=next(rng_seq),
#         )  # type: ignore

#     epoch_time = round(time.time() - start_time, 2)

#     logits_train = jnp.argmax(
#         predict(
#             dist=aprx_posterior,
#             net_fn=net_fn_t,
#             num_samples=100,
#             images=x_train,
#             key=next(rng_seq),
#         ),
#         axsi=1,
#     )
#     logits_test = jnp.argmax(
#         predict(
#             dist=aprx_posterior,
#             net_fn=net_fn_t,
#             num_samples=100,
#             images=x_test,
#             key=next(rng_seq),
#         ),
#         axsi=1,
#     )
#     train_acc = round(accuracy(params, x_train, y_train), 4)
#     test_acc = round(accuracy(params, x_test, y_test), 4)
#     print(
#         f"Epoch {epoch} in {epoch_time} sec; Training set accuracy: {train_acc}; Test set accuracy: {test_acc}"
#     )

# # Plot images
# # yhat = model.apply(params, X_test)
# # yhat = jnp.argmax(yhat, axis=-1)
# # i = 0
# # imgs = [124, 12, 314, 718, 2, 22, 917, 513, 15]
# # fix, axes = plt.subplots(nrows=3, ncols=3)
# # for i, idx in enumerate(imgs):
# #     r = i // 3
# #     c = i % 3
# #     ax = axes[r, c]
# #     ax.imshow(X_test[idx])
# #     ax.set_title(f"Pred: {yhat[idx]}")
# #     i += 1
# # plt.show()
