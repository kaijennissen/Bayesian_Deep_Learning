import pickle
import time
from copy import deepcopy
from typing import Callable, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import random
from jax.example_libraries import optimizers

Dist = Dict[str, jnp.ndarray]
NUM_CLASSES = 10
BETA = 0.001

# https://neptune.ai/blog/bayesian-neural-networks-with-jax
# https://gitlab.com/awarelab/spin-up-with-variational-bayes/-/blob/master/bayes.py


# @jax.jit
def gaussian_kl(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """Computes mean KL between parameterized Gaussian and Normal distributions.

    Gaussian parameterized by mu and logvar. Mean over the batch.

    NOTE: See Appendix B from VAE paper (Kingma 2014):
          https://arxiv.org/abs/1312.6114
    """

    var = jnp.exp(logvar)
    kl_divergence = jnp.sum(var + mu ** 2 - 1 - logvar) / 2
    kl_divergence /= mu.shape[0]

    return kl_divergence


# @jax.jit
def softmax_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])

    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot)


# @jax.jit
def accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    return jnp.argmax(logits, axis=1) == labels


@jax.jit
def sample_params(dist: Dist, key: jax.random.KeyArray) -> jnp.ndarray:
    def sample_gaussian(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
        eps = jax.random.normal(key, shape=mu.shape)
        return eps * jnp.exp(logvar / 2) + mu

    sample = jax.tree_multimap(sample_gaussian, dist["mu"], dist["logvar"])
    return sample


def calc_matrics(
    dist: Dist,
    # net_fn: hk.Transformed,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    key: jax.random.KeyArray,
    num_samples: int = 50,
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
        "accuracy": jnp.mean(jnp.argmax(probs, axis=1) == labels),
        "elbo": elbo_,
        "log_likelihood": log_likelihood,
        "kl_divergence": kl_divergence,
        "mean_approximate_evidence": mean_aprx_evidence,
    }


# Functions above are not model dependent, i.e. do not rely on global variables

# replace with CNN
def net(images: jnp.ndarray) -> jnp.ndarray:
    mlp = hk.Sequential(
        [
            hk.Flatten(preserve_dims=1),
            hk.Linear(64, name="linear_1"),
            jax.nn.relu,
            hk.Linear(64, name="linear_2"),
            jax.nn.relu,
            hk.Linear(10, name="linear_3"),
        ]
    )
    return mlp(images)


class CNN(hk.Module):
    def __init__(self):
        super().__init__(name="CNN")
        self.conv1 = hk.Conv2D(output_channels=6, kernel_shape=(5, 5))
        self.conv2 = hk.Conv2D(output_channels=16, kernel_shape=(5, 5))
        self.flatten = hk.Flatten()
        self.linear = hk.Linear(10)

    def __call__(self, x_batch):
        x = self.conv1(x_batch)
        # x = hk.max_pool(x, window_shape=(2, 2), strides=1, padding="VALID")
        x = jax.nn.relu(x)
        x = self.conv2(x)
        # x = hk.max_pool(x, window_shape=(2, 2), strides=1, padding="VALID")
        x = jax.nn.relu(x)
        x = self.flatten(x)
        # x = hk.Linear(128)(x)
        # x = jax.nn.relu(x)
        x = hk.Linear(64)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        # x = jax.nn.softmax(x)
        return x


def LeNet(x):
    cnn = CNN()
    return cnn(x)


def predict(
    dist: Dict,
    # net_fn: hk.Transformed,
    key: jax.random.KeyArray,
    num_samples: int,
    images: jnp.ndarray,
):
    stacked_probs = predict_probs(
        dist=dist, key=key, num_samples=num_samples, images=images
    )
    mean, var = jnp.mean(stacked_probs, axis=0), jnp.var(stacked_probs, axis=0)
    return mean, var


def predict_probs(
    dist: Dict,
    # net_fn: hk.Transformed,
    key: jax.random.KeyArray,
    num_samples: int,
    images: jnp.ndarray,
):
    probs = []

    for i in range(num_samples):
        key, subkey = random.split(key)
        param_sample = sample_params(dist, subkey)
        logits = net_fn_t.apply(param_sample, images)
        probs.append(jax.nn.softmax(logits))  # type:ignore

    stacked_probs = jnp.stack(probs)
    return stacked_probs


@jax.jit
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
    logits = net_fn_t.apply(params, images)  # type:ignore
    # compute log-likelihood
    log_likelihood = -softmax_cross_entropy(logits, labels)
    kl_divergence = jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_util.tree_multimap(gaussian_kl, dist["mu"], dist["logvar"]),
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
    elbo_, *_ = elbo(dist=dist, images=images, labels=labels, key=key)
    return -elbo_


@jax.jit
def sgd_update(
    dist: Dist,
    epoch: int,
    opt_state,
    # net_fn: hk.Transformed,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    key: jax.random.KeyArray,
) -> Tuple[Dist, optax.OptState]:
    #     # calc loss and update params
    #     grads = jax.grad(loss_fn)(opt_state, images, labels, key)
    #     updates, new_opt_state = optimizer.update(grads, opt_state)
    #     posterior = optax.apply_updates(params, updates)
    #     # Apply updates to parameters
    grads = jax.grad(loss_fn)(get_params(opt_state), images, labels, key)
    opt_state = opt_update(epoch, grads, opt_state)
    aprx_posterior = get_params(opt_state)
    return aprx_posterior, opt_state


if __name__ == "__main__":
    x_train = np.load("data/mnist_c_train_images.npy") / 255
    y_train = np.load("data/mnist_c_train_labels.npy") * 1.0
    x_test = np.load("data/mnist_c_test_images.npy") / 255
    y_test = np.load("data/mnist_c_test_labels.npy") * 1.0

    def get_batches(batch_size: int = 100):
        X = x_train.copy()
        y = y_train.copy()
        N = X.shape[0]

        idx = np.arange(N)
        np.random.shuffle(idx)
        splits = np.split(idx, N // batch_size)
        for split in splits:
            yield X[split, ...], y[split, ...]

    EPOCHS = 10
    BATCH_SIZE = 200
    LEARNING_RATE = 1e-3

    # Initialize Network
    net_fn_t = hk.transform(LeNet)
    net_fn_t = hk.without_apply_rng(net_fn_t)

    key = jax.random.PRNGKey(42)
    images = jnp.zeros((1, 28, 28, 1))
    labels = jnp.ones(1)
    params = net_fn_t.init(key, images)

    logvar = jax.tree_map(lambda x: -7.0 * jnp.ones_like(x), params)
    init_posterior = {"mu": params, "logvar": logvar}

    # TODO replace optax with jax optimizer
    opt_init, opt_update, get_params = optimizers.adam(LEARNING_RATE)
    opt_state = opt_init(init_posterior)

    rng_seq = hk.PRNGSequence(2686347)
    aprx_posterior = deepcopy(init_posterior)
    for epoch in range(1, EPOCHS + 1):

        for images, labels in get_batches(BATCH_SIZE):

            aprx_posterior, opt_state = sgd_update(
                dist=aprx_posterior,
                epoch=epoch,
                opt_state=opt_state,
                images=images,
                labels=labels,
                key=next(rng_seq),  # type: ignore
            )

        if epoch % 5 == 0:
            # train_metrics = calc_matrics(
            #     dist=aprx_posterior,
            #     images=x_train,
            #     labels=y_train,
            #     key=next(rng_seq),
            # )
            test_metrics = calc_matrics(
                dist=aprx_posterior,
                images=x_test,
                labels=y_test,
                key=next(rng_seq),  # type: ignore
            )
            print(f"Epoch: {epoch}")
            # print("Training metrics:")
            # for k, v in train_metrics.items():
            #     print(f"{k}: {v}")
            # print("\n")
            print("Test metrics:")
            for k, v in test_metrics.items():
                print(f"{k}: {v}")
            print("\n")

    # Plot images
    predicted_probs = predict_probs(
        dist=aprx_posterior,
        key=random.PRNGKey(2314),
        num_samples=100,
        images=x_test,
    )
    np.random.seed(293)
    y_hat = jnp.argmax(predicted_probs, axis=1)
    sample_images = x_test[y_test != y_hat, ...]
    sample_labels = y_test[y_test != y_hat]
    imgs = [np.random.choice(np.where(sample_labels == i)[0]) for i in range(10)]

    imgs = [np.random.choice(np.where(y_test == i)[0]) for i in range(10)]
    sample_images = x_test[imgs, ...]
    true_labels = y_test[imgs, ...]
    predicted_probs = predict_probs(
        dist=aprx_posterior,
        key=random.PRNGKey(2314),
        num_samples=100,
        images=sample_images,
    )
    num_images = len(imgs)
    fig, axes = plt.subplots(
        nrows=num_images,
        ncols=2,
        figsize=(14, 14),
        gridspec_kw={"width_ratios": [1, 4]},
    )
    for i in range(sample_images.shape[0]):
        image = sample_images[i]
        true_label = true_labels[i]
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        # Show the image and the true label
        ax1.imshow(image[..., 0], cmap="gray")
        ax1.axis("off")
        ax1.set_title("True label: {}".format(str(true_label)))

        # Show a 95% prediction interval of model predicted probabilities
        pct_2p5 = np.array(
            [np.percentile(predicted_probs[:, i, n], 2.5) for n in range(10)]
        )
        pct_97p5 = np.array(
            [np.percentile(predicted_probs[:, i, n], 97.5) for n in range(10)]
        )
        bar = ax2.bar(np.arange(10), pct_97p5, color="red")
        bar[int(true_label)].set_color("green")
        ax2.bar(
            np.arange(10),
            pct_2p5 - 0.05,
            color="white",
            linewidth=1,
            edgecolor="white",
        )
        ax2.set_xticks(np.arange(10))
        ax2.set_ylim([0, 1])
        ax2.set_ylabel("Probability")
        ax2.set_title("Model estimated probabilities")
    fig.tight_layout()
    plt.savefig(f"plots/MNIST_C_BCNN.jpg", dpi=300)
