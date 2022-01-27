import warnings

warnings.simplefilter("ignore", FutureWarning)
import pickle
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd


class CNN(nn.Module):
    """LeNet"""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=6, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x


def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def loss_fn(params, batch):
    X, y = batch
    logits = model.apply(params, X)
    return jnp.mean(softmax_cross_entropy(logits, y))


@jax.jit
def update(params, opt_state, batch):

    """Single SGD update step."""
    grads = jax.grad(loss_fn)(params, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


@jax.jit
def accuracy(params, x, y):
    y_hat = jnp.argmax(model.apply(params, x), axis=1)
    return jnp.mean(y_hat == y)


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

    learning_rate = 1e-4
    epochs = 20
    batch_size = 100

    optimizer = optax.adam(learning_rate)
    model = CNN()

    batch = jnp.ones((1, 28, 28, 1))  # (B, H, W, C) format
    params = model.init(jax.random.PRNGKey(0), batch)
    opt_state = optimizer.init(params)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for batch in get_batches(batch_size):
            params, opt_state = update(params, opt_state, batch)
        epoch_time = round(time.time() - start_time, 2)

        train_acc = round(accuracy(params, x_train, y_train), 4)
        test_acc = round(accuracy(params, x_test, y_test), 4)
        print(
            f"Epoch {epoch} in {epoch_time} sec; Training set accuracy: {train_acc}; Test set accuracy: {test_acc}"
        )

        np.random.seed(293)

        predicted_probs = jax.nn.softmax(model.apply(params, x_test), axis=1)
        y_hat = jnp.argmax(predicted_probs, axis=1)
        sample_images = x_test[y_test != y_hat, ...]
        sample_labels = y_test[y_test != y_hat]

        imgs = [np.random.choice(np.where(sample_labels == i)[0]) for i in range(10)]

        sample_images = sample_images[imgs, ...]
        true_labels = sample_labels[imgs, ...]
        predicted_probs = predicted_probs[imgs, ...]

        num_images = len(imgs)
        print(imgs)
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
                [np.percentile(predicted_probs[i, n], 2.5) for n in range(10)]
            )
            pct_97p5 = np.array(
                [np.percentile(predicted_probs[i, n], 97.5) for n in range(10)]
            )
            bar = ax2.bar(np.arange(10), pct_97p5 + 0.025, color="red")
            bar[int(true_label)].set_color("green")
            ax2.bar(
                np.arange(10),
                pct_2p5 - 0.025,
                color="white",
                linewidth=1,
                edgecolor="white",
            )
            ax2.set_xticks(np.arange(10))
            ax2.set_ylim([0, 1])
            ax2.set_ylabel("Probability")
            ax2.set_title("Model estimated probabilities")
        fig.tight_layout()
        plt.savefig(f"plots/MNIST_C_CNN.jpg", dpi=300)
