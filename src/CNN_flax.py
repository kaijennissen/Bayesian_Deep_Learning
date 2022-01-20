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

    with open("data/mnist_train.pickle", "rb") as file:
        train = pickle.load(file)
    with open("data/mnist_test.pickle", "rb") as file:
        test = pickle.load(file)

    x_train, y_train = train["image"] / 255, train["label"]
    x_test, y_test = test["image"] / 255, test["label"]

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

    # Plot images
    # yhat = model.apply(params, X_test)
    # yhat = jnp.argmax(yhat, axis=-1)
    # i = 0
    # imgs = [124, 12, 314, 718, 2, 22, 917, 513, 15]
    # fix, axes = plt.subplots(nrows=3, ncols=3)
    # for i, idx in enumerate(imgs):
    #     r = i // 3
    #     c = i % 3
    #     ax = axes[r, c]
    #     ax.imshow(X_test[idx])
    #     ax.set_title(f"Pred: {yhat[idx]}")
    #     i += 1
    # plt.show()
