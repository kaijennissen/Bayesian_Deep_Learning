from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
y = train.pop("label").values
y_train_onehot = (y[:, np.newaxis] == np.arange(10)) * 1
X = train.values.reshape(-1, 28, 28, 1) / 255


def make_dataset(seed: int, num_batches: int, batch_size: int):
    key = jax.random.PRNGKey(seed)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    y = train.pop("label").values
    y_train_onehot = (y[:, np.newaxis] == np.arange(10)) * 1
    X = train.values.reshape(-1, 28, 28, 1) / 255
    for _ in range(num_batches):
        key, subkey = jax.random.split(key, num=2)
        idx = jax.random.choice(key=subkey, a=y.shape[0], shape=(batch_size,))
        yield jnp.asarray(X[idx, ...]), jnp.asarray(y[idx, ...])


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=6, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=120)(x)
        x = nn.relu(x)
        x = nn.Dense(features=84)(x)
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


# TODO: split into train / test / validation datasets
learning_rate = 1e-4
epochs = 1000
train_ds = make_dataset(23, epochs, 64)
valid_ds = make_dataset(3125, epochs, 64)

optimizer = optax.adam(learning_rate)
model = CNN()

batch = jnp.ones((1, 28, 28, 1))  # (B, H, W, C) format
params = model.init(jax.random.PRNGKey(0), batch)
opt_state = optimizer.init(params)


for epoch, batch in enumerate(train_ds):
    params, opt_state = update(params, opt_state, batch)

    if epoch % 100 == 0:
        val_batch = next(valid_ds)
        val_loss = loss_fn(params, val_batch)
        print(f"Validation loss after epoch {epoch}: {val_loss}")


yhat = model.apply(params, X)
yhat = jnp.argmax(yhat, axis=-1)
acc = jnp.mean(yhat == y.ravel())
print(f"Accuraccy: {acc}")
# TODO: add plots
