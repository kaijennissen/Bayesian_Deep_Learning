import time
import warnings

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

warnings.simplefilter("ignore", FutureWarning)


class CNN(nn.Module):
    """LeNet"""

    training: bool

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
        x = nn.Dropout(rate=0.2)(x, deterministic=not self.training)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x


def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def loss_fn(params, batch, rng_key):
    X, y = batch
    logits = model.apply(params, X, rngs={"dropout": rng_key})
    return jnp.mean(softmax_cross_entropy(logits, y))


@jax.jit
def update(params, opt_state, batch, rng_key):
    """Single SGD update step."""
    grads = jax.grad(loss_fn)(params, batch, rng_key)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


@jax.jit
def accuracy(params, x, y):
    y_hat = jnp.argmax(eval_model.apply(params, x), axis=1)
    return jnp.mean(y_hat == y)


def predict_probs(params, rng_key, images, num_samples: int = 10):
    probs = []
    for _ in range(num_samples):
        rng_key, subkey = jax.random.split(rng_key)
        probs.append(
            jax.nn.softmax(
                model.apply(params, images, rngs={"dropout": subkey}), axis=1
            )
        )
    predicted_probs = jnp.stack(probs)
    return predicted_probs


if __name__ == "__main__":
    MODEL = "CNN_dropout"

    x_train = np.load("data/mnist_train_images.npy") / 255
    y_train = np.load("data/mnist_train_labels.npy") * 1.0
    x_test = np.load("data/mnist_test_images.npy") / 255
    y_test = np.load("data/mnist_test_labels.npy") * 1.0

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
    epochs = 10
    batch_size = 100

    optimizer = optax.adam(learning_rate)
    model = CNN(training=True)
    eval_model = CNN(training=False)

    batch = jnp.ones((1, 28, 28, 1))  # (B, H, W, C) format
    params = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(34)}, batch
    )
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(2534)  # required for the dropout layer

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for batch in get_batches(batch_size):
            key, subkey = jax.random.split(key)
            params, opt_state = update(params, opt_state, batch, rng_key=subkey)
        epoch_time = round(time.time() - start_time, 2)

        train_acc = round(accuracy(params, x_train, y_train), 4)
        test_acc = round(accuracy(params, x_test, y_test), 4)
        print(
            f"Epoch {epoch} in {epoch_time} sec; Training set accuracy: {train_acc}; Test set accuracy: {test_acc}"
        )

    x_test = np.load("data/mnist_c_test_images.npy") / 255
    y_test = np.load("data/mnist_c_test_labels.npy") * 1.0

    np.random.seed(293)
    # imgs = [np.random.choice(np.where(y_test == i)[0]) for i in range(10)]
    #  imgs = [np.where(y_test == i)[0] for i in range(10)]
    # plt.imshow(x_test[imgs[6][3]]);plt.show()
    # plt.imshow(x_test[58])
    # plt.show()

    # 5: 182,319, 6:35
    imgs = [4542, 652, 9971, 747, 7629, 319, 259, 6609, 9936, 58]
    # imgs = [np.random.choice(np.where(y_test == i)[0]) for i in range(10)]
    num_images = len(imgs)

    fig, axes = plt.subplots(
        nrows=num_images,
        ncols=1,
        figsize=(4, 12),
    )
    sample_images = x_test[imgs, ...]
    true_labels = y_test[imgs, ...]
    for i in range(sample_images.shape[0]):
        image = sample_images[i]
        true_label = true_labels[i]
        ax1 = axes[i]
        # Show the image and the true label
        ax1.imshow(image[..., 0], cmap="gray")
        ax1.axis("off")
        ax1.set_title(f"#{imgs[i]}; True label: {str(true_label)}")
    fig.tight_layout()
    plt.savefig("plots/MNIST_C_samples.jpg", dpi=300)

    predicted_probs = predict_probs(
        params,
        rng_key=jax.random.PRNGKey(2314),
        images=sample_images,
        num_samples=20,
    )

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
    plt.savefig(f"plots/MNIST_C_{MODEL}.jpg", dpi=300)

    fig, axes = plt.subplots(
        nrows=num_images,
        ncols=1,
        figsize=(4, 12),
    )
    for i in range(sample_images.shape[0]):
        image = sample_images[i]
        true_label = true_labels[i]
        ax = axes[i]
        # Show a 95% prediction interval of model predicted probabilities
        pct_2p5 = np.array(
            [np.percentile(predicted_probs[:, i, n], 2.5) for n in range(10)]
        )
        pct_97p5 = np.array(
            [np.percentile(predicted_probs[:, i, n], 97.5) for n in range(10)]
        )
        bar = ax.bar(np.arange(10), pct_97p5 + 0.025, color="red")
        bar[int(true_label)].set_color("green")
        ax.bar(
            np.arange(10),
            pct_2p5 - 0.025,
            color="white",
            linewidth=1,
            edgecolor="white",
        )
        ax.set_xticks(np.arange(10))
        ax.set_ylim([0, 1])
        ax.set_ylabel("Probability")
    fig.suptitle("Model estimated probabilities")
    fig.tight_layout()
    plt.savefig(f"plots/MNIST_C_{MODEL}_probs.jpg", dpi=300)
