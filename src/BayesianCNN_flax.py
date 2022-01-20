import warnings

warnings.simplefilter("ignore", FutureWarning)
import flax.linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from numpyro.contrib.module import random_flax_module
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
from scipy.stats import mode


class CNN(nn.Module):
    """LeNet"""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=8, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(6, 6), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x


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


def BayesianCNN(X, y=None):
    N, *_ = X.shape
    module = CNN()
    net = random_flax_module(
        "nn", module, dist.Normal(0.0, 10.0), input_shape=(2, 28, 28, 1)
    )
    # if y is not None:
    #     assert y.shape == (N,)
    # assert logits.shape == (N, 10)
    logits = net(X)
    numpyro.sample("y", dist.Categorical(logits=logits), obs=y)


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train.pop("label").values
    y_train_onehot = (y[:, np.newaxis] == np.arange(10)) * 1
    X = train.values.reshape(-1, 28, 28, 1) / 255
    X_train, y_train = stratified_samples(X, y, size=1000)

    X_train = jnp.asarray(X)  # type:ignore
    y_train = jnp.asarray(y)

    rng_key = random.PRNGKey(125)

    # guide = AutoNormal(BayesianCNN)
    # svi = SVI(BayesianCNN, guide, Adam(0.05), loss=Trace_ELBO())
    # svi_result, _, elbo = svi.run(random.PRNGKey(0), 8000, X=X_train, y=y_train)
    # predictive = Predictive(
    #     BayesianCNN, params=svi_result, num_samples=100, parallel=True
    # )

    rng_key, rng_key_predict = random.split(random.PRNGKey(915))
    kernel = NUTS(BayesianCNN, max_tree_depth=1)
    mcmc = MCMC(
        kernel,
        num_warmup=1000,
        num_samples=4000,
        num_chains=1,
        chain_method="vectorized",
    )
    mcmc.run(X=X, y=y, rng_key=rng_key)

    predictive = Predictive(
        BayesianCNN,
        posterior_samples=mcmc.get_samples(),
        num_samples=500,
        parallel=True,
    )

    post_samples = predictive(rng_key=rng_key, X=X_train[:1000, ...])

    yhat, count = mode(post_samples["y"], axis=0)
    yhat = yhat.ravel()
    acc = np.mean(yhat == y_train[:1000, ...])

    print(f"Accuraccy: {acc}")
    i = 0
    imgs = [124, 12, 314, 718, 2, 22, 917, 513, 15]
    fix, axes = plt.subplots(nrows=3, ncols=3)
    for i, idx in enumerate(imgs):
        r = i // 3
        c = i % 3
        ax = axes[r, c]
        ax.imshow(X_train[idx])
        ax.set_title(f"Pred: {yhat[idx]}")
        i += 1
    plt.show()
