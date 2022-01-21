# functions
# https://neptune.ai/blog/bayesian-neural-networks-with-jax
# https://gitlab.com/awarelab/spin-up-with-variational-bayes/-/blob/master/bayes.py

import haiku as hk
import jax
import jax.numpy as jnp


def kl_gaussian(mu1, sigma1, mu2, sigma2):

    kl = (
        jnp.log(sigma2 / sigma1)
        + 0.5 * (sigma1 / sigma2) ** 2
        + 0.5 * ((mu1 - mu2) / sigma2) ** 2
        - 0.5
    )
    return kl


def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def net(images):
    mlp = hk.Sequential(
        [
            hk.Flatten(preserve_dims=1),
            hk.Linear(16),
            jax.nn.relu,
            hk.Linear(10),
        ]
    )
    logits = mlp(images)
    return logits


net_fn_t = hk.transform(net)
net_fn_t = hk.without_apply_rng(net_fn_t)

rng = jax.random.PRNGKey(42)
images = jnp.ones((1, 28, 28, 1))
labels = jnp.ones(1)
params = net_fn_t.init(rng, images)
net_fn_t.apply(params, images).shape

prior = {"mu": params, "logstd": jax.tree_map(lambda x: -7 * jnp.ones_like(x), params)}


def sample_params(dist, rng):
    def sample_gaussian(mu, logstd):
        eps = jax.random.normal(rng, shape=mu.shape)
        return eps * jnp.exp(logstd) + mu

    sample = jax.tree_multimap(sample_gaussian, dist["mu"], dist["logstd"])
    return sample


param_sample = sample_params(prior, jax.random.PRNGKey(543))
net_fn_t.apply(param_sample, images)

param_sample["linear"]["w"].shape
param_sample["linear_1"]["w"].shape


prior["logstd"]["linear"]["w"].shape


def predict(dist, seed, n_samples, x):
    predicts = []
    rng_seq = hk.PRNGSequence(seed)
    for _ in range(n_samples):
        param_sample = sample_params(dist, next(rng_seq))
        predicts.append(net_fn_t.apply(param_sample, images))
    stack_probs = jnp.stack(predicts)
    mean, var = jnp.mean(stack_probs, axis=0), jnp.var(stack_probs, axis=0)
    return mean, var


def elbo():
    # calculate kl loss
    pass


def sgd_update(params):
    # calc loss and update params
    pass


def calc_matrics():
    # calc metrics like accuracy etc
    pass
