# Bayesian Deep Learning
The intention of this repository is to get a better understanding of Bayesian Deep Learning.
I've choosen to work with [jax](https://github.com/google/jax) and [numpyro](https://github.com/pyro-ppl/numpyro) because they provide the necessary tools (`HMC`, `SVI`, `grad`, `jit`) while, at the same time, they do this without abstractionn as it is the case with Tensorflow or PyTorch. F.e. it is possible to specify a (True) Bayesian Neural Network - HMC/NUTS and not Bayes by Backprob - in only 20 lines of code.

```python
import jax.numpy as jnp
from numpyro import sample, deterministic, plate
from numpyro.distributions import Normal, Gamma
from jax import jit, random


def BNN(X, y=None):

    N, k = X.shape
    D_H1 = 4
    D_H2 = 4

    # layer 1
    W1 = sample("W1", Normal(loc=jnp.zeros((k, D_H1)), scale=jnp.ones((k, D_H1))))
    b1 = sample("b1", Normal(loc=jnp.zeros(D_H1), scale=1.0))
    out1 = jnp.tanh(jnp.matmul(X, W1)) + b1

    # layer 2
    W2 = sample("W2", Normal(loc=jnp.zeros((D_H1, D_H2)), scale=jnp.ones((D_H1, D_H2))))
    b2 = sample("b2", Normal(loc=jnp.zeros(D_H2), scale=jnp.ones(D_H2)))
    out2 = jnp.tanh(jnp.matmul(out1, W2)) + b2

    # output layer
    W3 = sample("W3", Normal(loc=jnp.zeros((D_H2, 1)), scale=jnp.ones((D_H2, 1))))
    b3 = sample("b3", Normal(loc=jnp.zeros(1), scale=jnp.ones(1)))

    mean = deterministic("mean", jnp.matmul(out2, W3) + b3)
    prec_obs = sample("prec_obs", Gamma(3.0, 1.0))
    scale = 1.0 / jnp.sqrt(prec_obs)

    with plate("data", size=N, dim=-2):
        sample("y", Normal(loc=mean, scale=scale), obs=y)

```

## Aleatoric Uncertainty
#TODO add Two linear functions with different noise
![Bayesian Neural Net](./plots/MLP1_2021_12_19_10_41.jpg)
Consider f.e. the following plot where the crosses mark our observations and the orange line are out predictions.
Then it is obvious, that our predictions around x=0 will be close to the true observation (we are confident in our prediction) while for predictions outside this region our confidence will decline to to the higher noise in our observations.

This confidence can be captured by our model as depicted in the next plot. Here the shaded area denotes the 90%-confidece interval.
![Bayesian Neural Net](./plots/MLP3_2021_12_19_10_41.jpg)
## Epistemic Uncertainty
Uncertainty due to scarce data

![Bayesian Neural Net](./plots/BayesianDNN_2021_12_16_16_55.jpg)

![Gaussian Process](./plots/GaussianProcess_2021_12_15_11_18.jpg)

![Gaussian Process](./plots/GaussianProcess_2021_12_15_11_49.jpg)
