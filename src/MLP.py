"""
Step 1: Build a simple (deterministic) MLP with JAX and Backpropagation
Step 2: Build a probabilistic MLP handling epistemic unvertainty i.e. learn
the parameters (mu and sigma) of a Gaussian distribution with first order optimization.
Step 3: Commpare results from 2 with a second order optimizer / natural gradient
Step 4: Add aleatoric uncertainty i.e. Bayesian neural network
"""

import jax.numpy as jnp
import numpy as np
from jax import grad, hessian, jit, vmap
