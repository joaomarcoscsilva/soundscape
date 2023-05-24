import pickle
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import jax
import os
import optax
from soundscape import settings, dataset
import tensorflow_probability as tfp
from jax import numpy as jnp
from sklearn.metrics import confusion_matrix
import argparse

from soundscape.composition import Composable

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
