##### set specific gpu #####
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow.keras as keras
tf.compat.v1.disable_eager_execution()
##### gpu memory management #####
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model

import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from environment import Environment
from table import BlackBoxAgent


def debug(load_tables=False, save_tables=True):
    # hyper-parameters settings
    batch_size = 1
    image_shape = (batch_size, 28, 28, 1)
    noise_epsilon = 0.1 # max value of the images is 1.0
    similarity_threshold = 0.01
    exploration_decay = 0.8

    env = Environment(batch_size)
    agent = BlackBoxAgent(image_shape, noise_epsilon, similarity_threshold, exploration_decay)

    print(env.x_test.shape)


if __name__ == "__main__":
    debug()