##### set specific gpu #####
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow.keras as keras
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


a = tf.constant(1)

print(a.numpy())