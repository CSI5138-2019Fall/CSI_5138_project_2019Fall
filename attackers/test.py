##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
############################
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.models import load_model



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("---------------------------------------")
print(x_train.max())
print(x_train.min())


path = "../pre_trained_models/mnist_keras.h5"
classifier_model = load_model(path)
classifier_model.summary()