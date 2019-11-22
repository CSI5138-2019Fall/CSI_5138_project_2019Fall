##### set specific gpu #####
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
############################
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model
import logging

from art import DATA_PATH
from art.utils import load_dataset, get_file
from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod, ElasticNet
from art.attacks import NewtonFool,BasicIterativeMethod,HopSkipJump,ZooAttack,VirtualAdversarialMethod,UniversalPerturbation
from art.attacks.projected_gradient_descent import ProjectedGradientDescent
from art.attacks import AdversarialPatch, Attack, BoundaryAttack
from art.attacks.carlini import CarliniL2Method, CarliniLInfMethod
from art.attacks.deepfool import DeepFool
from art.attacks import  ProjectedGradientDescent, SaliencyMapMethod

import numpy as np
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def GetMnistWithModel():
    """
    Function:
        Load Mnist dataset and load a pre-trained mnist keras model.
    """
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # get the channel dimension
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    num_samples_train = 100
    num_samples_test = 100
    x_train = x_train[0:num_samples_train]
    y_train = y_train[0:num_samples_train]
    x_test = x_test[0:num_samples_test]
    y_test = y_test[0:num_samples_test]

    # classifier_model.summary()
    return x_train, y_train, x_test, y_test, 0., 1.

def GetCifar10WithModel():
    """
    Function:
        Load cifar-10 dataset and load a pre-trained cifar10 model.
    """
    (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('cifar10')
    num_samples_train = 100
    num_samples_test = 100
    x_train = x_train[0:num_samples_train]
    y_train = y_train[0:num_samples_train]
    x_test = x_test[0:num_samples_test]
    y_test = y_test[0:num_samples_test]

    class_descr = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


    # classifier_model.summary()
    return x_train, y_train, x_test, y_test, min_, max_

def Plot(dataset_name):
    """
    Function:
        for plotting the first image of the datset.
    """
    if dataset_name == "cifar":
        x_train, y_train, x_test, y_test, min_, max_ = GetCifar10WithModel()
    else:
        x_train, y_train, x_test, y_test, min_, max_ = GetMnistWithModel()

    x_test_example = x_test[:1]
    y_test_example = y_test[:1]

    img = np.squeeze(x_test_example)
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    if dataset_name == "mnist":
        ax1.imshow(img, cmap="gray")
    else:
        ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title("original", fontsize=45, fontweight='bold')
    plt.savefig(dataset_name + ".png")
    plt.show()

if __name__ == "__main__":
    dataset_name = 'mnist'
    Plot(dataset_name)