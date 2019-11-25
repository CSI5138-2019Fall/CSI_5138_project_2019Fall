##### set specific gpu #####
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow.keras as keras
# tf.compat.v1.disable_eager_execution()
##### gpu memory management #####
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model

import numpy as np


class Environment(object):
    def __init__(self, batch_size):
        """
        Function:
            Initialization.
        """
        self.batch_size = batch_size
        self.num_imgs = 100
        self.x_train, self.y_train, self.x_test, self.y_test = self.MnistDataset()
        self.model = self.GetMnistPretrainedModel()

    def MnistDataset(self,):
        """
        Function:
            Get Mnist Dataset.
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
        test_index = np.arange(len(x_test))
        np.random.shuffle(test_index)
        x_test = x_test[test_index]
        x_test = x_test[:self.num_imgs]
        y_test = y_test[test_index]
        y_test = y_test[:self.num_imgs]
        return x_train, y_train, x_test, y_test

    def GetMnistPretrainedModel(self,):
        """
        Function:
            Get a pre-trained model for classification on Mnist.
        """
        path = "../pre_trained_models/mnist_keras.h5"
        model = load_model(path)
        return model

    def State(self,):
        """
        Function:
            Generate a random image.
        """
        index = np.random.randint(len(self.x_test), size=self.batch_size)
        image = self.x_test[index]
        label = self.y_test[index]
        # image = np.expand_dims(image, axis=0)
        # label = np.expand_dims(label, axis=0)
        return image, label

    def Reward(self, image, label):
        """
        Function:
            Get the accuracy 
        """
        num_classes = 10
        label_onehot = keras.utils.to_categorical(label, num_classes)
        prediction = self.model.predict(image)
        acc = np.sum(label_onehot*prediction) / image.shape[0]
        return 1. - acc




