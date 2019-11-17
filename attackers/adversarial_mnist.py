##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow.keras as keras
tf.compat.v1.disable_eager_execution()
############################
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model

from art import DATA_PATH
from art.utils import load_dataset, get_file
from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod, ElasticNet

import numpy as np
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import time

def AccCalculation(y_pred, y_true):
    """
    Function:
        Calculate accuracy.
    """
    pred = np.argmax(y_pred, axis=1)
    gt = np.argmax(y_true, axis=1)
    num_correct = np.sum((pred == gt).astype(np.float32))
    num_all = float(pred.shape[0])
    return num_correct / num_all

def GetAdvAccuracy(classifier, data_true, data_adv, y_true):
    """
    Function:
        Get accuracy loss, perturbation and time duration on the specific 
        testing data (test set or adversarial set).
    """
    num_classes = 10
    y_true = keras.utils.to_categorical(y_true, num_classes)

    true_pred = classifier.predict(data_true)
    adv_pred = classifier.predict(data_adv)
    true_acc = AccCalculation(true_pred, y_true)
    adv_acc = AccCalculation(adv_pred, y_true)

    confidence_diff = true_acc - adv_acc

    perturbation = np.mean(np.abs(data_true - data_adv))
    print('Test acc: {:4.2f}%, adversarial acc: {:4.2f}%'.format(true_acc*100, adv_acc*100))
    print('Average Confidence lost: {:4.2f}%'.format(confidence_diff * 100))
    print('Average Image perturbation: {:4.2f}'.format(perturbation))
    return confidence_diff, perturbation

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

    path = "../pre_trained_models/mnist_keras.h5"
    classifier_model = load_model(path)

    # classifier_model.summary()
    return x_train, y_train, x_test, y_test, classifier_model, 0., 1.

def GetAttackers(classifier, x_test, attacker_name):
    """
    Function:
        Load classifier and generate adversarial samples
    """
    t_start = time.time()
    if attacker_name == "FGSM":
        attacker = FastGradientMethod(classifier=classifier, eps=0.3)
    elif attacker_name == "Elastic":
        attacker = ElasticNet(classifier=classifier, binary_search_steps=5, max_iter=20)
    else:
        raise ValueError("Please get the right attacker's name for the input.")
    test_adv = attacker.generate(x_test)
    dt = time.time() - t_start
    return test_adv, dt

def debug():
    """
    Function:
        For debugging.
    For attacker_name:
        "FGSM"
        "Elastic"
    add your attacker's name here.
    """
    x_train, y_train, x_test, y_test, model, min_, max_ = GetMnistWithModel()
    x_test_example = x_test[:10]
    y_test_example = y_test[:10]

    classifier = KerasClassifier(model=model, clip_values=(min_, max_))
    
    x_adv_fgsm, dt_fgsm = GetAttackers(classifier, x_test_example, "FGSM")
    x_adv_elastic, dt_elastic = GetAttackers(classifier, x_test_example, "Elastic")
    print("Time duration for FGSM: \t", dt_fgsm)
    print("Time duration for Elastic: \t", dt_elastic)
    print("-----------------------------------------------------------------------------------------")
    conf_l_fgsm, perturb_fgsm = GetAdvAccuracy(classifier, x_test_example, x_adv_fgsm, y_test_example)
    print("-----------------------------------------------------------------------------------------")
    conf_l_elast, perturb_elast = GetAdvAccuracy(classifier, x_test_example, x_adv_elastic, y_test_example)

if __name__ == "__main__":
    debug()