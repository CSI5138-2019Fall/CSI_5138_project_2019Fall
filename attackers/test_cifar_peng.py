import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
############################
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model

from art import DATA_PATH
from art.utils import load_dataset, get_file
from art.classifiers import KerasClassifier
from art.attacks import  SaliencyMapMethod, ProjectedGradientDescent

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
    num_correct = np.sum(pred == gt).astype(np.float32)
    num_all = float(pred.shape[0])
    return num_correct / num_all


def GetAdvAccuracy(classifier, data_true, data_adv, y_true):
    """
    Function:
        Get accuracy loss, perturbation and time duration on the specific
        testing data (test set or adversarial set).
    """
    true_pred = classifier.predict(data_true)
    adv_pred = classifier.predict(data_adv)
    true_acc = AccCalculation(true_pred, y_true)
    adv_acc = AccCalculation(adv_pred, y_true)
    confidence_diff = true_acc - adv_acc
    perturbation = np.mean(np.abs(data_true - data_adv))
    print('Test acc: {:4.2f}%, adversarial acc: {:4.2f}%'.format(true_acc * 100, adv_acc * 100))
    print('Average Confidence lost: {:4.2f}%'.format(confidence_diff * 100))
    print('Average Image perturbation: {:4.2f}'.format(perturbation))
    return confidence_diff, perturbation


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

    path = get_file('cifar_resnet.h5', extract=False, path=DATA_PATH,
                    url='https://www.dropbox.com/s/ta75pl4krya5djj/cifar_resnet.h5?dl=1')
    classifier_model = load_model(path)

    # classifier_model.summary()
    return x_train, y_train, x_test, y_test, classifier_model, min_, max_


def GetAttackers(classifier, x_test, attacker_name):
    """
    Function:
        Load classifier and generate adversarial samples
    """
    t_start = time.time()

    if attacker_name == "SMM":
        attacker = SaliencyMapMethod(classifier=classifier, theta=.5, gamma=1.)
    elif attacker_name == "PGD":
        attacker = ProjectedGradientDescent(classifier=classifier, norm=1, eps=1, eps_step=0.5, max_iter=100,
                                            targeted=False, num_random_init=0, batch_size=1)
    else:
        raise ValueError("Please get the right attacker's name for the input.")
    test_adv = attacker.generate(x_test)
    dt = time.time() - t_start
    return test_adv, dt


def debug():

    x_train, y_train, x_test, y_test, model, min_, max_ = GetCifar10WithModel()
    x_test_example = x_test[:10]
    y_test_example = y_test[:10]

    classifier = KerasClassifier(clip_values=(min_, max_), model=model, use_logits=False,
                                 preprocessing=(0.5, 1))

    x_adv_SMM, dt_SMM = GetAttackers(classifier, x_test_example, "SMM")
    # np.save("samples/SMM_adv_mnist.npy", x_adv_SMM)
    x_adv_PGD, dt_PGD = GetAttackers(classifier, x_test_example, "PGD")
    # np.save("samples/PGD_adv_mnist.npy", x_adv_PGD)

    print("Time duration for SMM: \t", dt_SMM)
    print("Time duration for PGD: \t", dt_PGD)
    print("---------------------------------------------------------------------")
    conf_l_SMM, perturb_SMM = GetAdvAccuracy(classifier, x_test_example, x_adv_SMM, y_test_example)
    print("---------------------------------------------------------------------")
    conf_l_PGD, perturb_PGD = GetAdvAccuracy(classifier, x_test_example, x_adv_PGD, y_test_example)

if __name__ == "__main__":
    debug()