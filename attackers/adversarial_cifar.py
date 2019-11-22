##### set specific gpu #####
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
############################
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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


def AccCalculation(y_pred, y_true):
    """
    Function:
        Calculate accuracy.
    """
    pred = np.argmax(y_pred, axis=1)
    gt = np.argmax(y_true, axis=1)
    num_correct = np.sum(y_pred[pred == gt]).astype(np.float32)
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
    print('Test acc: {:.4f}%, adversarial acc: {:.4f}%'.format(true_acc*100, adv_acc*100))
    print('Average Confidence lost: {:.4f}%'.format(confidence_diff * 100))
    print('Average Image perturbation: {:.4f}'.format(perturbation))
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

    path = get_file('cifar_resnet.h5',extract=False, path=DATA_PATH,
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
    if attacker_name == "FGSM":
        attacker = FastGradientMethod(classifier=classifier, eps=0.03)
    elif attacker_name == "Elastic":
        attacker = ElasticNet(classifier=classifier, confidence=0.5, batch_size=x_test.shape[0],)
                            # beta=1e-2, binary_search_steps=5, max_iter=20)
    elif attacker_name == "FGSM":
        attacker = FastGradientMethod(classifier=classifier, eps=0.05)
    elif attacker_name == "Elastic":
        attacker = ElasticNet(classifier=classifier, binary_search_steps=5, max_iter=20)
    elif attacker_name == "NewtonFool":
        attacker = NewtonFool(classifier=classifier, max_iter=20)
    elif attacker_name == "BasicIterativeMethod":
        attacker = BasicIterativeMethod(classifier=classifier, max_iter=20)
    elif attacker_name == "HopSkipJump":
        attacker = HopSkipJump(classifier=classifier, max_iter=20)
    elif attacker_name == "ZooAttack":
        attacker = ZooAttack(classifier=classifier, max_iter=20)
    elif attacker_name == "VirtualAdversarialMethod":
        attacker = VirtualAdversarialMethod(classifier=classifier, max_iter=20)
    elif attacker_name == "UniversalPerturbation":
        attacker = UniversalPerturbation(classifier=classifier, max_iter=20)
    elif attacker_name == "CarliniL2":
        attacker = CarliniL2Method(classifier=classifier, confidence=0.5, learning_rate=0.001, max_iter=15)
    elif attacker_name == "CarliniLinf":
        attacker = CarliniLInfMethod(classifier=classifier, confidence=0.5, learning_rate=0.001, max_iter=15)
    elif attacker_name == "DeepFool":
        attacker = DeepFool(classifier)
    elif attacker_name == "AdversarialPatch":
        attacker = AdversarialPatch(classifier=classifier, max_iter=20)
    elif attacker_name == "Attack":
        attacker = Attack(classifier=classifier)
    elif attacker_name == "BoundaryAttack":
        attacker = BoundaryAttack(classifier=classifier, targeted=False, epsilon=0.05, max_iter=20) #, max_iter=20
    elif attacker_name == "SMM":
        attacker = SaliencyMapMethod(classifier=classifier, theta=.5, gamma=1.)
    elif attacker_name == "PGD":
        attacker = ProjectedGradientDescent(classifier=classifier, norm=1, eps=1, eps_step=0.5, max_iter=100,
                                            targeted=False, num_random_init=0, batch_size=1)
    else:
        raise ValueError("Please get the right attacker's name for the input.")
    test_adv = attacker.generate(x_test)
    dt = time.time() - t_start
    return test_adv, dt

def debug(attacker_which):
    """
    Function:
        For debugging.
    For attacker_name:
        "FGSM"
        "Elastic"
    add your attacker's name here.
    """
    x_train, y_train, x_test, y_test, model, min_, max_ = GetCifar10WithModel()
    x_test_example = x_test[:1]
    y_test_example = y_test[:1]

    classifier = KerasClassifier(model=model, clip_values=(min_, max_))

    x_adv, dt = GetAttackers(classifier, x_test_example, attacker_which)
    np.save("samples/" + attacker_which + "_adv_mnist.npy", x_adv)

    # x_adv_fgsm, dt_fgsm = GetAttackers(classifier, x_test_example, "FGSM")
    # np.save("samples/FGSM_adv_cifar.npy", x_adv_fgsm)
    # x_adv_elastic, dt_elastic = GetAttackers(classifier, x_test_example, "Elastic")
    # np.save("samples/Elastic_adv_cifar.npy", x_adv_elastic)
    # print("Time duration for FGSM: \t", dt_fgsm)
    # print("Time duration for Elastic: \t", dt_elastic)
    # print("---------------------------------------------------------------------")
    # conf_l_fgsm, perturb_fgsm = GetAdvAccuracy(classifier, x_test_example, x_adv_fgsm, y_test_example)
    # print("---------------------------------------------------------------------")
    # conf_l_elast, perturb_elast = GetAdvAccuracy(classifier, x_test_example, x_adv_elastic, y_test_example)

if __name__ == "__main__":
    """
    attacker:
       "FGSM",
       "Elastic",
       "BasicIterativeMethod",
       "NewtonFool",
       "HopSkipJump",
       "ZooAttack",
       "VirtualAdversarialMethod",
       "UniversalPerturbation",
       # "AdversarialPatch",
       # "Attack",
       "BoundaryAttack",
       "CarliniL2",
       "CarliniLinf",
       "DeepFool",
       "SMM",
       "PGD",
    """
    attackers = ["FGSM",
       "Elastic",
       "BasicIterativeMethod",
       "NewtonFool",
       "HopSkipJump",
       "ZooAttack",
       "VirtualAdversarialMethod",
       "UniversalPerturbation",
       "BoundaryAttack",
       "CarliniL2",
       "CarliniLinf",
       "DeepFool",
       "SMM",
       "PGD",]
    for i in tqdm(range(len(attackers))):
        attacker = attackers[i]
        debug(attacker)