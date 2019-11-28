import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle

import tensorflow as tf
import tensorflow.keras as keras
# tf.compat.v1.disable_eager_execution()
# ##### gpu memory management #####
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.models import load_model

from environment import Environment
# from table import BlackBoxAgent
from agent_improv2 import BlackBoxAgent
# import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

PLOT_SIZE = (20, 2)

def ShuffleAndSelect(image_set, label_set, image_num):
    """
    Function:
        Shuffle the whole test set and select fixed num images
    """
    indexes = np.arange(len(image_set))
    np.random.shuffle(indexes)
    image_set = image_set[indexes]
    label_set = label_set[indexes]
    image_set_selected = image_set[:image_num]
    label_set_selected = label_set[:image_num]
    return image_set_selected, label_set_selected

def GetResults(agent, env, image_set, label_set, noise_epsilon, alpha):
    """
    Function:
        Get the results according to the current parameters.
    """
    fig = plt.figure(figsize = PLOT_SIZE)
    total_imgs = image_set.shape[0]
    axes = np.zeros((1, total_imgs), dtype=np.object)
    for i in range(total_imgs):
        axes[0, i] = fig.add_subplot(1, total_imgs, i+1)
        state = image_set[i]
        state_label = label_set[i]
        adv_noise = agent.VerifyAdvSample(state)
        if adv_noise is not None:
            adv_sample = state + adv_noise
            adv_sample = np.clip(adv_sample, 0., 1.)
            acc_loss = env.Reward(adv_sample, state_label)
            adv_sample = np.squeeze(adv_sample)
        axes[0, i].imshow(adv_sample, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('acc loss:' + str(np.round(acc_loss, 2)), fontsize=12, fontweight='bold')
    plt.savefig('plots/eps_' + str(np.round(noise_epsilon, 2)) \
                + '_alpha_' + str(np.round(alpha, 2)) + '.png')
    # plt.show()

def ResultsOnCurrentPara(noise_epsilon, alpha, env, image_set, label_set):
    """
    Function:
        Get the visualized results on current parameters.
    """
    # hyper-parameters settings
    batch_size = 1
    image_shape = (batch_size, 28, 28, 1)
    # noise_epsilon = 0.8 # max value of the images is 1.0
    exploration_decay = 0.8
    exploration_decay_steps = 800
    # alpha = 0.5

    agent = BlackBoxAgent(image_shape, noise_epsilon, alpha, exploration_decay)
    agent.LoadTables()

    GetResults(agent, env, image_set, label_set, noise_epsilon, alpha)

def OriginalPlot(image_set):
    """
    Function:
        Plot original set.
    """
    fig = plt.figure(figsize = PLOT_SIZE)
    total_imgs = image_set.shape[0]
    axes = np.zeros((1, total_imgs), dtype=np.object)
    for i in range(total_imgs):
        axes[0, i] = fig.add_subplot(1, total_imgs, i+1)
        state = image_set[i]
        sample = np.squeeze(state)
        axes[0, i].imshow(sample, cmap='gray')
        axes[0, i].axis('off')
    plt.savefig('plots/original.png')

def Plotting():
    """
    Function:
        Plot all adv samples.
    """
    # reset graph
    tf.keras.backend.clear_session()

    batch_size = 1
    img_num = 10
    env = Environment(batch_size)

    _, _, x_test, y_test = env.MnistDataset()
    x_select, y_select = ShuffleAndSelect(x_test, y_test, img_num)

    OriginalPlot(x_select)

    epsilons = [1.0, 0.9, 0.8]
    alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    for ind in tqdm(range(len(epsilons))):
        epsilon = epsilons[ind]
        for alpha in alphas:
            ResultsOnCurrentPara(epsilon, alpha, env, x_select, y_select)

def Comparing(mode):
    """
    Function:
        Read the saved results and compare the influence of different parameters.
    """
    epsilons = [1.0, 0.9, 0.8]
    alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    all_imgs = []
    original = imageio.imread('plots/original.png')
    all_imgs.append(original)
    if mode == "eps":
        for epsilon in epsilons:
            alpha = 0.8
            imgname = 'plots/eps_' + str(np.round(epsilon, 2)) \
                + '_alpha_' + str(np.round(alpha, 2)) + '.png'
            current_img = imageio.imread(imgname)
            all_imgs.append(current_img)
    elif mode == "alpha":
        for alpha in alphas:
            epsilon = 0.8
            imgname = 'plots/eps_' + str(np.round(epsilon, 2)) \
                + '_alpha_' + str(np.round(alpha, 2)) + '.png'
            current_img = imageio.imread(imgname)
            all_imgs.append(current_img)
    else:
        raise ValueError("------ wrong mode input, please doublecheck it. --------")
    
    all_imgs = np.concatenate(all_imgs, axis = 0)

    fig = plt.figure(figsize = (20, 20))
    ax1 = fig.add_subplot(111)
    ax1.imshow(all_imgs)
    ax1.axis('off')
    plt.savefig("plots/" + mode + ".png")
    plt.show()

if __name__ == "__main__":
    """
    mode: either "eps" or "alpha"
    """
    # Plotting()

    mode_name = "alpha"
    Comparing(mode_name)