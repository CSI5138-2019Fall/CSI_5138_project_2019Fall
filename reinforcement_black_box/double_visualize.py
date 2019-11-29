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
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.models import load_model

from environment import Environment
from agent_bb import BlackBoxAgent
from variety_agent import varietyAgent

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

def FirstAgentGetResults(agent, env, image_set, label_set, noise_epsilon, alpha, folder_name):
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
    plt.savefig(folder_name + 'one_agent.png')
    # plt.show()

def FirstAgentPlots(noise_epsilon, alpha, env, image_set, label_set, noise_type, folder_name):
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

    agent = BlackBoxAgent(image_shape, noise_epsilon, alpha, exploration_decay, noise_type)
    agent.LoadTables()

    FirstAgentGetResults(agent, env, image_set, label_set, noise_epsilon, alpha, folder_name)

def SecAgentGetResults(agent, image_set, folder_name):
    """
    Function:
        Get the results according to the current parameters.
    """
    fig = plt.figure(figsize = PLOT_SIZE)
    total_imgs = 10
    axes = np.zeros((1, total_imgs), dtype=np.object)
    count = 0
    feasible_img_inds = []
    for i in range(image_set.shape[0]):
        state = image_set[i]
        adv_sample, acc_loss = agent.VerifyAdvSample(state)
        if adv_sample is not None:
            feasible_img_inds.append(i)
            adv_sample = np.squeeze(adv_sample)
            axes[0, count] = fig.add_subplot(1, total_imgs, count+1)
            axes[0, count].imshow(adv_sample, cmap='gray')
            axes[0, count].axis('off')
            axes[0, count].set_title('acc loss:' + str(np.round(acc_loss, 2)), fontsize=12, fontweight='bold')
            count += 1
        if count >= total_imgs:
            break
    plt.savefig(folder_name + 'double_agent.png')
    # plt.show()
    return feasible_img_inds

def SecondAgentPlots(image_set, folder_name):
    """
    Function:
        Get the results of the second agent.
    """
    similarity_threshold = 0.1
    reward_thres = 0.85

    agent = varietyAgent(similarity_threshold, reward_thres)
    agent.LoadTables()

    img_inds = SecAgentGetResults(agent, image_set, folder_name)
    return img_inds

def OriginalPlot(image_set, folder_name):
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
    plt.savefig(folder_name + 'original.png')

def Plotting():
    """
    Function:
        Plot all adv samples.
    """
    # reset graph
    tf.keras.backend.clear_session()
    epsilon = 1.0 # max value of the images is 1.0
    alpha = 0.5
    noise_type = 'gaussian'

    batch_size = 1
    img_num = 50
    folder_name = 'plots_double/'
    env = Environment(batch_size)

    _, _, x_test, y_test = env.MnistDataset()
    x_select, y_select = ShuffleAndSelect(x_test, y_test, img_num)

    img_index = SecondAgentPlots(x_select, folder_name)
    x_select_new = x_select[img_index]
    y_select_new = y_select[img_index]

    OriginalPlot(x_select_new, folder_name)
    FirstAgentPlots(epsilon, alpha, env, x_select_new, y_select_new, noise_type, folder_name)

def Comparing():
    """
    Function:
        Read the saved results and compare the influence of different parameters.
    """
    all_imgs = []
    folder_name = 'plots_double/'

    original = imageio.imread(folder_name + 'original.png')
    oneag = imageio.imread(folder_name + 'one_agent.png')
    doubleag = imageio.imread(folder_name + 'double_agent.png')

    all_imgs = [original, oneag, doubleag]
    all_imgs = np.concatenate(all_imgs, axis = 0)

    fig = plt.figure(figsize = (18, 6))
    ax1 = fig.add_subplot(111)
    ax1.imshow(all_imgs)
    ax1.axis('off')
    plt.savefig(folder_name + "compare.png")
    plt.show()

if __name__ == "__main__":

    # Plotting()

    Comparing()