##### set specific gpu #####
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow.keras as keras
# tf.compat.v1.disable_eager_execution()
# ##### gpu memory management #####
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model

import numpy as np
import pickle
from tqdm import tqdm

from environment import Environment
# from table import BlackBoxAgent
from agent_improv2 import BlackBoxAgent


def debug(noise_epsilon, alpha, load_tables=False, save_tables=True):
    # reset graph
    tf.keras.backend.clear_session()
    # hyper-parameters settings
    batch_size = 1
    image_shape = (batch_size, 28, 28, 1)
    # noise_epsilon = 0.8 # max value of the images is 1.0
    exploration_decay = 0.7
    exploration_decay_steps = 800
    # alpha = 0.5

    env = Environment(batch_size)
    agent = BlackBoxAgent(image_shape, noise_epsilon, alpha, exploration_decay)

    # set tensorboard
    log_dir = "logs_uniform/" + "nmax_" + str(noise_epsilon) + "_alpha_" + str(agent.alpha) + "_threshold_" + str(agent.reward_threshold)
    summary_writer = tf.summary.create_file_writer(log_dir)

    if load_tables:
        agent.LoadTables()

    acc_calculator = []

    i = -1
    while agent.exploration_rate >= 1e-2:
        i += 1
        state, state_label = env.State()
        action, noise = agent.GenerateAdvSample(state)
        reward = env.Reward(action, state_label)
        agent.UpdateTable(state, noise, reward)

        img_table_size = len(agent.image_table)
        noise_table_size = len(agent.noise_table)

        acc_calculator.append(reward)

        with summary_writer.as_default():
            tf.summary.scalar('img_table_size', img_table_size, step=i)
            tf.summary.scalar('noise_table_size', noise_table_size, step=i)

        if (not agent.decay_cmd):
            agent.IfDecay()

        if agent.decay_cmd and ((i+1) % exploration_decay_steps == 0):
            agent.UpdateExplorationRate()
            with summary_writer.as_default():
                tf.summary.scalar('exploration_rate', agent.exploration_rate, step=i)

        if save_tables:
            if i % 2000 == 0:
                agent.SaveTables()

        if len(acc_calculator) >= 20:
            acc_calculator = np.array(acc_calculator)
            acc = np.mean(acc_calculator)
            with summary_writer.as_default():
                tf.summary.scalar('average_confidence_loss', acc, step=i)
            acc_calculator = []
        
        if i >= 25000:
            break
        else:
            continue


if __name__ == "__main__":
    epsilons = [0.8, 0.7, 1.0, 0.6, 0.5, 0.9]
    alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    for ind in tqdm(range(len(epsilons))):
        epsilon = epsilons[ind]
        for alpha in alphas:
            debug(epsilon, alpha)