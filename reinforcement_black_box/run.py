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
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model

import numpy as np
import pickle
from tqdm import tqdm
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from environment import Environment
from table import BlackBoxAgent


def debug(load_tables=False, save_tables=True):
    # hyper-parameters settings
    batch_size = 1
    image_shape = (batch_size, 28, 28, 1)
    noise_epsilon = 0.1 # max value of the images is 1.0
    similarity_threshold = 0.01
    exploration_decay = 0.5
    exploration_decay_steps = 1000

    env = Environment(batch_size)
    agent = BlackBoxAgent(image_shape, noise_epsilon, similarity_threshold, exploration_decay)

    # set tensorboard
    log_dir = "logs"
    summary_writer = tf.summary.create_file_writer(log_dir)

    if load_tables:
        agent.LoadTables()

    acc_calculator = []

    for i in tqdm(range(10000000000)):
        state, state_label = env.State()
        action, noise = agent.GenerateAdvSample(state)
        reward = env.Reward(action, state_label)
        agent.UpdateTable(state, noise, reward)

        img_table_size = agent.agent_table.shape[0]
        noise_table_size = agent.agent_table.shape[1]

        acc_calculator.append(reward)

        with summary_writer.as_default():
            tf.summary.scalar('img_table_size', img_table_size, step=i)
            tf.summary.scalar('noise_table_size', noise_table_size, step=i)

        if (i+1) % exploration_decay_steps == 0:
            agent.UpdateExplorationRate()
            with summary_writer.as_default():
                tf.summary.scalar('exploration_rate', agent.exploration_rate, step=i)

        if save_tables:
            if i % 1000 == 0:
                agent.SaveTables()
        
        if len(acc_calculator) >= 100:
            acc_calculator = np.array(acc_calculator)
            acc = np.mean(acc_calculator)
            with summary_writer.as_default():
                tf.summary.scalar('average_confidence_loss', acc, step=i)
            acc_calculator = []
        else:
            continue


if __name__ == "__main__":
    debug()