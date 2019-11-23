##### set specific gpu #####
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow.keras as keras
tf.compat.v1.disable_eager_execution()
##### gpu memory management #####
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model

import numpy as np
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
    exploration_decay = 0.8

    env = Environment(batch_size)
    agent = BlackBoxAgent(image_shape, noise_epsilon, similarity_threshold, exploration_decay)

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    x = []
    y1 = []
    y2 = []

    if load_tables:
        agent.LoadTables()

    for i in range(10000000000):
        state, state_label = env.State()
        action, noise = agent.GenerateAdvSample(state)
        reward = env.Reward(action, state_label)
        agent.UpdateTable(state, noise, reward)

        agent.UpdateExplorationRate()

        print(agent.agent_table.shape)
        x.append(i)
        y1.append(agent.agent_table.shape[0])
        y2.append(agent.agent_table.shape[1])

        if save_tables:
            if i % 1000 == 0:
                agent.SaveTables()

        ax1.clear()
        ax1.plot(x, y1, 'r')
        ax2.clear()
        ax2.plot(x, y2, 'b')
        fig.canvas.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    debug()