import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle

from environment import Environment
# from table import BlackBoxAgent
from agent_improv import BlackBoxAgent
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt




if __name__ == "__main__":
    # hyper-parameters settings
    batch_size = 1
    image_shape = (batch_size, 28, 28, 1)
    noise_epsilon = 0.8 # max value of the images is 1.0
    exploration_decay = 0.9
    exploration_decay_steps = 500
    similarity_threshold = 0.01

    env = Environment(batch_size)
    agent = BlackBoxAgent(image_shape, noise_epsilon, similarity_threshold, exploration_decay)

    agent.LoadTables()

    fig = plt.figure(figsize=(20,4))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    count = 0
    originals = []
    adv_samples = []
    for i in range(100):
        state, state_label = env.State()
        adv_noise = agent.VerifyAdvSample(state)
        if adv_noise is not None:
            adv_sample = state + adv_noise
            originals.append(np.squeeze(state))
            adv_samples.append(np.squeeze(adv_sample))
            count += 1
        if count >= 20:
            break
    
    originals = np.concatenate(originals, axis=1)
    adv_samples = np.concatenate(adv_samples, axis=1)

    ax1.imshow(np.squeeze(originals), cmap='gray')
    ax1.axis('off')
    ax2.imshow(np.squeeze(adv_samples), cmap='gray')
    ax2.axis('off')
    plt.show()