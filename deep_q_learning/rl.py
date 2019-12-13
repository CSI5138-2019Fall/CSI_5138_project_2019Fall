# -*- coding: utf-8 -*-
"""RL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JJSlqxNfeeFM4nxKXylrpJsn5J6RgFTX
"""

import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
import collections
import random
import matplotlib.pyplot as plt
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print("current TF version", tf.__version__)
print("is gpu available?", tf.test.is_gpu_available())

from scipy.ndimage.filters import gaussian_filter1d
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, ReLU, Dense, Input, Concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

PRETRAINED_MODEL = 'mnist_keras.h5'
AGENTS_DIR = 'agents/'
PLOTS_DIR = 'plots/'

class QLEnvironment:
    def __init__(self, 
                 batch_size=1,
                 num_imgs=100,
                 alpha=0.6):
        self.batch_size = batch_size
        self.num_imgs   = num_imgs
        self.x_train, self.y_train, self.x_test, self.y_test = self.get_mnist()
        self.model = self.get_pretrained_model()
        self.alpha = alpha
        self.num_classes = 10
    
    def get_mnist(self):
        """
        Function:
            Get Mnist Dataset.
        """
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        selected = (y_train == 7)
        x_train = x_train[selected]
        y_train = y_train[selected]
        # get the channel dimension
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        # x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_train = x_train.astype('float32')
        # x_test = x_test.astype('float32')
        x_train /= 255.
        # x_test /= 255.
        # test_index = np.arange(len(x_test))
        np.random.seed(11011)
        # np.random.shuffle(test_index)
        x_train = x_train[:self.num_imgs]
        y_train = y_train[:self.num_imgs]
        return x_train, y_train, x_test, y_test

    def get_pretrained_model(self):
        """
        Function:
            Get a pre-trained model for classification on Mnist.
        """
        path = PRETRAINED_MODEL
        model = load_model(path)
        return model
    
    def get_state(self):
        """
        Function:
            Generate a random image.
        """
        index = np.random.randint(len(self.x_train), size=self.batch_size)
        image = self.x_train[index]
        label = self.y_train[index]
        return image, label
    
    def get_reward(self, state, label, action):
        (adversarial_sample, noise) = action
        label_onehot = to_categorical(label, self.num_classes)
        prediction   = self.model.predict(adversarial_sample)
        # print('expected label', label, 'got', np.argmax(np.squeeze(prediction)))
        accuracy     = np.sum(label_onehot * prediction) / adversarial_sample.shape[0]
        score, accuracy_loss, noise_magn = self.__comput_score(state, accuracy, noise)
        return score, accuracy_loss, noise_magn
    
    def __comput_score(self, state, accuracy, noise):
        accuracy_loss = 1. - accuracy

        noise_magn = np.linalg.norm(np.squeeze(noise), ord=2)
        
        r = 0.
        if accuracy_loss > 0.7:
            r += (2.5 * accuracy_loss)
        else:
            r = accuracy_loss
        r += (1 - self.alpha) * self.__compute_noise_score(noise_magn)

        return r, accuracy_loss, noise_magn

    def __compute_noise_score(self, noise):
        return -1. * np.tanh(noise - 7.)


class DQN:
    def __init__(self,
                 input_shape,
                 learning_rate=0.001,
                 exp_rate=1.0,
                 min_exp_rate=0.01,
                 exp_decay=0.995,
                 batch_size=32):
        self.input_shape    = input_shape
        self.learning_rate  = learning_rate
        self.exp_rate       = exp_rate
        self.min_exp_rate   = min_exp_rate
        self.exp_decay      = exp_decay
        self.memory         = collections.deque(maxlen=100000)
        self.batch_size     = batch_size
        self.action_space   = [0.0, 0.2, 0.6, 0.8]
        self.action_space_unfair = [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.6, 0.6, 0.8, 0.8]
        
        self.agent = self.__create_agent()
        # self.agent.summary()
        
    def __create_agent(self):
        inp = Input(shape=self.input_shape)
        x   = Conv2D(kernel_size=3, filters=16, strides=2, padding='same', input_shape=self.input_shape)(inp)
        x   = BatchNormalization()(x)
        x   = ReLU()(x)
        x   = Conv2D(kernel_size=3, filters=32, padding='same')(x)
        x   = BatchNormalization()(x)
        x   = ReLU()(x)
        x   = Conv2D(kernel_size=3, filters=64, strides=2, padding='same')(x)
        x   = BatchNormalization()(x)
        x   = ReLU()(x)
        x   = Conv2DTranspose(kernel_size=3, filters=32, strides=2, padding='same')(x)
        x   = BatchNormalization()(x)
        x   = ReLU()(x)
        x   = Conv2DTranspose(kernel_size=3, filters=16, strides=2, padding='same')(x)
        x   = BatchNormalization()(x)
        x   = ReLU()(x)
        
        noise = Conv2D(kernel_size=1, filters=4)(x)
        
        agent = Model(inp, noise)
        
        agent.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return agent
    
    def act(self, env):
        state, label = env.get_state()
    
        expected_rewards = self.agent.predict(state)
        expected_rewards = np.squeeze(expected_rewards)
        
        adversarial_sample, noise, actions = self.__construct_adversarial_sample(state, expected_rewards)
        # print(adversarial_sample.shape, noise.shape, actions.shape)
        reward, accuracy_loss, noise_magn = env.get_reward(state, label, (adversarial_sample, noise))
        # print('Got reward of', reward, 'accuracy loss of', accuracy_loss)
        self.add_memory(state, actions, reward)
        
        return adversarial_sample, reward, accuracy_loss, noise_magn
        
    def __construct_adversarial_sample(self, state, expected_rewards):
        noise_map_shape = state.shape
        noise_map       = np.zeros(noise_map_shape)
        noise_map       = np.squeeze(noise_map)
        
        actions_w, actions_h = expected_rewards.shape[0], expected_rewards.shape[1]
        for i in range(actions_w):
            for j in range(actions_h):
                if np.random.rand() <= self.exp_rate: # explore
                    action1          = np.random.randint(low=0, high=len(self.action_space) - 1)
                    action2          = np.random.randint(low=0, high=len(self.action_space_unfair) - 1)
                    if np.random.rand() <= 0.6:
                        noise_map[i, j] = self.action_space_unfair[action2]
                        if noise_map[i,j] == 0.0:
                            action = 0
                        elif noise_map[i,j] == 0.2:
                            action = 1
                        elif noise_map[i,j] == 0.6:
                            action = 2
                        elif noise_map[i,j] == 0.8:
                            action = 3
                    else:
                        action = action1
                        noise_map[i, j] = self.action_space[action]
                else: # exploit
                    action         = np.argmax(expected_rewards[i,j])
                    noise_map[i,j] = self.action_space[action]
                    
                expected_rewards[i,j,:] = 0.
                expected_rewards[i,j,action] = 1.
        
        noise_map = np.reshape(noise_map, state.shape)
        adversarial_sample = state + noise_map
        adversarial_sample[adversarial_sample > 1.] = 1.
        return adversarial_sample, noise_map, expected_rewards
    
    def add_memory(self, state, action, reward):
        self.memory.append((state, action, reward))
        
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        X, y = [], []
        for state, action, reward in minibatch:
            target = np.expand_dims(reward * action, axis=0)
            X.append(np.squeeze(state, axis=0))
            y.append(np.squeeze(target, axis=0))
        self.agent.train_on_batch(x=np.array(X), y=np.array(y))

    def save_model(self, episode):
        # serialize model to JSON
        model_json = self.agent.to_json()
        with open(AGENTS_DIR + "agent.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.agent.save_weights(AGENTS_DIR + "agent.h5")
        print("Saved model to disk")

        
print("Current TF version is {0}".format(tf.__version__))
environment  = QLEnvironment()
dqn          = DQN((28, 28, 1), batch_size=32)

rewards, acc_losses, noise_magns = [], [], []
min_exp_iter = 2000

for i in range(1000000):
    adv, r, acc_loss, noise_magn = dqn.act(environment)
    rewards.append(r)
    acc_losses.append(acc_loss)
    noise_magns.append(noise_magn)
    if i > min_exp_iter:
        dqn.replay()
    if i % 100 == 0:
        print('done with {0} iterations, reward: {1}, accuracy loss: {2}, exploration rate: {3}'.format(i, np.round(r, 4), np.round(acc_loss, 4), np.round(dqn.exp_rate, 4)))
    if i is not 0 and i % min_exp_iter == 0:
      if dqn.exp_rate > dqn.min_exp_rate:
        dqn.exp_rate *= dqn.exp_decay
    if i is not 0 and i % 5000 == 0:
      dqn.save_model(i)

      plt.plot(gaussian_filter1d(rewards, sigma=i//100), label='reward')
      plt.plot(gaussian_filter1d(acc_losses, sigma=i//100), label='acc_loss')
      plt.legend()
      plt.savefig(PLOTS_DIR + 'reward_accloss.png')
      plt.clf()
      
      plt.plot(gaussian_filter1d(noise_magns, sigma=i//100), label='noise_magn')
      plt.legend()
      plt.savefig(PLOTS_DIR + 'noise_magn.png')
      plt.clf()
      
      plt.imshow(np.squeeze(adv))
      plt.savefig(PLOTS_DIR + 'adv_{0}_{1}.png'.format(i, np.round(acc_loss, 4)))
      plt.clf()

# sample, r, acc_loss, noise_magn = dqn.act(environment)

# print(r, acc_loss, noise_magn)

# plt.imshow(np.squeeze(sample))