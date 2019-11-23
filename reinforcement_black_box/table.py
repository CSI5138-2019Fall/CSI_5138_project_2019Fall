##### set specific gpu #####
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings('ignore')

# import tensorflow as tf
# import tensorflow.keras as keras
# tf.compat.v1.disable_eager_execution()
# ##### gpu memory management #####
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# from tensorflow.keras.models import load_model

import numpy as np


class BlackBoxAgent(object):
    def __init__(self, image_shape, epsilon, similarity_threshold, exploration_decay):
        """
        Function:
            Initialization.
        """
        self.image_shape = image_shape
        self.epsilon = epsilon
        self.exploration_rate = 1.0
        self.exploration_decay = exploration_decay
        self.eps_dcimal_places = str(self.epsilon)[::-1].find('.')
        self.precision = 2
        self.image_table = {}
        self.noise_table = {}
        self.agent_table = np.zeros((1, 1))

    def UpdateExplorationRate(self,):
        """
        Function:
            Apply epsilon exploration decay.
        """
        self.exploration_rate *= self.exploration_decay

    def UpdateState(self, new_image):
        """
        Function:
            Update tables for previous unseen input data.
        """
        current_len_imgt = len(self.image_table)
        new_index = current_len_imgt
        new_key = 'img' + str(new_index)
        self.image_table[new_key] = new_image
        if new_index == 0:
            self.agent_table = self.agent_table
        else:
            new_row = np.zeros((1, self.agent_table.shape[1]))
            self.agent_table = np.concatenate([self.agent_table, new_row],
                                                axis = 0)
    
    def UpdateNoise(self,):
        """
        Function:
            Update tables for new noise.
        """
        current_len_noiset = len(self.noise_table)
        new_index = current_len_noiset
        new_key = 'noise' + str(new_index)

        new_noise = np.random.uniform(low=-self.epsilon, high=self.epsilon, 
                                        size=self.image_shape)
        new_noise = np.round(new_noise, self.eps_dcimal_places + self.precision)

        if new_index == 0:
            self.noise_table[new_key] = new_noise
            self.agent_table = self.agent_table
        else:
            while self.ExistInTable(new_noise, self.noise_table):
                new_noise = np.random.uniform(low=-self.epsilon, high=self.epsilon, 
                                        size=self.image_shape)
                new_noise = np.round(new_noise, self.eps_dcimal_places + self.precision)
            self.noise_table[new_key] = new_noise
            new_col = np.zeros((self.agent_table.shape[0], 1))
            self.agent_table = np.concatenate([self.agent_table, new_col],
                                                axis = 1)
        return new_noise

    def SelectNoise(self, row_index):
        """
        Function:
            Select the noise w.r.t the maximum reward.
        """
        noise_to_select = self.agent_table[row_index]
        noise_index = np.argmax(noise_to_select)
        noise_name = 'noise' + str(noise_index)
        noise = self.noise_table[noise_name]
        return noise

    def ExistInTable(self, value, table):
        """
        Function:
            Find out whether the value is in the table
        """
        return np.any([(value==y).all() for x,y in table.items()])

    def FindIndex(self, data, table, keywords):
        """
        Function: 
            Find the key according to the value in a dictionary.
        """
        finding = [x for x,y in table.items() if (y==data).all()]
        found = finding[0]
        index = found.replace(keywords, '')
        index = int(index)
        return index

    def GenerateAdvSample(self, input_image):
        """
        Function:
            Agent is working.
        """
        if not self.ExistInTable(input_image, self.image_table):
            self.UpdateState(input_image)
        #     print("-------------",len(self.image_table))
        # else:
        #     print("-------------, not updated.")

        row_index = self.FindIndex(input_image, self.image_table, 'img')
        
        exploring = np.random.uniform(low=0.0, high=1.0)
        if (exploring < self.exploration_rate) or (self.agent_table[row_index].max() == 0.):
            noise = self.UpdateNoise()
        else:
            noise = self.SelectNoise(row_index)

        adv_sample = input_image + noise
        return adv_sample, noise
    
    def UpdateTable(self, input_image, noise, reward):
        """
        Function:
            Update Agent table according to the reward from the environment.
        """
        row_index = self.FindIndex(input_image, self.image_table, 'img')
        col_index = self.FindIndex(noise, self.noise_table, 'noise')
        if (reward >= self.agent_table[row_index, col_index]):
            self.agent_table[row_index, col_index] = reward


            

