##### set specific gpu #####
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle

class varietyAgent(object):
    def __init__(self, similarity_threshold, reward_threshold):
        """ 
        Function:
            Initialization.
        """
        self.delta = similarity_threshold
        self.gamma = reward_threshold
        # gonna use np.array since it occupys less memory i believe
        self.adv_sample_table = None

    def VerifyReward(self, reward):
        """
        Function:
            Find out the current sample is a good sample or not.
        """
        ret = reward >= self.gamma
        return ret

    def SimilarToExistance(self, adv_sample):
        """
        Function:
            Find out if the current adversarial sample is similar
        to any of the adv samples inside the table.
        """
        exits = False
        for i in range(self.adv_sample_table.shape[0]):
            if np.max(np.abs(self.adv_sample_table[i] - adv_sample)) > self.delta:
                exits = True
                break
            else:
                exist = False
        return exits

    def UpdateAndReset(self, image, noise, reward):
        """
        Function:
            If it is worth storing, then store it into the table
        and reset the reward as 0.
        """
        if self.VerifyReward(reward):
            adv_sample = image + noise
            adv_sample = np.where(adv_sample < 0, 0., adv_sample)
            if self.adv_sample_table is None:
                self.adv_sample_table = adv_sample
            else:
                if not self.SimilarToExistance(adv_sample):
                    self.adv_sample_table = np.concatenate([self.adv_sample_table, adv_sample], axis = 0)
                    reward = 0.
                else:
                    reward = 0.
        else:
            reward = reward
        return reward




