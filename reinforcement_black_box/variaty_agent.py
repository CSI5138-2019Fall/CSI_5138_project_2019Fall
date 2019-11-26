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

    def UpdateAndReset(self, image, noise, reward):
        """
        Function:
            If it is worth storing, then store it into the table
        and reset the reward as 0.
        """
        if self.VerifyReward(reward):
            adv_sample = image + noise




