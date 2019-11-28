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
        self.original_img_table = {}
        self.adv_sample_table = {}

    def UpdateState(self, new_image):
        """
        Function:
            Update tables for previous unseen input data.
        """
        current_len_imgt = len(self.adv_sample_table)
        new_index = current_len_imgt
        new_key = 'img' + str(new_index)
        self.original_img_table[new_key] = new_image
        self.adv_sample_table[new_key] = {'adv': [], 
                                        'accl': [],
                                        'mag': []}

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

    def StateSearching(self, input_image):
        """
        Function:
            Search the image inside the teble and output keyname.
        """
        image_index = self.FindIndex(input_image, self.original_img_table, 'img')
        image_keyname = 'img' + str(image_index)
        return image_keyname

    def ExistInTable(self, value, table):
        """
        Function:
            Find out whether the value is in the table
        """
        return np.any([(value==y).all() for x,y in table.items()]) 

    def UpdateDict(self, image_keyname, adv_image, accl, magnitude):
        """
        Function:
            Literally update the table.
        """
        self.adv_sample_table[image_keyname]['adv'].append(adv_image)
        self.adv_sample_table[image_keyname]['accl'].append(accl)
        self.adv_sample_table[image_keyname]['mag'].append(magnitude)

    def SimilarToList(self, value, val_list):
        """
        Function:
            If the value similar to any value in the table.
        """
        exits = False
        for each in val_list:
            similarity = np.abs(each - value)
            simi_max = np.max(similarity)
            if simi_max < self.delta:
                exits = True
                break
            else:
                exits = False
        return exits

    def UpdateTable(self, image, noise, reward):
        """
        Function:
            Update the table and reset the reward if necessary.
        """
        if not self.ExistInTable(image, self.image_table):
            self.UpdateState(image)
        image_keyname = self.StateSearching(image)
        current_adv = np.clip(image + noise, 0., 1.)
        noise_magnitude = current_adv - image
        if len(self.adv_sample_table[image_keyname]['adv']) == 0:
            self.UpdateDict(image_keyname, current_adv, reward, noise_magnitude)
            exitance = False
        else:
            exitance = self.SimilarToList(current_adv, self.adv_sample_table[image_keyname]['adv'])
            if not exitance:
                self.UpdateDict(image_keyname, current_adv, reward, noise_magnitude)
        return 1 - exitance

    def PipeLine(self, image, noise, reward):
        """
        Function:
            Go through the pipeline and update the reward.
        """
        if reward > self.gamma:
            updated = self.UpdateTable(image, noise, reward)
            if updated:
                reward = 0
            else:
                reward = reward 
        else:
            reward = reward     
        return reward

