##### set specific gpu #####
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle


class BlackBoxAgent(object):
    def __init__(self, image_shape, epsilon, alpha, exploration_decay):
        """
        Function:
            Initialization.
        """
        self.image_shape = image_shape
        self.epsilon = epsilon
        self.alpha = alpha
        self.exploration_rate = 1.0
        self.exploration_decay = exploration_decay
        self.eps_dcimal_places = str(self.epsilon)[::-1].find('.')
        self.precision = 2
        # self.reward_threshold = 0.5*self.epsilon - (1-self.alpha)*0.5
        self.reward_threshold = -100
        self.decay_threshold = 0.6
        self.decay_cmd = False
        self.image_table = {}
        self.noise_table = {}
        self.agent_table = {}

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
        self.agent_table[new_key] = {'noise': [], 
                                    'reward': [0.]}

    def UpdateNoiseTable(self, new_noise):
        """
        Function:
            Update tables for previous unseen noise data.
        """
        current_len_noiset = len(self.noise_table)
        new_index = current_len_noiset
        new_key = 'noise' + str(new_index)
        self.noise_table[new_key] = new_noise
        return new_key

    def StateSearching(self, input_image):
        """
        Function:
            Search the image inside the teble and output keyname.
        """
        image_index = self.FindIndex(input_image, self.image_table, 'img')
        image_keyname = 'img' + str(image_index)
        return image_keyname

    def NoiseSearching(self, input_noise):
        """
        Function:
            Search the noise inside the table and output keyname.
        """
        noise_index = self.FindIndex(input_noise, self.noise_table, 'noise')
        noise_keyname = 'noise' + str(noise_index)
        return noise_keyname
    
    def UpdateNoise(self, image_keyname):
        """
        Function:
            Update tables for new noise.
        """
        # new_noise = np.random.uniform(low=-self.epsilon, high=self.epsilon, 
        #                                 size=self.image_shape)
        # new_noise = np.random.uniform(low=0., high=self.epsilon, 
        #                                 size=self.image_shape)
        # new_noise = np.round(new_noise, self.eps_dcimal_places + self.precision)
        new_noise = np.random.normal(loc=0., scale=self.epsilon * 0.5, 
                                        size=self.image_shape)
        new_noise = np.abs(new_noise)
        new_noise = np.round(new_noise, self.eps_dcimal_places + self.precision)

        if len(self.agent_table[image_keyname]['noise']) == 0:
            noise_list_tmp = []
        else:
            noise_list_tmp = [self.noise_table[x] for x in self.agent_table[image_keyname]['noise']]

        while self.ExistanceInList(new_noise, noise_list_tmp):
            # new_noise = np.random.uniform(low=-self.epsilon, high=self.epsilon, 
            #                         size=self.image_shape)
            # new_noise = np.random.uniform(low=0., high=self.epsilon, 
            #                         size=self.image_shape)
            # new_noise = np.round(new_noise, self.eps_dcimal_places + self.precision)
            new_noise = np.random.normal(loc=0., scale=self.epsilon * 0.5, 
                                    size=self.image_shape)
            new_noise = np.abs(new_noise)
            new_noise = np.round(new_noise, self.eps_dcimal_places + self.precision)

        if self.ExistInTable(new_noise, self.noise_table):
            noise_keyname = self.NoiseSearching(new_noise)
        else:
            noise_keyname = self.UpdateNoiseTable(new_noise)
        self.agent_table[image_keyname]['noise'].append(noise_keyname)

        if len(self.agent_table[image_keyname]['noise']) > len(self.agent_table[image_keyname]['reward']):
            self.agent_table[image_keyname]['reward'].append(0.)
        assert len(self.agent_table[image_keyname]['noise']) == len(self.agent_table[image_keyname]['reward'])
        return new_noise

    def SelectNoise(self, image_keyname):
        """
        Function:
            Select the noise w.r.t the maximum reward.
        """
        reward_list = self.agent_table[image_keyname]['reward']
        noise_index = np.argmax(reward_list)
        noise_key = self.agent_table[image_keyname]['noise'][noise_index]
        noise = self.noise_table[noise_key]
        return noise

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

    def ExistanceInList(self, value, value_list):
        """
        Function:
            Find out whether the value is in the table
        """
        return np.any([(value==y).all() for y in value_list])
    
    def ExistInTable(self, value, table):
        """
        Function:
            Find out whether the value is in the table
        """
        return np.any([(value==y).all() for x,y in table.items()]) 

    def IfDecay(self,):
        """
        Function:
            Give the cmd whether start to apply exploration decay.
        """
        all_boolean = []
        for i in range(len(self.agent_table)):
            boolean_mask = np.max(self.agent_table['img' + str(i)]['reward']) > self.reward_threshold
            all_boolean.append(boolean_mask)
        all_boolean = np.array(all_boolean).astype(np.float32)
        if all_boolean.mean() > self.decay_threshold:
            self.decay_cmd = True
        else:
            self.decay_cmd = False

    def GenerateAdvSample(self, input_image):
        """
        Function:
            Agent is working.
        """
        if not self.ExistInTable(input_image, self.image_table):
            self.UpdateState(input_image)

        image_keyname = self.StateSearching(input_image)
        
        exploring = np.random.uniform(low=0.0, high=1.0)
        if (exploring < self.exploration_rate) or \
            (np.max(self.agent_table[image_keyname]['reward']) < self.reward_threshold):
            noise = self.UpdateNoise(image_keyname)
        else:
            noise = self.SelectNoise(image_keyname)

        adv_sample = input_image + noise
        adv_sample = np.clip(adv_sample, 0., 1.)
        return adv_sample, noise

    def VerifyAdvSample(self, input_image):
        """
        Function:
            Verify the agent we got by generating the samples.
        """
        if not self.ExistInTable(input_image, self.image_table):
            print("----- The current image has never been seen before. -----")
            noise = None
        else:
            image_keyname = self.StateSearching(input_image)
            noise = self.SelectNoise(image_keyname)
        return noise

    def UpdateTable(self, input_image, noise, reward):
        """
        Function:
            Update Agent table according to the reward from the environment.
        """
        image_keyname = self.StateSearching(input_image)
        noise_list_tmp = [self.noise_table[x] for x in self.agent_table[image_keyname]['noise']]
        index = [np.array_equal(noise,x) for x in noise_list_tmp].index(True)
        if (reward >= self.agent_table[image_keyname]['reward'][index]):
            self.agent_table[image_keyname]['reward'][index] = self.alpha * reward - \
                    (1 - self.alpha) * np.mean(np.abs(noise_list_tmp[index]))

    def SaveTables(self,):
        """
        Function:
            Save all tables and dictionaries
        """
        print("----------- saving image table -----------")
        with open("tables/image_table" + str(self.epsilon) + str(self.alpha) + str(self.reward_threshold) + ".pickle", "wb") as f_img:
            pickle.dump(self.image_table, f_img)
        print("----------- saving noise table -----------")
        with open("tables/noise_table" + str(self.epsilon) + str(self.alpha) + str(self.reward_threshold) + ".pickle", "wb") as f_noise:
            pickle.dump(self.noise_table, f_noise)
        print("----------- saving agent table -----------")
        with open("tables/agent_table" + str(self.epsilon) + str(self.alpha) + str(self.reward_threshold) + ".pickle", "wb") as f_agent:
            pickle.dump(self.agent_table, f_agent)

    def LoadTables(self,):
        """
        Function:
            Load all saved tables.
        """
        print("----------- loading image table -----------")
        with open("tables/image_table" + str(self.epsilon) + str(self.alpha) + str(self.reward_threshold) + ".pickle", "rb") as f_img:
            self.image_table = pickle.load(f_img)
        print("----------- loading noise table -----------")
        with open("tables/noise_table" + str(self.epsilon) + str(self.alpha) + str(self.reward_threshold) + ".pickle", "rb") as f_noise:
            self.noise_table = pickle.load(f_noise)
        print("----------- loading noise table -----------")
        with open("tables/agent_table" + str(self.epsilon) + str(self.alpha) + str(self.reward_threshold) + ".pickle", "rb") as f_agent:
            self.agent_table = pickle.load(f_agent)

            

