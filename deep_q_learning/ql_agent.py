
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
import collections

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, ReLU, Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from keract import get_activations

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
        # get the channel dimension
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255.
        x_test /= 255.
        test_index = np.arange(len(x_test))
        np.random.seed(11011)
        np.random.shuffle(test_index)
        x_test = x_test[test_index]
        x_test = x_test[:self.num_imgs]
        y_test = y_test[test_index]
        y_test = y_test[:self.num_imgs]
        return x_train, y_train, x_test, y_test

    def get_pretrained_model(self):
        """
        Function:
            Get a pre-trained model for classification on Mnist.
        """
        path = "../pre_trained_models/mnist_keras.h5"
        model = load_model(path)
        return model
    
    def get_state(self):
        """
        Function:
            Generate a random image.
        """
        index = np.random.randint(len(self.x_test), size=self.batch_size)
        image = self.x_test[index]
        label = self.y_test[index]
        return image, label
    
    def get_reward(self, state, label, action):
        (adversarial_sample, noise) = action
        label_onehot = to_categorical(label, self.num_classes)
        prediction   = self.model.predict(adversarial_sample)
        print('expected label', label, 'got', np.argmax(np.squeeze(prediction)))
        accuracy     = np.sum(label_onehot * prediction) / adversarial_sample.shape[0]
        
        score = self.__comput_score(accuracy, noise)
        return score
    
    def __comput_score(self, accuracy, noise):
        accuracy_loss = 1. - accuracy
        noise_mean    = np.mean(noise)
        print('acc', accuracy, 'acc_loss', accuracy_loss, 'mean', noise_mean)
        return self.alpha * accuracy - (1. - self.alpha) * noise_mean
    
class Memory:
    def __init__(self,
                 state,
                 mask_map,
                 noise_map,
                 threshold,
                 reward):
        self.state = state
        self.mask = mask_map
        self.noise_value = noise_map
        self.threshold = threshold
        self.reward = reward

class DQN:
    def __init__(self,
                 input_shape,
                 learning_rate=0.001,
                 discount_rate=0.95,
                 exp_rate=1.0,
                 min_exp_rate=0.01,
                 exp_decay=0.995):
        self.input_shape   = input_shape
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exp_rate      = exp_rate
        self.min_exp_rate  = min_exp_rate
        self.exp_decay     = exp_decay
        self.memory        = collections.deque(maxlen=2000)
        
        self.agent = self.__create_agent()
        self.agent.summary()
        
    def __create_agent(self):
        inp = Input(shape=self.input_shape)
        x   = Conv2D(kernel_size=3, filters=32, strides=2, padding='same', input_shape=self.input_shape)(inp)
        x   = BatchNormalization()(x)
        x   = ReLU()(x)
        x   = Conv2D(kernel_size=3, filters=64, padding='same')(x)
        x   = BatchNormalization()(x)
        x   = ReLU()(x)
        x   = Conv2D(kernel_size=3, filters=64, strides=2, padding='same')(x)
        x   = BatchNormalization()(x)
        x   = ReLU()(x)
        
        mask        = Conv2D(kernel_size=3, filters=1, padding='same', activation='sigmoid', name='mask')(x)
        noise_value = Conv2D(kernel_size=3, filters=16, padding='same', activation='sigmoid', name='noise_value')(x)
        
        x = Flatten()(x)
        threshold   = Dense(units=1, activation='sigmoid', name='threshold')(x)
        
        flattened_mask  = Flatten()(mask)
        flattened_noise = Flatten()(noise_value)
        output_noise    = Concatenate()([flattened_mask, flattened_noise, threshold])
        expected_reward = Dense(units=1)(output_noise)
        
        agent = Model(inp, expected_reward)
        
        agent.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return agent
    
    def act(self, env):
        state, label = env.get_state()
        
        expected_reward = self.agent.predict(state)
        mask            = get_activations(self.agent, state, 'mask')
        noise           = get_activations(self.agent, state, 'noise_value')
        threshold       = get_activations(self.agent, state, 'threshold')
        
        mask      = np.squeeze(mask['mask/Identity:0'])
        noise     = np.squeeze(noise['noise_value/Identity:0'])
        threshold = np.squeeze(threshold['threshold/Identity:0'])
        
        adversarial_sample, noise = self.__construct_adversarial_sample(state, mask, noise, threshold)
        reward = env.get_reward(state, label, (adversarial_sample, noise))
        
        print(np.amin(state), np.amax(state), np.mean(state))
        print(np.amin(adversarial_sample), np.amax(adversarial_sample), np.mean(adversarial_sample))
        print(expected_reward, reward)
        
        self.add_memory(state, (mask, noise, threshold), reward)
        
    def __construct_adversarial_sample(self, state, mask, noise, threshold):
        noise_map_shape = state.shape
        noise_map       = np.zeros(noise_map_shape)
        
        mask_w, mask_h = mask.shape[0], mask.shape[1]
        grid_w, grid_h = int(np.sqrt(noise.shape[2])), int(np.sqrt(noise.shape[2]))
        for i in range(mask_w):
            for j in range(mask_h):
                if mask[i,j] > threshold:
                    '''
                        i,j -> grid index in mask map
                    '''
                    cur_noise = noise[i, j]
                    assert cur_noise.shape == (16,)
                    
                    '''
                        Top left corner of current grid cell in the 7x7 mask map
                    '''
                    top_left_i, top_left_j = i * grid_w, j * grid_h
                    
                    for g_i in range(grid_w):
                        for g_j in range(grid_h):
                            cur_i, cur_j  = top_left_i + g_i, top_left_j + g_j
                            cur_noise_idx = g_i * grid_w + g_j
                            
                            noise_map[0, cur_i, cur_j] = noise[i, j, cur_noise_idx]
        
        adversarial_sample = state + noise_map            
        return adversarial_sample, noise_map
    
    def add_memory(self, state, action, reward):
        pass
        
print("Current TF version is {0}".format(tf.__version__))
environment = QLEnvironment()
dqn          = DQN((28, 28, 1))

dqn.act(environment)
# dqn.act()
# dqn.act()
# dqn.act()