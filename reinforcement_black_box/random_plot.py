import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

def debug(epsilon, alpha):
    """
    Function:
        comparison of different noise type
    """
    all_imgs = []
    folder_name1 = 'plots_gaussian/'
    folder_name2 = 'plots_uniform/'

    original = imageio.imread(folder_name1 + 'original.png')
    all_imgs.append(original)

    imgname = folder_name1 + 'eps_' + str(np.round(epsilon, 2)) \
        + '_alpha_' + str(np.round(alpha, 2)) + '.png'
    current_img = imageio.imread(imgname)
    all_imgs.append(current_img)

    imgname = folder_name2 + 'eps_' + str(np.round(epsilon, 2)) \
        + '_alpha_' + str(np.round(alpha, 2)) + '.png'
    current_img = imageio.imread(imgname)
    all_imgs.append(current_img)
    
    all_imgs = np.concatenate(all_imgs, axis = 0)

    fig = plt.figure(figsize = (20, 20))
    ax1 = fig.add_subplot(111)
    ax1.imshow(all_imgs)
    ax1.axis('off')
    plt.savefig("other_figures/noooise.png")
    plt.show()

if __name__ == "__main__":
    debug(1.0, 0.6)