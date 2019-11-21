import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import time


def ReadAndPlot(attacker_name, dataset_name):
    """
    Function:
        Read the file saved as np array.
    """
    filename = 'samples/' + attacker_name + '_adv_' + dataset_name + '.npy'
    imgname = 'samples_imgs/' + attacker_name + '_adv_' + dataset_name + '.png'
    all_images = np.load(filename)
    image = np.squeeze(all_images[0])
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    ax1.imshow(image)
    ax1.axis('off')
    plt.savefig(imgname)
    # plt.show()
    return all_images

if __name__ == "__main__":
    """
    attacker_name:
        All according to the specific names
    dataset_name"
        'mnist'
        'cifar'
    """
    attacker_names = ['FGSM', 'Elastic']
    dataset_names = ['mnist', 'cifar']
    for attacker_name in attacker_names:
        for dataset_name in dataset_names:
            samples = ReadAndPlot(attacker_name, dataset_name)
        

