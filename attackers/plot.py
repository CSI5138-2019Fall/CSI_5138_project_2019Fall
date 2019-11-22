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
    if dataset_name == "mnist":
        ax1.imshow(image, cmap="gray")
    else:
        ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(attacker_name, fontsize=45, fontweight='bold')
    plt.savefig(imgname)
    # plt.show()
    return all_images

if __name__ == "__main__":
    """
    attacker:
       "FGSM",
       "Elastic",
       "BasicIterativeMethod",
       "NewtonFool",
       "HopSkipJump",
       "ZooAttack",
       "VirtualAdversarialMethod",
       "UniversalPerturbation",
       "AdversarialPatch",
       "Attack",
       "BoundaryAttack",
       "CarliniL2",
       "CarliniLinf",
       "DeepFool",
       "SMM",
       "PGD",
    """
    attacker_names = ["FGSM",
       "Elastic",
       "BasicIterativeMethod",
       "NewtonFool",
       "HopSkipJump",
       "ZooAttack",
       "VirtualAdversarialMethod",
       "UniversalPerturbation",
       "BoundaryAttack",
       "CarliniL2",
       "CarliniLinf",
       "DeepFool",
       "SMM",
       "PGD",]
    dataset_names = ['mnist']#, 'cifar']
    imgs = []
    for attacker_name in attacker_names:
        for dataset_name in dataset_names:
            samples = ReadAndPlot(attacker_name, dataset_name)

