import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import time
from glob import glob
import cv2

def PlotAll(dataset_name):
    """
    Function:
        Plot all of them and arranged into one image.
    """
    name_imgs = glob("samples_imgs/*_" + dataset_name + ".png")
    imgs_row1 = []
    imgs_row2 = []
    count = 0
    for img_name in name_imgs:
        count += 1
        img = cv2.imread(img_name)
        blank = np.zeros(img.shape)
        img = img[..., ::-1]
        if count <=7:
            imgs_row1.append(img)
        else:
            imgs_row2.append(img)

    imgs_row1 = np.concatenate(imgs_row1, axis=1)
    imgs_row2 = np.concatenate(imgs_row2, axis=1)
    imgs_all = np.concatenate([imgs_row1, imgs_row2], axis=0)
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(111)
    if dataset_name == "mnist":
        ax1.imshow(imgs_all)
    else:
        ax1.imshow(imgs_all)
    ax1.axis('off')
    plt.savefig('presentation_imgs/' + dataset_name + '.png')
    plt.show()

if __name__ == "__main__":
    dataset_name = "mnist"
    PlotAll(dataset_name)
