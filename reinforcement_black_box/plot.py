import numpy as np
# import matplotlib
# # for remote plot through X11
# matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import csv
from glob import glob

from environment import Environment
# from table import BlackBoxAgent
from agent_improv2 import BlackBoxAgent


def ReadCsv(filename):
    """
    Function:
        Read the csv file saved from tensorboard.
    """
    x = []
    y = []
    # open csv file.
    with open(filename) as f:
        csv_reader = list(csv.reader(f, delimiter=','))
        for i in range(len(csv_reader)):
            if i == 0:
                continue
            else:
                current_line = csv_reader[i]
                x.append(int((current_line[1])))
                y.append(float((current_line[2])))
    # make it into numpy array
    x = np.array(x)
    y = np.array(y)
    return x, y

def SparseData(x_in, y_in, window = 25, order = 1):
    """
    Function:
        Smooth the plot.
    """
    x = x_in
    y = savgol_filter(y_in, window, order)
    return x, y

def PlotWithSparse(x, y, ax, color, line_label):
    """
    Function:
        Plot and save the figure.
    """
    # two different styles for plotting
    style1 = color + '-.'
    style2 = color + '-'
    ax.plot(x, y, style1, alpha = 0.3)
    x_prime, y_prime = SparseData(x, y)
    line, = ax.plot(x_prime, y_prime, style2)
    # set labels for legend
    line.set_label(line_label)

def GetOnePlot(epsilon, alpha, threshold, noise_type):
    """
    Function:
        Get the tensorboard file with the specific requirements.
    """
    # get the file name
    file_pref = "tensorboard_csv_" + noise_type + "/run-nmax_" + str(epsilon) + "_alpha_" + \
                str(alpha) + "_threshold_" + str(threshold) + "-tag-"

    file_descrip = "average_confidence_loss.csv"

    file_name = file_pref + file_descrip
    x, y = ReadCsv(file_name)
    return x, y

def PlotAll(mode, noise_type, xlim, ylim):
    """
    Function:
        Plot all data for comparison.
    """
    # hyper-parameters settings
    batch_size = 1
    image_shape = (batch_size, 28, 28, 1)
    # noise_epsilon = 0.8 # max value of the images is 1.0
    exploration_decay = 0.8
    exploration_decay_steps = 800
    # alpha = 0.5
    epsilons = [1.0, 0.8, 0.6]
    alphas = [1.0, 0.8, 0.6, 0.4]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ['r', 'b', 'm', 'g', 'k', 'y']

    i = 0
    if mode == "eps":
        for epsilon in epsilons:
            alpha = 1.0
            agent = BlackBoxAgent(image_shape, epsilon, alpha, exploration_decay)
            threshold = agent.reward_threshold
            x, y = GetOnePlot(epsilon, alpha, threshold, noise_type)
            PlotWithSparse(x, y, ax, colors[i], "epsilon_" + str(np.round(epsilon,2)) + \
                                                "_alpha_" + str(np.round(alpha,2)))
            i += 1
    elif mode == "alpha":
        for alpha in alphas:
            epsilon = 1.0
            agent = BlackBoxAgent(image_shape, epsilon, alpha, exploration_decay)
            threshold = agent.reward_threshold
            x, y = GetOnePlot(epsilon, alpha, threshold, noise_type)
            PlotWithSparse(x, y, ax, colors[i], "epsilon_" + str(np.round(epsilon,2)) + \
                                                "_alpha_" + str(np.round(alpha,2)))
            i += 1
    else:
        raise ValueError("------ wrong mode input, please doublecheck it. --------")
    
    if mode == "eps":
        title_content = "influence of epsilon"
    elif mode == "alpha":
        title_content = "influence of alpha"
    
    ax.set_title(title_content)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("training steps")
    ax.set_ylabel("accuracy loss")
    ax.legend()
    ax.grid()
    plt.savefig("plots_" + noise_type + "/compare_" + mode + ".png")
    plt.show()

if __name__ == "__main__":
    """
    mode: either "eps" or "alpha"
    """

    mode_name = "alpha"
    noise_type = "gaussian"
    PlotAll(mode_name, noise_type, [0, 20000], [0, 1])