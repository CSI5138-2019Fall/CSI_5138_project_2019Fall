import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

mu = 0
sigma = 0.5
x = np.linspace(-1, 1, 100)
x_uni = np.ones(len(x))
x_new = x[x >= 0]
x_new = x_new[x_new <= 1]
x_ones = np.ones(len(x_new))

fig = plt.figure(figsize = (10, 6))
ax = fig.add_subplot(111)

# ax.plot(x, stats.norm.pdf(x, mu, sigma), 'black')
# ax.axhline(y=1.0, xmin=-1, xmax=1, ls='-', color='black')
ax.plot(x, x_uni, 'black')
ax.axvline(x=-1.0, ymin=0.0, ymax=0.8333, ls='-', color='black')
ax.axvline(x=1.0, ymin=0.0, ymax=0.8333, ls='-', color='black')
ax.axvline(x=0.0, ls='-.', color='black')
ax.axvline(x=1.0, ls='-.', color='black')
ax.grid(linestyle='--', linewidth=0.5, alpha=0.8)
# ax.text(-1.4, .75, r'$\mu=0.,\ \sigma=0.5*epsilon$')
ax.fill_between(x_new, x_ones, alpha=0., hatch='/')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([0, 1.2])
ax.set_xlabel('value (* epsilon)')
ax.set_ylabel('probability')

plt.savefig("./uniform_noise.png")
plt.show()
