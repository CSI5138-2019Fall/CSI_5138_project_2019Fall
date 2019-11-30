import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

mu = 0
sigma = 0.5
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
x_new = x[x >= 0]
x_new = x_new[x_new <= 1]
x_zeros = np.zeros(len(x_new))

fig = plt.figure(figsize = (10, 6))
ax = fig.add_subplot(111)

ax.plot(x, stats.norm.pdf(x, mu, sigma), 'black')
ax.axvline(x=0.0, ls='-.', color='black')
ax.axvline(x=1.0, ls='-.', color='black')
ax.grid(linestyle='--', linewidth=0.5, alpha=0.8)
ax.text(-1.4, .75, r'$\mu=0.,\ \sigma=0.5*epsilon$')
ax.fill_between(x_new, stats.norm.pdf(x_new, mu, sigma), alpha=0., hatch='/')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([0, 0.85])
ax.set_xlabel('deviation (* epsilon)')
ax.set_ylabel('probability')

plt.savefig("./noise.png")
plt.show()
