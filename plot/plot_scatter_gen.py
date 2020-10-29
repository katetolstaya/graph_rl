import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# exp = 'coverage'
exp = 'explore'

fnames = ['data/nl_', 'data/greedy_']
fnames = [fname + exp + '_generalize.csv' for fname in fnames]
colors = ['tab:orange', 'tab:red']
linestyles = ['-', '--']

labels = ['Non-Linear Agg. GNN', 'Greedy Controller']
fig = plt.figure(figsize=(6, 4))

for fname, label, color, ls in zip(fnames, labels, colors, linestyles):
    data = np.loadtxt(fname, skiprows=1)
    plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2]/10, label=label, color=color, ls=ls)

plt.ylabel('Avg. Reward')
plt.xlabel('Number of Agents')
plt.legend(loc='lower right')
plt.savefig(exp + '_generalize.eps', format='eps')
plt.show()
