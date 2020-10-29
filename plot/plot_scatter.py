import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

exp = 'coverage'
# exp = 'explore'

# colors = ['g', 'r', 'b', 'c', 'm']
# colors = ['tab:green', 'tab:red', 'tab:blue', 'tab:orange']
colors = ['tab:green', 'tab:red', 'tab:blue', 'tab:pink']
# linestyles = ['-.', '--', '-', '-', '-' ]
linestyles = ['-.', '--', '-', '-' ]

fnames = ['data/expert_', 'data/greedy_', 'data/id_', 'data/rl_id_',]
# fnames = ['data/expert_', 'data/greedy_', 'data/id_', 'data/nl_']
fnames = [fname + exp + '.csv' for fname in fnames]

labels = ['Expert Controller', 'Greedy Controller', 'Agg. GNN', 'Agg. GNN using RL']
# labels = ['Expert Controller', 'Greedy Controller', 'Agg. GNN', 'Non-Linear Agg. GNN']

#
# fnames = ['data/id_', 'data/nl_', 'data/expert_', 'data/greedy_']
# fnames = [fname + exp + '.csv' for fname in fnames]
#
# labels = ['Agg. GNN', 'Non-Linear Agg. GNN', 'Expert Controller', 'Greedy Controller']


fig = plt.figure(figsize=(6, 4))

for fname, label, color, ls in zip(fnames, labels, colors, linestyles):
    data = np.loadtxt(fname, skiprows=1)
    plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2]/10, label=label, color=color, ls=ls)

plt.ylabel('Avg. Reward')
plt.xlabel('GNN Receptive Field / Controller Horizon')
plt.legend(loc='lower right')
# plt.savefig(exp + 'rl.eps', format='eps')
plt.savefig(exp + '_rl2.eps', format='eps')
plt.show()



