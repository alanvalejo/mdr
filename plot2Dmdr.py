import numpy as np
import helper
import matplotlib.pyplot as plt

dv = np.loadtxt('output/iris-mdr-mob-gmb-0.dat', delimiter=' ')
labels_name = np.loadtxt('input/iris.labels', dtype='str', delimiter=' ')
labels = helper.encode_categorical(labels_name)
# this formatter will label the colorbar with the correct target names
target = ['setosa', 'versicolor', 'virginica']
formatter = plt.FuncFormatter(lambda i, *args: target[int(i)])

x = dv[:, 1]
y = dv[:, 0]

size = (5, 4)
ylabel = 'y'
xlabel = 'x'
title = 'Iris 2D'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Times')
plt.figure(figsize=size)
axes = plt.gca()
axes.spines['right'].set_visible(False)
axes.spines['left'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.spines['bottom'].set_visible(False)
axes.tick_params(axis='y', which='both', length=5, labelsize=15)
axes.tick_params(axis='x', which='both', length=5, labelsize=15)
plt.scatter(x, y, c=labels)
plt.grid(True, linestyle=":", color='black', alpha=0.2, linewidth=0.5)
plt.xlabel(xlabel, fontsize=15, labelpad=15)
plt.ylabel(ylabel, fontsize=15, labelpad=15)
cbar = plt.colorbar(ticks=[0, 1, 2], format=formatter)
cbar.ax.tick_params(labelsize=15)
plt.savefig('output/iris-mdr-mob-gmb.png', dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
