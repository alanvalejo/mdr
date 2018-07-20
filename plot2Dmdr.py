import numpy as np
import helper
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

if sys.argv[1] == 'iris':
	dv = np.loadtxt('output/iris-mdr-mob-rgmb-0.dat', delimiter=' ')
	labels_name = np.loadtxt('input/iris.labels', dtype='str', delimiter=' ')
	target = ['class 1', 'class 2', 'class 3']
elif sys.argv[1] == 'wine':
	dv = np.loadtxt('output/wine-mdr-mob-gmb-0.dat', delimiter=' ')
	labels_name = np.loadtxt('input/wine.labels', dtype='str', delimiter=' ')
	target = ['class 1', 'class 2', 'class 3']
elif sys.argv[1] == 'breast-cancer':
	dv = np.loadtxt('output/breast-cancer-mdr-mob-gmb-0.dat', delimiter=' ')
	labels_name = np.loadtxt('input/breast-cancer.labels', dtype='str', delimiter=' ')
	target = ['class 1', 'class 2']

labels = helper.encode_categorical(labels_name)
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
sc = plt.scatter(x, y, c=labels, alpha=0.5)
plt.grid(True, linestyle=":", color='black', alpha=0.2, linewidth=0.5)
plt.xlabel(xlabel, fontsize=15, labelpad=15)
plt.ylabel(ylabel, fontsize=15, labelpad=15)

classes = target
class_colours = []
for i in range(0, len(target)):
	class_colours.append(sc.to_rgba(i))
recs = []
for i in range(0, len(class_colours)):
	recs.append(mpatches.Circle((0, 0), fc=class_colours[i], alpha=0.6))
leg_param = dict(prop={'size': 16}, ncol=1, fancybox=True, shadow=False, frameon=False)
axes.legend(recs, classes, loc='upper left', **leg_param)

plt.savefig('output/' + sys.argv[1] + '-mdr.png', dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
