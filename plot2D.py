import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import TruncatedSVD
from skl.mdr import MDR
import sys

if sys.argv[1] == 'iris':
	data = datasets.load_iris()
	target = ['class 1', 'class 2', 'class 3']
elif sys.argv[1] == 'wine':
	data = datasets.load_wine()
	target = ['class 1', 'class 2', 'class 3']
elif sys.argv[1] == 'breast-cancer':
	data = datasets.load_breast_cancer()
	target = ['class 1', 'class 2']

if sys.argv[2] == 'pca':
	X_reduced = PCA(n_components=2).fit_transform(data.data)
elif sys.argv[2] == 'lsa':
	X_reduced = TruncatedSVD(n_components=2, n_iter=7, random_state=42).fit_transform(data.data)
elif sys.argv[2] == 'fa':
	X_reduced = FeatureAgglomeration(n_clusters=2).fit_transform(data.data)
elif sys.argv[2] == 'mdr':
	X_reduced = MDR(max_levels=int(sys.argv[3]), reduction_factor=float(sys.argv[4]), matching=sys.argv[5]).transform(data.data)

x = X_reduced[:, 0]
y = X_reduced[:, 1]
labels = data.target
formatter = plt.FuncFormatter(lambda i, *args: target[int(i)])

size = (5, 4)
ylabel = 'y'
xlabel = 'x'
title = 'Iris 2D'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Times')
plt.figure(figsize=size)
sc = plt.scatter(x, y, c=labels, alpha=0.5)
plt.grid(True, linestyle=":", color='black', alpha=0.2, linewidth=0.5)
plt.xlabel(xlabel, fontsize=15, labelpad=15)
plt.ylabel(ylabel, fontsize=15, labelpad=15)
axes = plt.gca()
axes.spines['right'].set_visible(False)
axes.spines['left'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.spines['bottom'].set_visible(False)
axes.tick_params(axis='y', which='both', length=5, labelsize=15)
axes.tick_params(axis='x', which='both', length=5, labelsize=15)

# cbar = plt.colorbar(ticks=[0, 1, 2], format=formatter)
# cbar.ax.tick_params(labelsize=15)
# handles, labels = axes.get_legend_handles_labels()
# prop={'labelspacing':0.25}
# leg_param = dict(prop={'size': 16}, ncol=1, fancybox=True, shadow=False, frameon=False)
# axes.legend(target, loc='upper left', **leg_param)
# for line in leg.get_lines():
# 	line.set_linewidth(4.0)

classes = target
class_colours = []
for i in range(0, len(target)):
	class_colours.append(sc.to_rgba(i))
recs = []
for i in range(0, len(class_colours)):
	recs.append(mpatches.Circle((0, 0), fc=class_colours[i], alpha=0.6))
leg_param = dict(prop={'size': 16}, ncol=1, fancybox=True, shadow=False, frameon=False)
axes.legend(recs, classes, loc='best', **leg_param)

plt.savefig('output/' + sys.argv[1] + '-' + sys.argv[2] + '.png', dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
