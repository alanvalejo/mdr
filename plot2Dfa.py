import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import FeatureAgglomeration

# import some data to play with
iris = datasets.load_iris()
X_reduced = FeatureAgglomeration(n_clusters=2).fit_transform(iris.data)
x = X_reduced[:, 0]
y = X_reduced[:, 1]
labels = iris.target
target = ['setosa', 'versicolor', 'virginica']
formatter = plt.FuncFormatter(lambda i, *args: target[int(i)])

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
plt.savefig('output/iris-fa.png', dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
