import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from skl.mdr import MDR
from sklearn.preprocessing import StandardScaler
import sys
import numpy as np
import models.helperigraph as helperigraph
import pandas as pd
from sklearn.datasets import make_blobs

if sys.argv[1] == 'iris':
	data = datasets.load_iris()
	target = ['class 1', 'class 2', 'class 3']
	X = data.data
	labels = data.target
elif sys.argv[1] == 'wine':
	data = datasets.load_wine()
	target = ['class 1', 'class 2', 'class 3']
	X = data.data
	labels = data.target
elif sys.argv[1] == 'breast-cancer':
	data = datasets.load_breast_cancer()
	target = ['class 1', 'class 2']
	X = data.data
	labels = data.target
elif sys.argv[1] == 'artificial':
	X, labels = make_blobs(n_features=100, n_samples=1000, centers=4, random_state=1, cluster_std=1)
elif sys.argv[1] == 'digits':
	data = datasets.load_digits()
	target = range(0, 8)
	X = data.data
	labels = data.target
elif sys.argv[1] == 'diabetes':
	data = datasets.load_diabetes()
	target = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9', 'class 10']
	data.data = StandardScaler().fit_transform(data.data)
	X = data.data
	labels = data.target
elif sys.argv[1] == 'cbrson':
	graph = helperigraph.load('input/cbrson.ncol', [675, 2408])
	X = helperigraph.biajcent_matrix(graph)
	# X = StandardScaler().fit_transform(X)
	labels = np.loadtxt('input/cbrson.labels', delimiter='\n')
elif sys.argv[1] == 'network-kk-5000':
	graph = helperigraph.load('input/network-kk-5000.ncol', [4000, 1000])
	X = helperigraph.biajcent_matrix(graph)
	X = StandardScaler().fit_transform(X)
	labels = np.loadtxt('input/network-kk-5000.labels', delimiter='\n')
elif sys.argv[1] == 'network-kk-2000':
	graph = helperigraph.load('input/network-kk-2000.ncol', [1000, 1000])
	X = helperigraph.biajcent_matrix(graph)
	# X = StandardScaler().fit_transform(X)
	labels = np.loadtxt('input/network-kk-2000.labels', delimiter='\n')
elif sys.argv[1] == 'lastfm':
	graph = helperigraph.load('input/lastfmSampled.ncol', [22461, 18144])
	X = helperigraph.biajcent_matrix(graph)
	X = StandardScaler().fit_transform(X)
	labels = np.loadtxt('input/lastfmSampled.labels', delimiter='\n')
elif  sys.argv[1] == 'sonar':
	df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data')
	labels = np.array(df.iloc[:,-1])
	names, labels = np.unique(labels, return_inverse=True)
	df = df.iloc[:, :-1]
	X = df.values
	X = StandardScaler().fit_transform(X)
elif  sys.argv[1] == 'tae':
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data')
	labels = np.array(df.iloc[:,-1])
	names, labels = np.unique(labels, return_inverse=True)
	df = df.iloc[:, :-1]
	X = df.values
elif  sys.argv[1] == 'yeast':
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data', delimiter='\s+', header=None)
	labels = np.array(df.iloc[:,-1])
	names, labels = np.unique(labels, return_inverse=True)
	df = df.iloc[:, :-1]
	df = df.iloc[:, 1:]
	df = df.drop(df.columns[[4,5]], axis=1)
	X = df.values
elif  sys.argv[1] == 'german':
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric', delimiter='\s+', header=None)
	labels = np.array(df.iloc[:,-1])
	names, labels = np.unique(labels, return_inverse=True)
	df = df.iloc[:, :-1]
	X = df.values
	X = StandardScaler().fit_transform(X)
elif  sys.argv[1] == 'zoo':
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data', delimiter=',', header=None)
	labels = np.array(df.iloc[:,-1])
	names, labels = np.unique(labels, return_inverse=True)
	df = df.iloc[:, :-1]
	df = df.iloc[:, 1:]
	X = df.values
	# X = StandardScaler().fit_transform(X)
elif  sys.argv[1] == 'vehicle':
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaa.dat', delimiter='\s+', header=None)
	labels = np.array(df.iloc[:,-1])
	names, labels = np.unique(labels, return_inverse=True)
	df = df.iloc[:, :-1]
	df = df.iloc[:, 1:]
	X = df.values
	X = StandardScaler().fit_transform(X)

print X.shape
print labels

if sys.argv[2] == 'pca':
	print 'oi'
	X_reduced = PCA(n_components=2).fit_transform(X)
elif sys.argv[2] == 'lsa':
	X_reduced = TruncatedSVD(n_components=2, n_iter=7, random_state=42).fit_transform(X)
elif sys.argv[2] == 'fa':
	X_reduced = FeatureAgglomeration(n_clusters=2).fit_transform(X)
elif sys.argv[2] == 'tsne':
	X_reduced = TSNE(n_components=2).fit_transform(X)
elif sys.argv[2] == 'mdr':
	if sys.argv[5] in ['gmb', 'rgmb', 'hem', 'lem', 'rm']:
		X_reduced = MDR(max_levels=int(sys.argv[3]), reduction_factor=float(sys.argv[4]), matching=sys.argv[5]).transform(X)
	elif sys.argv[5] in ['mlpb', 'nmlpb']:
		X_reduced = MDR(max_levels=int(sys.argv[3]), reduction_factor=float(sys.argv[4]), matching=sys.argv[5], global_min_vertices=int(sys.argv[6]), upper_bound=float(sys.argv[7]), itr=int(sys.argv[8]), tolerance=float(sys.argv[9])).transform(X)

print X_reduced.shape
x = X_reduced[:, 0]
y = X_reduced[:, 1]
formatter = plt.FuncFormatter(lambda i, *args: target[int(i)])

size = (20, 20)
ylabel = 'y'
xlabel = 'x'
title = 'Iris 2D'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Times')
plt.figure()
fig = plt.gcf()
DPI = fig.get_dpi()
fig.set_size_inches(400.0/float(DPI), 400.0/float(DPI))
sc = plt.scatter(x, y, c=labels, alpha=0.5, s=35)
# plt.grid(True, linestyle=":", color='black', alpha=0.5, linewidth=0.5)
plt.xlabel(xlabel, fontsize=15, labelpad=15)
plt.ylabel(ylabel, fontsize=15, labelpad=15)
axes = plt.gca()
axes.spines['right'].set_visible(False)
axes.spines['left'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.spines['bottom'].set_visible(False)
axes.tick_params(axis='y', which='both', length=5, labelsize=15)
axes.tick_params(axis='x', which='both', length=5, labelsize=15)
axes.get_xaxis().set_visible(False)
axes.get_yaxis().set_visible(False)
# axes.xaxis.label.set_visible(False)
# axes.yaxis.label.set_visible(False)
# axes.set_yticklabels([])
# axes.set_xticklabels([])
plt.axis('off')

# cbar = plt.colorbar(ticks=[0, 1, 2], format=formatter)
# cbar.ax.tick_params(labelsize=15)
# handles, labels = axes.get_legend_handles_labels()
# prop={'labelspacing':0.25}
# leg_param = dict(prop={'size': 16}, ncol=1, fancybox=True, shadow=False, frameon=False)
# axes.legend(target, loc='upper left', **leg_param)
# for line in leg.get_lines():
# 	line.set_linewidth(4.0)

# classes = target
# class_colours = []
# for i in range(0, len(target)):
# 	class_colours.append(sc.to_rgba(i + 6))

# recs = []
# for i in range(0, len(class_colours)):
# 	recs.append(mpatches.Circle((0, 0), fc=class_colours[i], alpha=0.0))
# leg_param = dict(prop={'size': 16}, ncol=1, fancybox=True, shadow=False, frameon=False)
# axes.legend(recs, classes, loc='best', **leg_param)

if sys.argv[2] == 'mdr':
	output = 'output/' + sys.argv[1] + '-' + sys.argv[2] + '-' + sys.argv[5] + '.pdf'
else:
	output = 'output/' + sys.argv[1] + '-' + sys.argv[2] + '.pdf'
plt.savefig(output, dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
