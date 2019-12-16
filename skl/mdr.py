#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coarsening
=====================================================

Copyright (C) 2017 Alan Valejo <alanvalejo@gmail.com> All rights reserved.

In coarsening strategy a sequence (or hierarchy) of smaller networks is
constructed from the original network, such that $|V_0| > |V_1| > ... > |V_N|$.
Such a hierarchy represents the network on multiple scales.

This file is part of Mdr.

Mdr is a free software and non-commercial use only: you can be use it for
creating unlimited applications, distribute in binary or object form only,
modify source-code and distribute modifications (derivative works). Please,
giving credit to the author by citing the papers. License will expire in 2018,
July, and will be renewed.

Owner or contributors are not liable for any direct, indirect, incidental,
special, exemplary, or consequential damages, (such as loss of data or profits,
and others) arising in any way out of the use of this software,
even if advised of the possibility of such damage.

Required:
	.. _igraph: http://igraph.sourceforge.net
	.. _scipy: http://www.scipy.org/
	.. _sklearn: http://scikit-learn.org/
	.. _numpy: http://www.numpy.org/
"""

import numpy
import sys

import models.args as args
import models.helper as helper
import models.helperigraph as helperigraph
from models.similarity import Similarity

import sharedmem
from multiprocessing import Process
from matplotlib import pylab
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin

__maintainer__ = 'Alan Valejo'
__author__ = 'Alan Valejo'
__email__ = 'alanvalejo@gmail.com'
__credits__ = ['Alan Valejo', 'Geraldo Pereira Rocha Filho', 'Maria Cristina Ferreira de Oliveira', 'Alneu de Andrade Lopes']
__homepage__ = 'https://github.com/alanvalejo/mob'
__license__ = 'GNU'
__docformat__ = 'markdown en'
__version__ = '0.1'
__date__ = '2018-10-05'

class MDR(BaseEstimator, ClassifierMixin):

	def __init__(self, **kwargs):

		prop_defaults = {
			'max_levels': 3
			, 'reduction_factor': 0.5
			, 'similarity': 'weighted_common_neighbors'
			, 'matching': 'gmb'
			, 'global_min_vertices': None
			, 'upper_bound': 0.2
			, 'tolerance': 0.01
			, 'itr': 10
			, 'logger': None
		}

		self.__dict__.update(prop_defaults)
		self.__dict__.update(kwargs)

		if self.logger is None:
			self.logger = helper.initialize_logger('log')

		# Validation of similarity measure
		valid_similarity = ['common_neighbors', 'weighted_common_neighbors',
		'salton', 'preferential_attachment', 'jaccard', 'adamic_adar',
		'resource_allocation', 'sorensen', 'hub_promoted', 'hub_depressed',
		'leicht_holme_newman', 'weighted_jaccard']
		if self.similarity.lower() not in valid_similarity:
			self.logger.warning('Similarity misure is unvalid.')
			sys.exit(1)

		# Validation of matching method
		valid_matching = ['mlpb', 'nmlpb', 'gmb', 'rgmb', 'hem', 'lem', 'rm']
		if self.matching.lower() not in valid_matching:
			self.logger.warning('Matching method is unvalid.')
			sys.exit(1)

	def fit(self, X):
		pass

	def transform2(self, X):

		self.g = helperigraph.load_matrix(X)
		self.g['level'] = 0

		while not self.g['level'] == self.max_levels:

			matching = range(self.g.vcount())
			levels = self.g['level']

			levels += 1
			self.g['similarity'] = getattr(Similarity(self.g, self.g['adjlist']), self.similarity)
			start = sum(self.g['vertices'][0:1])
			end = sum(self.g['vertices'][0:1 + 1])
			vertices = range(start, end)
			param = dict(reduction_factor=self.reduction_factor)
			if self.matching in ['gmb', 'rgmb']:
				param['vertices'] = vertices

			if self.matching in ['hem', 'lem', 'rm']:
				one_mode_graph = self.g.weighted_one_mode_projection(vertices)
				matching_method = getattr(one_mode_graph, self.matching)
			else:
				matching_method = getattr(self.g, self.matching)

			matching_method(matching, **param)

			coarse = self.g.contract(matching)
			coarse['level'] = levels
			self.g = coarse

		return helperigraph.biajcent_matrix(self.g)

	def transform(self, X):

		self.g = helperigraph.load_matrix(X)
		n =  self.g['vertices'][1]
		self.g['level'] = 0

		new_min = 0.1
		new_max = 10
		old_min = min(self.g.es['weight'])
		old_max = max(self.g.es['weight'])
		with open("../bnoc-src/output/cbrson.ncol", "w+") as f:
			for e in self.g.es():
				e['weight'] = helper.remap(e['weight'], old_min, old_max, new_min, new_max)
				f.write(str(e.tuple[0]) + ' ' + str(e.tuple[1]) + ' ' + str(e['weight']) + '\n')

		# print self.g.ecount()
		# # print self.g['vertices']
		# dd = self.g.degree_distribution()
		# print dd
		# print self.g['vertices']
		# print 'grau zero', len(self.g.vs.select(_degree = 0))
		# print 'grau um', len(self.g.vs.select(_degree = 1))
		# print 'grau dois', len(self.g.vs.select(_degree = 2))
		# print 'grau tres', len(self.g.vs.select(_degree = 3))
		# print 'grau quatro', len(self.g.vs.select(_degree = 4))
		# exit()
		# plt.plot(dd).show()
		# xs, ys = zip(*[(left, count) for left, _, count in self.g.degree_distribution().bins()])
		# pylab.bar(xs, ys)
		# pylab.show()

		running = True
		while running:
			running = False

			membership = range(self.g.vcount())
			levels = self.g['level']
			contract = False

			matching_layer = True
			if (self.global_min_vertices is None):
				if levels >= self.max_levels:
					matching_layer = False
			elif (int(self.g['vertices'][1]) <= int(self.global_min_vertices)):
				matching_layer = False

			if matching_layer:
				contract = True
				running = True
				levels += 1

				self.g['similarity'] = getattr(Similarity(self.g, self.g['adjlist']), self.similarity)
				start = sum(self.g['vertices'][0:1])
				end = sum(self.g['vertices'][0:1 + 1])
				vertices = range(start, end)

				param = dict(reduction_factor=self.reduction_factor)

				if self.matching in ['mlpb', 'nmlpb', 'nmb']:
					param['upper_bound'] = self.upper_bound
					param['n'] = n
					param['global_min_vertices'] = self.global_min_vertices
				if self.matching in ['mlpb', 'nmlpb', 'gmb', 'rgmb']:
					param['vertices'] = vertices
				if self.matching in ['mlpb']:
					param['tolerance'] = self.tolerance
					param['itr'] = self.itr

				if self.matching in ['hem', 'lem', 'rm']:
					one_mode_graph = self.g.weighted_one_mode_projection(vertices)
					matching_method = getattr(one_mode_graph, self.matching)
				else:
					matching_method = getattr(self.g, self.matching)

				matching_method(membership, **param)

			if contract:
				coarse = self.g.contract(membership)
				coarse['level'] = levels

				if coarse.vcount() == self.g.vcount():
					break

				self.g = coarse

		return helperigraph.biajcent_matrix(self.g)
