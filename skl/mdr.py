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
			, 'matching': 'greedy_twohops'
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
		valid_matching = ['greedy_seed_twohops', 'greedy_twohops', 'hem',
		'lem', 'rm']
		if self.matching.lower() not in valid_matching:
			self.logger.warning('Matching method is unvalid.')
			sys.exit(1)

	def fit(self, X):
		pass

	def transform(self, X):

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
			if self.matching in ['hem', 'lem', 'rm']:
				one_mode_graph = self.g.weighted_one_mode_projection(vertices)
				matching_method = getattr(one_mode_graph, self.matching)
				matching_method(matching, reduction_factor=self.reduction_factor)
			else:
				matching_method = getattr(self.g, self.matching)
				matching_method(vertices, matching, reduction_factor=self.reduction_factor)

			coarse = self.g.contract(matching)
			coarse['level'] = levels
			self.g = coarse

		return helperigraph.biajcent_matrix(self.g)
