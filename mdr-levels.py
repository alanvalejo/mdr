#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coarsening
=====================================================

Copyright (C) 2017 Alan Valejo <alanvalejo@gmail.com> All rights reserved.

In coarsening strategy a sequence (or hierarchy) of smaller networks is
constructed from the original network, such that $|V_0| > |V_1| > ... > |V_N|$.
Such a hierarchy represents the network on multiple scales.

This file is part of MOB.

MOB is a free software and non-commercial use only: you can be use it for
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

import args
import sys
import logging
import json
import numpy
import helper
import helperigraph

from timing import Timing
from similarity import Similarity

__maintainer__ = 'Alan Valejo'
__author__ = 'Alan Valejo'
__email__ = 'alanvalejo@gmail.com'
__credits__ = ['Alan Valejo', 'Geraldo Pereira Rocha Filho', 'Maria Cristina Ferreira de Oliveira', 'Alneu de Andrade Lopes']
__homepage__ = 'https://github.com/alanvalejo/mob'
__license__ = 'GNU'
__docformat__ = 'markdown en'
__version__ = '0.1'
__date__ = '2018-10-05'

def main():
	"""
	Main entry point for the application when run from the command line.
	"""

	# Timing instanciation
	timing = Timing(['Snippet', 'Time [m]', 'Time [s]'])

	with timing.timeit_context_add('Pre-processing'):

		# Setup parse options command line
		parser = args.setup_parser('args/mdr-levels.json')
		options = parser.parse_args()
		args.update_json(options)
		args.check_output(options)

		# Log instanciation
		log = helper.initialize_logger(dir='log', output='log')

		if options.input and options.vertices is None:
			log.warning('Vertices are required when input is given.')
			sys.exit(1)

		# Create default values for optional parameters
		if options.reduction_factor is None:
			options.reduction_factor = 0.5
		if options.max_levels is None:
			options.max_levels = 3
		if options.matching is None:
			options.matching = 'greedy_seed_twohops'
		if options.similarity is None:
			options.similarity = 'weighted_common_neighbors'

		# Validation of matching method
		valid_matching = ['greedy_seed_twohops', 'greedy_twohops']
		if options.matching.lower() not in valid_matching:
			log.warning('Matching method is unvalid.')
			sys.exit(1)

		# Validation of input extension
		valid_input = ['.arff', '.dat']
		if options.extension not in valid_input:
			log.warning('Input is unvalid.')
			sys.exit(1)

		# Validation of similarity measure
		valid_similarity = ['common_neighbors', 'weighted_common_neighbors', 'salton', 'preferential_attachment', 'jaccard', 'adamic_adar', 'resource_allocation', 'sorensen', 'hub_promoted', 'hub_depressed', 'leicht_holme_newman']
		if options.similarity.lower() not in valid_similarity:
			log.warning('Similarity misure is unvalid.')
			sys.exit(1)

		options.vertices = map(int, options.vertices)
		options.max_levels = int(options.max_levels)
		options.reduction_factor = float(options.reduction_factor)

	# Load bipartite graph
	with timing.timeit_context_add('Load'):
		if options.extension == '.arff':
			graph = helperigraph.load_csr(options.input)
		elif options.extension == '.dat':
			graph = helperigraph.load_dat(options.input)
		graph['level'] = 0

	# Coarsening
	with timing.timeit_context_add('Coarsening'):
		hierarchy_graphs = []
		hierarchy_levels = []
		while not graph['level'] == options.max_levels:

			matching = range(graph.vcount())
			levels = graph['level']

			levels += 1
			graph['similarity'] = getattr(Similarity(graph, graph['adjlist']), options.similarity)
			start = sum(graph['vertices'][0:1])
			end = sum(graph['vertices'][0:1 + 1])
			matching_method = getattr(graph, options.matching)
			matching_method(range(start, end), matching, reduction_factor=options.reduction_factor)

			coarse = graph.contract(matching)
			coarse['level'] = levels
			graph = coarse
			if options.save_hierarchy or (graph['level'] == options.max_levels):
				hierarchy_graphs.append(graph)
				hierarchy_levels.append(levels)

	# Save
	with timing.timeit_context_add('Save'):

		output = options.output
		for index, obj in enumerate(reversed(zip(hierarchy_levels, hierarchy_graphs))):
			levels, graph = obj

			if options.save_conf:
				with open(output + '-' + str(index) + '.conf', 'w+') as f:
					d = {}
					d['source_filename'] = options.input
					d['source_v0'] = options.vertices[0]
					d['source_v1'] = options.vertices[1]
					d['source_vertices'] = options.vertices[0] + options.vertices[1]
					d['edges'] = graph.ecount()
					d['vertices'] = graph.vcount()
					d['reduction_factor'] = options.reduction_factor
					d['max_levels'] = options.max_levels
					d['similarity'] = options.similarity
					d['matching'] = options.matching
					d['levels'] = levels
					for layer in range(graph['layers']):
						vcount = str(len(graph.vs.select(type=layer)))
						attr = 'v' + str(layer)
						d[attr] = vcount
					json.dump(d, f, indent=4)

			if options.save_ncol:
				graph.write(output + '-' + str(index) + '.ncol', format='ncol')

			if options.save_source:
				with open(output + '-' + str(index) + '.source', 'w+') as f:
					for v in graph.vs():
						f.write(' '.join(map(str, v['source'])) + '\n')

			if options.save_predecessor:
				with open(output + '-' + str(index) + '.predecessor', 'w+') as f:
					for v in graph.vs():
						f.write(' '.join(map(str, v['predecessor'])) + '\n')

			if options.save_successor:
				numpy.savetxt(output + '-' + str(index) + '.successor', graph.vs['successor'], fmt='%d')

			if options.save_weight:
				numpy.savetxt(output + '-' + str(index) + '.weight', graph.vs['weight'], fmt='%d')

			if options.save_adjacency:
				numpy.savetxt(output + '-' + str(index) + '.dat', helper.biajcent_matrix(graph), fmt='%d')

			if options.save_gml:
				del graph['adjlist']
				del graph['similarity']
				graph['layers'] = str(graph['layers'])
				graph['vertices'] = ','.join(map(str, graph['vertices']))
				graph['level'] = str(graph['level'])
				graph.vs['name'] = map(str, range(0, graph.vcount()))
				graph.vs['type'] = map(str, graph.vs['type'])
				graph.vs['weight'] = map(str, graph.vs['weight'])
				graph.vs['successor'] = map(str, graph.vs['successor'])
				for v in graph.vs():
					v['source'] = ','.join(map(str, v['source']))
					v['predecessor'] = ','.join(map(str, v['predecessor']))
				graph.write(output + '-' + str(index) + '.gml', format='gml')

			if not options.save_hierarchy:
				break

	if options.show_timing:
		timing.print_tabular()
	if options.save_timing:
		timing.save_json(output + '.timing')


if __name__ == "__main__":
	sys.exit(main())
