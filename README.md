## MOb (Multilevel framework to bipartite network)

**About**

This is an Python implementation of multilevel framework for handling bipartite networks, published by Alan et. al. [1]. The implementation is based on multilevel approach for combinatorial optimization that can perform in bipartite context. The aim to reduce the cost of an optimization process by applying it to a reduced or coarsened version of the original bipartite network.

**Download**

* You can download the MOb software in http://www.alanvalejo.com.br/software?name=MOb

**Usage**

> Coarsening by levels. The user defines the number of levels and the reduction factor at each level.

	$ python coarsening-level.py [options]

> Coarsening until a max number of vertices. The process will automatically control the total number of levels and the reduction factor at each level.

	$ python coarsening-max-vertices.py [options]

|Option            |Domain           |Default   |Description                          |
|------------------|-----------------|----------|-------------------------------------|
|-f --filename     |string [FILE]    |None      |name of the FILES to be loaded       |
|-v --vertices     |int              |None      |number of vertices for each layer    |
|-d --directory    |string [DIR]     |'.'       |directory of output FILEs            |
|-o --output       |string [FILE]    |filename  |name of the FILE to be save          |
|-r --rf           |array in (0 0.5] |[0.5 0.5] |reduction factor for each layer      |
|-m --ml           |array in [0 n]   |[3 3]     |max levels for each layer            |
|-mv --max_vertices|array in [0 n]   |None      |max number of vertices for each layer; supress -r and -m parameters|
|-c --matching     |string           |None      |matching method for each layer       |
|-s --similarity   |string           |None      |similarity measure for each layer    |
|-cf --conf        |string [FILE]    |None      |name of the config FILE to be loaded |
|--save_conf       |boolean          |false     |save config file                     |
|--save_ncol       |boolean          |false     |save ncol format                     |
|--save_gml        |boolean          |false     |save gml format                      |
|--save_source     |boolean          |false     |save source reference                |
|--save_predecessor|boolean          |false     |save predecessor reference           |
|--save_hierarchy  |boolean          |false     |save all levels of hierarchy         |
|--save_timing     |boolean          |false     |save timing in json                  |
|--show_timing     |boolean          |false     |show timing                          |
|--unique_key      |boolean          |false     |output date and time as unique_key   |

The matching strategy selects the best pairs of vertices for matching. Formally, a matching $M$ can be denoted by a set of pairwise non-adjacent edges, i.e., a set of edges with no common vertices. In this software it is possible use two matching methods:

> * Greed Rand Twohopes
> * Greed Twohopes

The matching strategy is, therefore, a key component of an effective multilevel optimization, as it leads to a good hierarchy of coarsened networks for supporting the local search algorithm. A poor choice of the matching impairs the multilevel process, hence, the performance of the local search algorithm. In this software it is possible use some similarity measures:

> * Common Neighbors
> * Weighted Common Neighbors
> * Preferential Attachment
> * Jaccard
> * Salton
> * Adamic Adar
> * Resource Allocation
> * Sorensen
> * Hub Promoted
> * Hub Depressed
> * Leicht Holme Newman

**Quick benchmark results**

We test a scientific collaboration network (Cond-Mat), available [here](https://toreopsahl.com/datasets/#newman2001), which is based on preprints posted in the Condensed Matter section (arXiv) between 1995 and 1999 and has 38.742 vertices (authors and papers) and 58.595 edges (authorship) among different types of vertices.

	$ python coarsening-level.py -f input/condmat1995to1999.ncol -v 16726 22016 -m 1 1 -r 0.5 0.5 --show_timing

|Snippet   |Time [m]|Time [s]|
|----------|--------|--------|
|Load      |0.0     |0.4741  |
|Coarsening|0.0     |0.7274  |
|Save      |0.0     |0.1087  |

	$ python coarsening-level.py -f input/condmat1995to1999.ncol -v 16726 22016 -m 2 2 -r 0.5 0.5 --show_timing

|Snippet   |Time [m]|Time [s]|
|----------|--------|--------|
|Load      |0.0     |0.4807  |
|Coarsening|0.0     |1.1317  |
|Save      |0.0     |0.0340  |


	$ python coarsening-level.py -f input/condmat1995to1999.ncol -v 16726 22016 -m 3 3 -r 0.5 0.5 --show_timing

|Snippet   |Time [m]|Time [s]|
|----------|--------|--------|
|Load      |0.0     |0.4830  |
|Coarsening|0.0     |1.3437  |
|Save      |0.0     |0.0467  |


You can use a config file (.json) to specify the parameters.

    $ python coarsening-level.py -cf input/condmat9599R16726C22016.json

**Dependencies**

* Python: tested with version 2.7.13.
* Packages required: [igraph](http://igraph.sourceforge.net); [scipy](http://www.scipy.org/); [sklearn](http://scikit-learn.org/); [numpy](http://www.numpy.org/)

**Known Bugs**

Please contact the author for problems and bug report.

**Contact**

* Alan Valejo.
* Ph.D. candidate at University of São Paolo (USP), Brazil.
* alanvalejo@gmail.com.

**License (COPYING.md)**

* Can be used for creating unlimited applications
* Can be distributed in binary or object form only
* Non-commercial use only
* Can modify source-code and distribute modifications (derivative works)
* Giving credit to the author by citing the papers [1,2]
* License will expire in 2018, July, and will be renewed.

**References**
> [1] Valejo, A.; Rocha Filho, G. P.; Oliveira, M. C. F.; and Lopes, A. A.: A Multilevel approach for combinatorial optimization in bipartite networks. Knowledge-Based Systems (2018)

~~~~~{.bib}
@article{Alan_2014,
    author={Valejo, A.; Rocha Filho, G. P.; Oliveira, M. C. F.; and Lopes, A. A.},
    title={Multilevel approach for combinatorial optimization in bipartite networks},
    journal={Knowledge-Based Systems},
    year={2018},
}
~~~~~

<div class="footer"> &copy; Copyright (C) 2017 Alan Valejo &lt;alanvalejo@gmail.com&gt; All rights reserved.</div>
