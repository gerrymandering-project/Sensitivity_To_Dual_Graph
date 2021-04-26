import facefinder
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, Lock
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
import networkx as nx
import numpy as np
import copy
import random
import math
import json
import sys
import os
import traceback
import types
from gerrychain import Graph
from gerrychain import MarkovChain
from gerrychain import accept
from gerrychain.constraints import (Validator, single_flip_contiguous,
                                    within_percent_of_ideal_population, UpperBound)
# 
from gerrychain.updaters import Election, Tally, cut_edges
# 
from gerrychain.partition import Partition
from gerrychain.proposals import recom


def face_sierpinski_mesh(partition, special_faces):
    """'Sierpinskifies' certain faces of the graph by adding nodes and edges to
    certain faces.

    Args:
        partition (Gerrychain Partition): partition object which contain assignment
        and whose graph will have edges and nodes added to
        special_faces (list): list of faces that we want to add node/edges to

    Raises:
        RuntimeError if SIERPINSKI_POP_STYLE of config file is neither 'uniform'
        nor 'zero'
    """

    graph = partition.graph
    # Get maximum node label.
    label = max(list(graph.nodes()))

    # Assign each node to its district in partition
    for node in graph.nodes():
        graph.nodes[node][config['ASSIGN_COL']] = partition.assignment[node]

    for face in special_faces:
        neighbors = [] #  Neighbors of face
        locationCount = np.array([0,0]).astype("float64")
        # For each face, add to neighbor_list and add to location count
        for vertex in face:
            neighbors.append(vertex)
            locationCount += np.array(graph.nodes[vertex]["pos"]).astype("float64")
        # Save the average of each of the face's positions
        facePosition = locationCount / len(face)

        # In order, store relative position of each vertex to the position of the face
        locations = [graph.nodes[vertex]['pos'] - facePosition for vertex in face]
        # Sort neighbors according to each node's angle with the center of the face
        angles = [float(np.arctan2(x[0], x[1])) for x in locations]
        neighbors.sort(key=dict(zip(neighbors, angles)).get)

        newNodes = []
        newEdges = []
        # For each consecutive pair of nodes, remove their edge, create a new
        # node at their average position, and connect edge node to the new node:
        for vertex, next_vertex in zip(neighbors, neighbors[1:] + [neighbors[0]]):
            label += 1
            # Add new node to graph with corresponding label at the average position
            # of vertex and next_vertex, and with 0 population and 0 votes
            graph.add_node(label)
            avgPos = (np.array(graph.nodes[vertex]['pos']) +
                      np.array(graph.nodes[next_vertex]['pos'])) / 2
            graph.nodes[label]['pos'] = avgPos
            graph.nodes[label][config['X_POSITION']] = avgPos[0]
            graph.nodes[label][config['Y_POSITION']] = avgPos[1]
            graph.nodes[label][config['POP_COL']] = 0
            graph.nodes[label][config['PARTY_A_COL']] = 0
            graph.nodes[label][config['PARTY_B_COL']] = 0

            # For each new node, 'move' a third of the population, Party A votes,
            # and Party B votes from its two adjacent nodes which previously exists
            # to itself (so that each previously existing node equally shares
            # its statistics with the two new nodes adjacent to it)
            # Note: should only be used when all special faces are NOT on edges
            if config['SIERPINSKI_POP_STYLE'] == 'uniform':
                for vert in [vertex, next_vertex]:
                    for keyword, orig_keyword in zip(['POP_COL', 'PARTY_A_COL', 'PARTY_B_COL'],
                                                     ['orig_pop', 'orig_A', 'orig_B']):
                        # Save original values if not done already
                        if orig_keyword not in graph.nodes[vert]:
                            graph.nodes[vert][orig_keyword] = graph.nodes[vert][config[keyword]]

                        # Increment values of new node and decrement values of old nodes
                        # by the appropriate amount.
                        graph.nodes[label][config[keyword]] += graph.nodes[vert][orig_keyword] // 3
                        graph.nodes[vert][config[keyword]] -= graph.nodes[vert][orig_keyword] // 3

                # Assign new node to same district as neighbor. Note that intended
                # behavior is that special_faces do not correspond to cut edges,
                # and therefore both vertex and next_vertex will be of the same
                # district.
                graph.nodes[label][config['ASSIGN_COL']] = graph.nodes[vertex][config['ASSIGN_COL']]

            # Choose a random adjacent node, assign the new node to the same partition,
            # and move half of its votes and population to said node
            elif config['SIERPINSKI_POP_STYLE'] == 'random':
                chosenNode = random.choice([graph.nodes[vertex], graph.nodes[next_vertex]])
                graph.nodes[label][config['ASSIGN_COL']] = chosenNode[config['ASSIGN_COL']]
                for keyword in ['POP_COL', 'PARTY_A_COL', 'PARTY_B_COL']:
                    graph.nodes[label][config[keyword]] += chosenNode[config[keyword]] // 2
                    chosenNode[config[keyword]] -= chosenNode[config[keyword]] // 2
            # Set the population and votes of the new nodes to zero. Do not change
            # previously existing nodes. Assign to random neighbor.
            elif config['SIERPINSKI_POP_STYLE'] == 'zero':
                graph.nodes[label][config['ASSIGN_COL']] =\
                random.choice([graph.nodes[vertex][config['ASSIGN_COL']],
                               graph.nodes[next_vertex][config['ASSIGN_COL']]]
                             )
            else:
                raise RuntimeError('SIERPINSKI_POP_STYLE must be "uniform", "random", or "zero"')

            # Remove edge between consecutive nodes if it exists
            if graph.has_edge(vertex, next_vertex):
                graph.remove_edge(vertex, next_vertex)

            # Add edge between both of the original nodes and the new node
            graph.add_edge(vertex, label)
            newEdges.append((vertex, label))
            graph.add_edge(label, next_vertex)
            newEdges.append((label, next_vertex))
            # Add node to connections
            newNodes.append(label)

        # Add an edge between each consecutive new node
        for vertex in range(len(newNodes)):
            graph.add_edge(newNodes[vertex], newNodes[(vertex+1) % len(newNodes)])
            newEdges.append((newNodes[vertex], newNodes[(vertex+1) % len(newNodes)]))
        # For each new edge of the face, set sibilings to be the tuple of all
        # new edges
        siblings = tuple(newEdges)
        for edge in newEdges:
            graph.edges[edge]['siblings'] = siblings

def createGridGraph(config):
    gridSize = config['GRID_SIZE']
    numDistricts = config['NUM_DISTRICTS']
    percentPartyA = config['PERCENT_PARTY_A']

    graph=nx.grid_graph([gridSize,gridSize])

    for i, n in enumerate(graph.nodes()):
        graph.nodes[n][config["POP_COL"]]=1
        graph.nodes[n][config["ASSIGN_COL"]] = int((numDistricts * i)/(gridSize**2))
        graph.nodes[n][config['X_POSITION']] = i % gridSize
        graph.nodes[n][config['Y_POSITION']] = int(i / gridSize)

        if i < ((gridSize) ** 2) * percentPartyA:
            graph.nodes[n][config["PARTY_A_COL"]]=1
            graph.nodes[n][config["PARTY_B_COL"]]=0
        else:
            graph.nodes[n][config["PARTY_A_COL"]]=0
            graph.nodes[n][config["PARTY_B_COL"]]=1
        if 0 in n or gridSize-1 in n:
            graph.nodes[n]["boundary_node"]=True
            graph.nodes[n]["boundary_perim"]=1
        else:
            graph.nodes[n]["boundary_node"]=False

    labels = {x: (gridSize*x[1] + x[0]) for x in graph.nodes()}
    graph = nx.relabel_nodes(graph, labels)
    return graph

def preprocessing(config):
    """Takes file path to JSON graph, and returns the appropriate

    Args:
        path_to_json ([String]): path to graph in JSON format

    Returns:
        graph (Gerrychain Graph): graph in JSON file following cleaning
        dual (Gerrychain Graph): planar dual of graph
    """
    if config['INPUT_GRAPH_FILENAME'] == 'GRID_GRAPH':
        graph = createGridGraph(config)
    else:
        graph = Graph.from_json(config['INPUT_GRAPH_FILENAME'])
    # For each node in graph, set 'pos' keyword to position
    for node in graph.nodes():
        graph.nodes[node]['pos'] = (graph.nodes[node][config['X_POSITION']],
                                    graph.nodes[node][config['Y_POSITION']])

    dual = facefinder.restricted_planar_dual(graph)

    return graph, dual

def determine_special_faces(graph, dist):
    """Determines the special faces, which are those nodes whose distance is
    at least k

    Args:
        graph (Gerrychain Graph): graph to determine special faces of
        dist (numeric): distance such that nodes are considered special if
        they have a 'distance' of at least this value

    Returns:
        list: list of nodes which are special
    """
    return [node for node in graph.nodes() if graph.nodes[node]['distance'] >= dist]

def determine_special_faces_random(graph, exp=1):
    """Determines the special faces, which are determined randomly with the probability
    of a given node being considered special being proportional to its distance
    raised to the exp power

    Args:
        graph (Gerrychain Graph): graph to determine special faces of
        exp (float, optional): exponent appearing in probability of a given node
        being considered special. Defaults to 1.

    Returns:
        list: list of nodes which are special
    """
    max_dist = max(graph.nodes[node]['distance'] for node in graph.nodes())
    return [node for node in graph.nodes() if random.uniform < (graph.nodes[node]['distance'] / max_dist) ** exp]

def metamander_around_partition(graph, assignment, dual, secret=False, special_param=2):
    """Metamanders around a partition by determining the set of special faces,
    and then sierpinskifying them.

    Args:
        partition (Gerrychain Partition): Partition to metamander around
        dual (Networkx Graph): planar dual of partition's graph
        secret (Boolean): whether to metamander 'in secret'. If True, determines
        special faces randomly, else not.
        special_param (numeric): additional parameter passed to special faces function
    """

    partition = Partition(graph, assignment)
    # Set of edges which cross from one district to another one
    cross_edges = facefinder.compute_cross_edges(partition)
    # Edges of dual graph corresponding to cross_edges
    dual_crosses = [edge for edge in dual.edges if dual.edges[edge]['original_name'] in cross_edges]

    # Assigns the graph distance from the dual_crosses to each node of the dual graph
    facefinder.distance_from_partition(dual, dual_crosses)
    # Assign special faces based upon set distances
    if secret:
        special_faces = determine_special_faces_random(dual, special_param)
    else:
        special_faces = determine_special_faces(dual, special_param)
    # Metamander around the partition by Sierpinskifying the special faces
    face_sierpinski_mesh(partition, special_faces)

def createDirectory(config):
    num = 0
    suffix = lambda x: f'-{x}' if x != 0 else ''
    while os.path.exists(config['EXPERIMENT_NAME'] + suffix(num)):
        num += 1
    os.mkdir(config['EXPERIMENT_NAME'] + suffix(num))
    metadataFile = os.path.join(config['EXPERIMENT_NAME'] + suffix(num), config['METADATA_FILE'])
    with open(metadataFile, 'w') as metaFile:
        json.dump(config, metaFile, indent=2)
    dataFile = os.path.join(config['EXPERIMENT_NAME'] + suffix(num), config['DATA_FILE'])
    with open(dataFile, 'x') as f:
        f.write('[]')
    return dataFile

def runMetamander(graph, assignment, dual, config, id):
    try:
        metamander_around_partition(graph, assignment, dual)
        # Initialize partition
        election = Election(
                            config['ELECTION_NAME'],
                            {'PartyA': config['PARTY_A_COL'],
                            'PartyB': config['PARTY_B_COL']}
                            )

        updaters = {'population': Tally(config['POP_COL']),
                    # 'cut_edges': cut_edges,
                    config['ELECTION_NAME'] : election,
                    }
        partition = Partition(graph, assignment=config['ASSIGN_COL'], updaters=updaters)
        # List of districts in original graph
        parts = list(set([graph.nodes[node][config['ASSIGN_COL']] for node in graph.nodes()]))
        # Ideal population of districts
        ideal_pop = sum([graph.nodes[node][config['POP_COL']] for node in graph.nodes()]) / len(parts)
        popbound = within_percent_of_ideal_population(partition, config['EPSILON'])
        # Determine proposal for generating spanning tree based upon parameter
        if config['CHAIN_TYPE'] == "tree":
            tree_proposal = partial(recom, pop_col=config["POP_COL"], pop_target=ideal_pop,
                            epsilon=config['EPSILON'], node_repeats=config['NODE_REPEATS'])

        elif config['CHAIN_TYPE'] == "uniform_tree":
            tree_proposal = partial(recom, pop_col=config["POP_COL"], pop_target=ideal_pop,
                            epsilon=config['EPSILON'], node_repeats=config['NODE_REPEATS'])
        else:
            print("Chaintype used: ", config['CHAIN_TYPE'])
            raise RuntimeError("Chaintype not recognized. Use 'tree' or 'uniform_tree' instead")

        # Chain to be run
        chain = MarkovChain(tree_proposal, Validator([popbound]), accept=accept.always_accept, initial_state=partition,
                                total_steps=config['META_RUN_LENGTH'])

        electionDict = {
            'seats' : (lambda x: x[config['ELECTION_NAME']].seats('PartyA')),
            'won' : (lambda x: x[config['ELECTION_NAME']].seats('PartyA')),
            'efficiency_gap' : (lambda x: x[config['ELECTION_NAME']].efficiency_gap()),
            'mean_median' : (lambda x: x[config['ELECTION_NAME']].mean_median()),
            'mean_thirdian' : (lambda x: x[config['ELECTION_NAME']].mean_thirdian()),
            'partisan_bias' : (lambda x: x[config['ELECTION_NAME']].partisan_bias()),
            'partisan_gini' : (lambda x: x[config['ELECTION_NAME']].partisan_gini())
        }

        # Run chain, save each desired statistic
        statistics = {statistic : [] for statistic in config['ELECTION_STATISTICS']}
        for i, part in enumerate(chain):
            # Save statistics of partition
            for statistic in config['ELECTION_STATISTICS']:
                statistics[statistic].append(electionDict[statistic](part))
            if i % 500 == 0:
                print('{}: {}'.format(id, i))
        lock.acquire()
        with open(config['DATA_FILE_LOCATION'], 'rb+') as filehandle:
            filehandle.seek(-1, os.SEEK_END)
            filehandle.truncate()
        with open(config['DATA_FILE_LOCATION'], 'r+') as f:
            if f.read() == '[':
                f.write(json.dumps(statistics, indent=2) + ']')
            else:
                f.write(', ' + json.dumps(statistics, indent=2) + ']')
        lock.release()
    except Exception as e:
        # Print notification if any experiment fails to complete
        track = traceback.format_exc()
        print(id, track)

def init(l):
    global lock
    lock = l

def main():
    """Runs a single experiment with the given config file. Loads a graph,
    runs a Chain to search for a Gerrymander, metamanders around that partition,
    runs another chain, and then saves the generated data.

    Args:
        config_data (Object): configuration of experiment loaded from JSON file
        id (String): id of experiment, used in tags to differentiate between
        experiments
    """
    global config
    config = {
        "POP_COL" : "population",
        "ASSIGN_COL" : "part",
        "INPUT_GRAPH_FILENAME" : "GRID_GRAPH",
        "X_POSITION" : "C_X",
        "Y_POSITION" : "C_Y",
        'EPSILON' : 0.01,
        "ELECTION_NAME" : "2016_Presidential",
        "PARTY_A_COL" : "EL16G_PR_R",
        "PARTY_B_COL" : "EL16G_PR_D",
        "NODE_REPEATS" : 1,
        "SIERPINSKI_POP_STYLE" : "random",
        "CHAIN_TYPE" : "tree",
        "ELECTION_STATISTICS" : ["seats", "efficiency_gap", "mean_median"],
        'EXPERIMENT_NAME' : 'statisticVsMetaStatistic',
        'METADATA_FILE' : 'config',
        'DATA_FILE' : 'data',
        'NUM_PROCESSORS' : 4,
        'NUM_SAMPLE_PARTITIONS' : 10,
        'STEPS_IN_BETWEEN_SAMPLES' : 20,
        'META_RUN_LENGTH': 50,
        'GRID_SIZE': 30,
        'NUM_DISTRICTS': 6,
        'PERCENT_PARTY_A': 0.45,
    }
    try:
        timeBeg = time.time()
        config['DATA_FILE_LOCATION'] = createDirectory(config)
        # Get graph and dual graph
        graph, dual = preprocessing(config)
        # List of districts in original graph
        parts = list(set([graph.nodes[node][config['ASSIGN_COL']] for node in graph.nodes()]))
        # Ideal population of districts
        ideal_pop = sum([graph.nodes[node][config['POP_COL']] for node in graph.nodes()]) / len(parts)
        # Initialize partition
        election = Election(
                            config['ELECTION_NAME'],
                            {'PartyA': config['PARTY_A_COL'],
                            'PartyB': config['PARTY_B_COL']}
                            )

        updaters = {'population': Tally(config['POP_COL']),
                    'cut_edges': cut_edges,
                    config['ELECTION_NAME'] : election
                    }
        initPartition = Partition(graph=graph, assignment=config['ASSIGN_COL'], updaters=updaters)
        popbound = within_percent_of_ideal_population(initPartition, config['EPSILON'])
        # Determine proposal for generating spanning tree based upon parameter
        tree_proposal = partial(recom, pop_col=config["POP_COL"], pop_target=ideal_pop,
                        epsilon=config['EPSILON'], node_repeats=config['NODE_REPEATS'])
        # Chain to be run
        chain = MarkovChain(tree_proposal, Validator([popbound]), accept=accept.always_accept, initial_state=initPartition,
                                total_steps=config['NUM_SAMPLE_PARTITIONS'] * config['STEPS_IN_BETWEEN_SAMPLES'])

        # Run NUM_EXPERIMENTS experiments using NUM_PROCESSORS - 1 extra processors
        l = Lock()
        pool = Pool(config['NUM_PROCESSORS'] - 1, initializer=init, initargs=(l,))
        for i, partition in enumerate(chain):
            if i != 0 and i % config['STEPS_IN_BETWEEN_SAMPLES'] == 0:
                pool.apply_async(runMetamander, args = (graph, partition.assignment, dual, config, i, ))
        pool.close()
        pool.join()
        print('All experiments completed in {:.2f} seconds'.format(time.time() - timeBeg))
    except Exception as e:
        # Print notification if any experiment fails to complete
        track = traceback.format_exc()
        print(track)
        print('Experiment failed to complete after {:.2f} seconds'.format(time.time() - timeBeg))

if __name__ == '__main__':
    main()