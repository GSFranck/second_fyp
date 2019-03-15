import networkx as nx
import backboning as bb 
import network_map2 as nm2 
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from operator import itemgetter
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


def graph_reader(inputfile, *args): # Reads the file, cleans isolates and non_connected nodes.
    G = nx.read_adjlist(inputfile, delimiter = " ", nodetype = int)
    isolates = nx.isolates(G) 

    if nx.is_connected(G) == False:
        print("Graph is not connected")
        H = G.degree()
        non_connected = [n[0] for n in H if n[1] == 0]
        G.remove_nodes_from(non_connected)
        print(len(non_connected), "disconnected nodes removed")
     

    if len(list(nx.isolates(G))) >= 1:
        print("There are", list(len(isolates)), "isolated nodes")
        G.remove_nodes_from(isolates)
   
    else:
        print("No isolates")
        print("Number of connected components:", nx.number_connected_components(G))

    return G

def plot_network(somegraph, graph_type, *args): #Plots. NOTE: IT IS EXTREMELY SLOW AS IT BUILDS ON PAGERANK AND CLOSENESS MEASURE 
    if graph_type == 1:
        nx.draw_kamada_kawai(somegraph, labels = {n: n for n in somegraph.nodes})
        plt.show()
    
    if graph_type == 2:
        nx.draw_spectral(somegraph, labels = {n: n for n in somegraph.nodes})
        plt.show()

    if graph_type == 3:
        nx.draw_spring(somegraph, labels = {n: n for n in somegraph.nodes})
        plt.show()
    
    if graph_type == 4:
        close = nx.closeness_centrality(somegraph)
        posNX = nx.spring_layout(somegraph)
        pr = nx.pagerank(somegraph, alpha = 0.1)
        nsize = np.array([v for v in close.values()])
        nsize = (nsize - min(nsize)) /(max(nsize) - min(nsize))
        nodes = nx.draw_networkx_nodes(somegraph, pos = posNX, node_size = nsize)
        edges = nx.draw_networkx_edges(somegraph, pos = posNX, alpha = 0.2, with_labels = True)
        plt.show()

def hist_plotter(somegraph): #Shitty histogram plotter

    degrees = somegraph.degree() #Returns a tuple
    print(degrees)
    only_degrees = [x[1] for x in degrees] 
    
    plt.hist(only_degrees)
    plt.show()

    return  

def check_bi(somegraph): #Checks for bipartite. Has to run, global variables are declared in here
        global bottom, top, list_of_nodes
        bottom, top = bipartite.sets(somegraph)
        print(top, bottom) 
      
        print("This many nodes in top:", len(top), "and in bottom:", len(bottom))
        list_of_nodes = list(top)

        return list_of_nodes

def bipartite_plot(somegraph, top): #A plot function for bipartite graphs (didnt fit the other function)
    pos = dict()
    pos.update( (n, (1, i)) for i, n in enumerate(bottom)) # put nodes from X at x=1
    pos.update( (n, (2, i)) for i, n in enumerate(top)) # put nodes from Y at x=2
    nx.draw(somegraph, pos=pos)
    plt.show()

def projector(network, nodes, projector_type): #Found a bug in Micheles code. This is copypasted, but I implemented it with another parameter to change between projections faster 

    if projector_type == 1:
        T = nx.algorithms.bipartite.matrix.biadjacency_matrix(network, row_order = nodes)
        U = T * T.T
        U.setdiag(0)
        U.eliminate_zeros()
        G = nx.from_scipy_sparse_matrix(U)
        return nx.relabel_nodes(G, {i: nodes[i] for i in range(len(nodes))})


    if projector_type == 2:
        T = nx.algorithms.bipartite.matrix.biadjacency_matrix(network, row_order = nodes)
        T /= T.sum(axis = 0)
        T = sparse.csr_matrix(T)
        U = T * T.T
        U.setdiag(0)
        U.eliminate_zeros()
        G = nx.from_scipy_sparse_matrix(U)
        return nx.relabel_nodes(G, {i: nodes[i] for i in range(len(nodes))})


    if projector_type == 3:
        T = nx.algorithms.bipartite.matrix.biadjacency_matrix(network, row_order = nodes)
        T_norm = normalize(T, norm = 'l1', axis = 1) # Divide each row element with the row sum (Eq. [1] in the paper)
        T_t_norm = normalize(T.T, norm = 'l1', axis = 1) # Divide each row element of the transpose with the transposed row sum (Eq. [2] in the paper) 
        T = T_norm.dot(T_t_norm) # Multiply (Eq. [3] in the paper)

        if nx.is_directed(network): 
            G = nx.from_scipy_sparse_matrix(T, create_using = nx.DiGraph())
        else:
            G = nx.from_scipy_sparse_matrix(T)
        return nx.relabel_nodes(G, {i: nodes[i] for i in range(len(nodes))})

def community_disc(somegraph): ## ASSUMES ALREADY PROJECTED GRAPH --> Returns a frozenset of nodes in communities. 
    B = nx.karate_club_graph()
    c = list(greedy_modularity_communities(B))

    return c
    
mygraph = graph_reader("data_chunk.txt") # Reads it 
bipart_g = check_bi(mygraph) # Call this function in order to gain global variables top and bottom. 
projected_graph = projector(mygraph, list_of_nodes, projector_type = 3) 
communities = community_disc(projected_graph)


""" 
    Projector function:
    1: Simple projector
    2: Hyperbolic
    3: ProbS projection

    Plot function:
    I like number 4 the best, as the others are pretty shit.
    NOTE: IT IS EXTREMELY SLOW AS IT BUILDS ON PAGERANK AND CLOSENESS MEASURE 

"""





