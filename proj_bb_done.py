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
    global nodes, rows, cols
    nodes = nx.algorithms.bipartite.basic.sets(G)
    rows = sorted(list(nodes[1]))
    cols = sorted(list(nodes[0])) 
    isolates = nx.isolates(G)

    if nx.is_connected(G) == False:
        print("Graph is not connected")
        H = G.degree()
        non_connected = [n[0] for n in H if n[1] == 0]
        G.remove_nodes_from(non_connected)
        print(len(non_connected), "disconnected nodes removed")
     
    if len(list(nx.isolates(G))) >= 1:
        print("There are", len(list(isolates)), "isolated nodes")
        G.remove_nodes_from(isolates)
   
    else:
     
        print("No isolates")
        print("Number of connected components:", nx.number_connected_components(G))

    return G

def plot_network(somegraph, graph_type, *args): #Plots. NOTE: IT IS EXTREMELY SLOW AS IT BUILDS ON PAGERANK AND CLOSENESS MEASURE 
    if graph_type == 1:
        nx.draw_kamada_kawai(somegraph, labels = {n: n for n in somegraph.nodes})
    
    if graph_type == 2:
        nx.draw_spectral(somegraph, labels = {n: n for n in somegraph.nodes})

    if graph_type == 3:
        nx.draw_spring(somegraph, labels = {n: n for n in somegraph.nodes})
    
    if graph_type == 4:
        close = nx.closeness_centrality(somegraph)
        posNX = nx.spring_layout(somegraph)
        pr = nx.pagerank(somegraph, alpha = 0.1)
        nsize = np.array([v for v in close.values()])
        nsize = (nsize - min(nsize)) /(max(nsize) - min(nsize))
        nodes = nx.draw_networkx_nodes(somegraph, pos = posNX, node_size = nsize)
        edges = nx.draw_networkx_edges(somegraph, pos = posNX, alpha = 0.2, with_labels = True)
    
    plt.show()
    #plt.savefig("plot.png", dpi=1000) #check on dpi=
    
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
    
        print("This many nodes in top:", len(top), "and in bottom:", len(bottom))
        

        return 

def bipartite_plot(somegraph, top): #A plot function for bipartite graphs (didnt fit the other function)
    pos = dict()
    pos.update( (n, (1, i)) for i, n in enumerate(bottom)) # put nodes from X at x=1
    pos.update( (n, (2, i)) for i, n in enumerate(top)) # put nodes from Y at x=2
    nx.draw(somegraph, pos=pos)
    plt.show()

def projector(network, nodes, projector_type): #Found a bug in Micheles code. This is copypasted, but I implemented it with another parameter to change between projections faster 
    
    # Simple projection
    if projector_type == 1:    
        T = nx.algorithms.bipartite.matrix.biadjacency_matrix(network, row_order = nodes)
        U = T * T.T
        U.setdiag(0)
        U.eliminate_zeros()
        G = nx.from_scipy_sparse_matrix(U)
        return nx.relabel_nodes(G, {i: nodes[i] for i in range(len(nodes))})

    #  Hyperbolic weights projecton
    if projector_type == 2:    
        T = nx.algorithms.bipartite.matrix.biadjacency_matrix(network, row_order = nodes)
        T /= T.sum(axis = 0)
        T = sparse.csr_matrix(T)
        U = T * T.T
        U.setdiag(0)
        U.eliminate_zeros()
        G = nx.from_scipy_sparse_matrix(U)
        return nx.relabel_nodes(G, {i: nodes[i] for i in range(len(nodes))})

     # Probs projection 
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


def transform_for_bb(G):
    G_df = nx.to_pandas_edgelist(G)
    G_df.columns = ("src", "trg", "nij")
    G_df = bb.make_symmetric(G_df)
    return G_df

def backboning(somegraph, type):
    
    if type == 1:     
        return bb.noise_corrected(somegraph)

    if type == 2:
        return bb.high_salience_skeleton(somegraph)

    if type == 3: 
        return bb.naive(somegraph)

    if type == 4: 
        return bb.disparity_filter(somegraph)


load_graph = graph_reader('data.txt')
graph_projected = projector(load_graph, rows, 1)
graph_backbone_ready = transform_for_bb(graph_projected)
backboned = backboning(graph_backbone_ready, 1)
backboned.to_csv('noise_corr.csv', sep = ";", encoding='utf-8')




""" 
    Projector function:
    1: Simple projector
    2: Hyperbolic
    3: ProbS projection

    Plot function:
    I like number 4 the best, as the others are pretty shit.
    NOTE: IT IS EXTREMELY SLOW AS IT BUILDS ON PAGERANK AND CLOSENESS MEASURE 

USEFUL CHUNKS

mygraph = graph_reader(testfile)
bipart_g = check_bi(mygraph) # Call this function in order to gain global variables top and bottom. 
projected_graph = projector(mygraph, list_of_nodes, projector_type = 3) 


print(bb.test_densities(G_simple_df_naive, 0, 3, 1))    

#print(comms)
#comms = list(nx.algorithms.community.label_propagation.label_propagation_communities(simple))
np.savetxt('data.csv', (col1_array, col2_array, col3_array, col4_array, col5_array), delimiter=',')

G_simple_df_naive = bb.naive(G_simple_df, undirected = True)
G_simple_df_naive_bb = bb.thresholding(G_simple_df_naive, 1)
G_simple_naive = nx.from_pandas_edgelist(G_simple_df_naive_bb, source = "src", target = "trg", edge_attr = ("nij", "score"))
graph_simply_projected = transform_for_bb(simple)


PROJECTION:

print(G_simple_df)


def backboning(somegraph, type):
    if type == 1:
        return bb.noise_corrected(somegraph)

    if type == 2:
        return bb.


"""




