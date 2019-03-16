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
    
    plt.figure(figsize=(500,500))
    
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
    #plt.savefig("plot.png") #check on dpi=
    

graph = graph_reader("toy_data.txt")
plot_network(graph, 3)


