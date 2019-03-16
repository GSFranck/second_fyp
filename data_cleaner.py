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


main_data = 'data.txt'



G = nx.read_adjlist( main_data , delimiter = " ", nodetype = int )
bottom , top = nx.bipartite.sets(G)


if len(bottom) > len(top):
    smol_set = list(top)
else:
    smol_set = list(bottom)

the_entires_shit = []
clean_shit = []

#print(smol_set)

with open ( main_data , 'r') as main:
    for line in main:
        adj_list_temp = line.strip('\n').split(' ')
        #adj_list_temp = [int(i) for i in adj_list_temp]
        the_entires_shit.append(adj_list_temp)

# writing data to new file, without unnessescary lines.

with open ('clean_data.txt', 'w') as file:
    for enum,line in enumerate(the_entires_shit):
        print("line in entire_shit", line)
        if int(line[0]) in smol_set:         #line[0] first node in list (costumer)
            for i in line:
                file.write(i + ' ')
            file.write('\n')

            


    
