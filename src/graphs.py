import networkx as nx
from networkx.readwrite import json_graph  #TODO: this seems to not be used. Figure out if i can remove it
import matplotlib.pyplot as plt
import numpy as np 
from utils import jsonify

"""
function to generate graphs with different topology strategies and weight generation strategies
topologyStrategy
   0 = 3-regular
   1 = random/Erdos-Eenyi
   2 = Fully Connected
   3 = Barabasi-Albert (power-law) with preferential attachment m=1
weightStrategy
   0 = Random Choice w_ij = {-1,+1} uniformly
   1 = Uniform: w_ij ~ U[-1,1]
   2 = Gaussian: w_ij ~ N(0,1)
"""
def genGraph(numNodes, topologyStrategy, weightStrategy, weighNodes = False):
    G = nx.Graph()  #generate empty undirected graph
    
    if topologyStrategy<0 or topologyStrategy>3 or weightStrategy<0 or weightStrategy>2:
        print("topologyStrategy(" + str(topologyStrategy) + ") or weightStrategy(" + str(weightStrategy) + ") out of bounds")
        print("returning empty graph")
        return G  #return an empty graph
    
    ## Generate list of edges based on topologyStrategy
    if topologyStrategy == 0:   #3-regular
        G = nx.random_regular_graph(d=3, n=numNodes)  #NOTE: 3-regular graphs must have even number of nodes
    
    if topologyStrategy == 1:  #random/Erdos-Eenyi
        edgeProb = 0.5
        G = nx.erdos_renyi_graph(numNodes, edgeProb)
        
    if topologyStrategy == 2:  #Fully Connected
        G = nx.complete_graph(numNodes)
        
    if topologyStrategy == 3:  #Barabasi-Albert (power-law)
        G = nx.barabasi_albert_graph(numNodes, 1)
        
    ## assign weights to edges based on weightStrategy
    for u,v in G.edges():
        if weightStrategy == 0:    #Random Choice
            G[u][v]['weight'] = np.random.choice([-1,1])

        if weightStrategy == 1:    #Uniform Dist
            G[u][v]['weight'] = np.random.uniform(low=-1, high=1)

        if weightStrategy == 2:    #Gaussian Dist
            G[u][v]['weight'] = np.random.normal(0,1)
    
    ## assign weights to nodes if needed
    if weighNodes == True:
        #give nodes weight of -1 or +1 randomly
        for i in G.nodes():
            G.nodes[i]["weight"] = np.random.choice([-1, 1])
    else:
        #assign all nodes weight of 0
        for i in G.nodes():
            G.nodes[i]["weight"] = 0.0
    
    return G

"""
Function to display a graph with its edges and verticies labeled
"""
def drawGraph(G, showNodeWeight=False):
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=500)

    # --- Draw node labels ---
    if showNodeWeight:
        #Show weight AND label
        node_labels = {i: G.nodes[i].get('weight', 0) for i in G.nodes()}
    else:
        #Only show label
        node_labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='black')

    # --- Draw edge labels for weights ---
    edge_labels = {(u, v): round(d.get('weight', 0.0), 2) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()
    return

"""
Function to make all graph attributes json serializable
"""
def sanitize_for_json(G):
    H = G.copy()

    # Fix node attributes
    for _, attrs in H.nodes(data=True):
        for k, v in attrs.items():
            attrs[k] = jsonify(v)

    # Fix edge attributes
    for _, _, attrs in H.edges(data=True):
        for k, v in attrs.items():
            attrs[k] = jsonify(v)

    # Fix graph-level attributes
    for k, v in H.graph.items():
        H.graph[k] = jsonify(v)

    return H