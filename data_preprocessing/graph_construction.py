import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
from geopy.distance import geodesic

__file__ = "your_path/graph_construction.py"

# load data
features_comuni_global = np.load(
    os.path.join(os.path.dirname(__file__), "../data", "features_comuni_global.npy"),
    allow_pickle=True,
).item()
dict_df_comuni_global = np.load(
    os.path.join(os.path.dirname(__file__), "../data", "dict_df_comuni_global.npy"),
    allow_pickle=True,
).item()

G = nx.Graph()
for k in tqdm(features_comuni_global.keys()):
    G.add_node(
        k,
        popolazione=features_comuni_global[k]["popolazione"],
        reddito=features_comuni_global[k]["reddito_procapite"],
        latitudine=features_comuni_global[k]["latitudine_scaled"],
        longitudine=features_comuni_global[k]["longitudine_scaled"],
        area=features_comuni_global[k]["area_scaled"],
    )

for k in tqdm(features_comuni_global.keys()):
    for j in features_comuni_global.keys():
        if k != j:

            # spending coefficient
            features_nodek = np.array(
                dict_df_comuni_global[k]["spesa media"]
            ) / np.array(dict_df_comuni_global[k]["popolazione_anno"])
            features_nodej = np.array(
                dict_df_comuni_global[j]["spesa media"]
            ) / np.array(dict_df_comuni_global[j]["popolazione_anno"])

            coeff_spesa = np.exp(-np.linalg.norm(features_nodek - features_nodej) ** 2)

            # area coefficient
            coeff_area = np.exp(
                -np.linalg.norm(
                    np.array(features_comuni_global[k]["area_scaled"])
                    - np.array(features_comuni_global[j]["area_scaled"])
                )
                ** 2
            )

            # distance coefficient
            pos1 = (
                features_comuni_global[k]["latitudine_scaled"],
                features_comuni_global[k]["longitudine_scaled"],
            )
            pos2 = (
                features_comuni_global[j]["latitudine_scaled"],
                features_comuni_global[j]["longitudine_scaled"],
            )
            distance = geodesic(pos1, pos2).km
            coeff_distance = np.exp(-distance)

            # final edge weight
            coeff = (
                (0.25) * coeff_spesa + (0.25) * coeff_area + (0.5) * (coeff_distance)
            )

            G.add_edge(k, j, weight=coeff, added_info=coeff)
        else:
            pass

edge_list = nx.to_pandas_edgelist(G)

# create the final graph
names = []
for i in G.nodes():
    names.append(i)

indexes = [i for i in range(0, len(names))]
zip_iterator = zip(indexes, names)
a_dictionary = dict(zip_iterator)

adj = nx.from_pandas_edgelist(edge_list, edge_attr=["weight"])
adj = pd.DataFrame(nx.adjacency_matrix(adj, weight="weight").todense())

newgraph = nx.from_pandas_adjacency(adj)

newgraph = nx.relabel_nodes(newgraph, a_dictionary)
popolazione = nx.get_node_attributes(G, "popolazione")
reddito = nx.get_node_attributes(G, "reddito")
latitudine = nx.get_node_attributes(G, "latitudine")
longitudine = nx.get_node_attributes(G, "longitudine")
nx.set_node_attributes(newgraph, popolazione, "popolazione")
nx.set_node_attributes(newgraph, reddito, "reddito")
nx.set_node_attributes(newgraph, reddito, "latitudine")
nx.set_node_attributes(newgraph, reddito, "longitudine")

nx.set_edge_attributes(newgraph, nx.get_edge_attributes(G, "added_info"), "added_info")

# compute laplacian
adjacency = adj.copy()

D = np.diag(np.sum(adjacency, axis=1))
I = np.identity(adjacency.shape[0])
D_inv_sqrt = np.linalg.inv(np.sqrt(D))
L = I - np.dot(D_inv_sqrt, adjacency).dot(D_inv_sqrt)

normalized_laplacian_version = L.copy()

# save the graph and the matrix
with open(
    os.path.join(os.path.dirname(__file__), "../data", "newgraph.pkl"), "wb"
) as f:
    pickle.dump(newgraph, f)

np.save(
    os.path.join(
        os.path.dirname(__file__), "../data", "normalized_laplacian_version.npy"
    ),
    normalized_laplacian_version,
)
np.save(
    os.path.join(os.path.dirname(__file__), "../data", "adjacency.npy"),
    adj,
)
