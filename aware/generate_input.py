# Generate input data

from collections import Counter
import pandas as pd
import numpy as np
import random
import networkx as nx
import torch
from torch_geometric.data import Data


def split_data(num_y):
    split_idx = list(range(num_y))
    random.shuffle(split_idx)
    train_idx = split_idx[ : int(len(split_idx) * 0.8)] # Train mask
    train_mask = torch.zeros(num_y, dtype = torch.bool)
    train_mask[train_idx] = 1
    val_idx = split_idx[ int(len(split_idx) * 0.8) : int(len(split_idx) * 0.9)] # Val mask
    val_mask = torch.zeros(num_y, dtype=torch.bool)
    val_mask[val_idx] = 1
    test_idx = split_idx[int(len(split_idx) * 0.9) : ] # Test mask
    test_mask = torch.zeros(num_y, dtype=torch.bool)
    test_mask[test_idx] = 1
    return train_mask, val_mask, test_mask


def read_ppi(f, G, run):
    orig_ppi_layers = dict()
    ppi_layers = dict()
    ppi_train = dict()
    ppi_val = dict()
    ppi_test = dict()
    with open(f) as fin:
        for lin in fin:
            cluster = lin.split("\t")[1]

            if len(lin.split("\t")) > 2: ppi = lin.strip().split("\t")[2].split(",")
            else: ppi = []
                
            # Relabel PPI nodes
            mapping = {n: idx for idx, n in enumerate(ppi)}
            ppi_layers[cluster] = nx.relabel_nodes(G.subgraph(ppi), mapping) 
            orig_ppi_layers[cluster] = G.subgraph(ppi)
            assert nx.is_connected(ppi_layers[cluster])
            
            # Split into train/val/test
            ppi_train[cluster], ppi_val[cluster], ppi_test[cluster] = split_data(len(ppi_layers[cluster].edges))
    return orig_ppi_layers, ppi_layers, ppi_train, ppi_val, ppi_test


def create_data(G, train_mask, val_mask, test_mask, node_type, edge_type, feat_type='normal', feat_mat=2048):
    x = torch.zeros(len(G.nodes), feat_mat)
    if feat_type == 'normal': x = torch.normal(x, std=1)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    y = torch.ones(edge_index.size(1))
    num_classes = len(torch.unique(y))
    node_type = torch.tensor(node_type)
    edge_type = torch.tensor(edge_type)
    new_G = Data(x = x, y = y, num_classes = num_classes, edge_index = edge_index, node_type = node_type, edge_attr = edge_type, train_mask = train_mask, val_mask = val_mask, test_mask = test_mask)
    return new_G


def read_data(G_f, ppi_f, cci_bto_f, feat_mat, run, good_annot=True, global_G=False):

    # Read global PPI 
    G = nx.read_edgelist(G_f)
    if global_G:
        mapping = {n: idx for idx, n in enumerate(G)}
        relabeled_G = nx.relabel_nodes(G, mapping)
        print("Number of nodes:", len(G.nodes), "Number of edges:", len(G.edges))
        G_train, G_val, G_test = split_data(len(G.edges))
        G_nodetype = [2] * len(G.nodes) # protein nodes = 2
        G_edgetype = [3] * len(G.edges) # protein-protein edge
        G_data = dict()
        G_layer = dict()
        G_data[0] = create_data(relabeled_G, G_train, G_val, G_test, G_nodetype, G_edgetype, feat_mat=feat_mat)
        G_layer["Global"] = G
        print(G_data)
    
    # Read PPI layers
    orig_ppi_layers, ppi_layers, ppi_train, ppi_val, ppi_test = read_ppi(ppi_f, G, run)
    print("Number of PPI layers:", len(ppi_layers), len(ppi_train), len(ppi_val), len(ppi_test))

    # Read CCI-BTO
    cci_bto = nx.read_edgelist(cci_bto_f, data=False, delimiter = "\t")
    print("Number of nodes:", len(cci_bto.nodes), "Number of edges:", len(cci_bto.edges))
    orig_cci_bto = cci_bto
    print("Number of nodes:", len(cci_bto.nodes), "Number of edges:", len(cci_bto.edges))
    cci_bto_mapping = {n: i for i, n in enumerate(sorted(ppi_layers))}
    cci_bto_mapping.update({n: i + len(ppi_layers) for i, n in enumerate(sorted([n for n in cci_bto.nodes if "BTO" in n]))})
    assert len(cci_bto_mapping) == len(cci_bto.nodes), set(cci_bto.nodes).difference(set(list(cci_bto_mapping.keys())))

    # Set up Data object
    cci_bto_nodetype = [0 if "BTO" in n else 1 for n in cci_bto_mapping] # Tissue nodes = 0, Cell-type nodes = 1, protein nodes = 2
    cci_bto_edgetype = []
    for edges in cci_bto.edges:
        if "BTO" in edges[0] and "BTO" in edges[1]: cci_bto_edgetype.append(0) # tissue-tissue edge
        elif "BTO" in edges[0] and "BTO" not in edges[1]: cci_bto_edgetype.append(1) # tissue-cell edge
        elif "BTO" not in edges[0] and "BTO" in edges[1]: cci_bto_edgetype.append(1) # cell-tissue edge
        elif "BTO" not in edges[0] and "BTO" not in edges[1]: cci_bto_edgetype.append(2) # cell-cell edge
    tissue_neighbors = {cci_bto_mapping[t]: [cci_bto_mapping[n] for n in cci_bto.neighbors(t)] for t in cci_bto.nodes if "BTO" in t}
    cci_bto = nx.relabel_nodes(cci_bto, cci_bto_mapping)
    #cci_bto_train, cci_bto_val, cci_bto_test = split_data(len(cci_bto.edges))
    cci_bto_mask = torch.ones(len(cci_bto.edges), dtype = torch.bool) # Pass in all meta graph edges during training, validation, and test
    cci_bto_data = create_data(cci_bto, cci_bto_mask, cci_bto_mask, cci_bto_mask, cci_bto_nodetype, cci_bto_edgetype, 'zeros', feat_mat=feat_mat)

    # Set up PPI Data objects
    ppi_layers = {cci_bto_mapping[k]: v for k, v in ppi_layers.items() if k in cci_bto_mapping}
    ppi_train = {cci_bto_mapping[k]: v for k, v in ppi_train.items() if k in cci_bto_mapping}
    ppi_val = {cci_bto_mapping[k]: v for k, v in ppi_val.items() if k in cci_bto_mapping}
    ppi_test = {cci_bto_mapping[k]: v for k, v in ppi_test.items() if k in cci_bto_mapping}    
    ppi_data = dict()
    for cluster, ppi in ppi_layers.items():
        ppi_nodetype = [2] * len(ppi.nodes) # protein nodes = 2
        ppi_edgetype = [3] * len(ppi.edges) # protein-protein edge
        ppi_data[cluster] = create_data(ppi, ppi_train[cluster], ppi_val[cluster], ppi_test[cluster], ppi_nodetype, ppi_edgetype, feat_mat=feat_mat)

    #  Set up edge attr dict
    edge_attr_dict = {"tissue_tissue": 0, "tissue_cell": 1, "cell_cell": 2, "protein_protein": 3}
    
    # Return only global PPI network data
    if global_G:
        return G_data, cci_bto_data, edge_attr_dict, {"Global": 0}, tissue_neighbors, G_layer, orig_cci_bto
    
    # Return celltype specific PPI network data
    return ppi_data, cci_bto_data, edge_attr_dict, cci_bto_mapping, tissue_neighbors, orig_ppi_layers, orig_cci_bto


def subset_ppi(num_subset, ppi_data, ppi_layers):
    
    # Take a subset of PPI data objects
    new_ppi_data = dict()
    for celltype, ppi in ppi_data.items():
        if len(new_ppi_data) < num_subset:
            new_ppi_data[celltype] = ppi
    ppi_data = new_ppi_data

    # Take a subset of PPI layers
    new_ppi_layers = dict()
    for celltype, ppi in ppi_layers.items():
        if len(new_ppi_layers) < num_subset:
            new_ppi_layers[celltype] = ppi_layers[celltype]
    ppi_layers = new_ppi_layers

    return ppi_data, ppi_layers


def get_metapaths():
    ppi_metapaths = [[3]] # Get PPI metapaths
    cci_bto_metapaths = [[0], [1], [2]]
    return ppi_metapaths, cci_bto_metapaths


def get_centerloss_labels(args, celltype_map, ppi_layers):
    center_loss_labels = []
    train_mask = []
    val_mask = []
    test_mask = []
    print(celltype_map)
    for celltype, ppi in ppi_layers.items():
        center_loss_labels += [celltype_map[celltype]] * len(ppi.nodes)
    center_loss_idx = random.sample(range(len(center_loss_labels)), len(center_loss_labels))
    train_mask = center_loss_idx[ : int(0.8 * len(center_loss_idx))]
    val_mask = center_loss_idx[len(train_mask) : len(train_mask) + int(0.1 * len(center_loss_idx))]
    test_mask = center_loss_idx[len(train_mask) + len(val_mask) : ]
    print("Center loss labels:", Counter(center_loss_labels))
    return center_loss_labels, train_mask, val_mask, test_mask
