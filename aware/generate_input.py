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


def read_ppi(ppi_dir):
    orig_ppi_layers = dict()
    ppi_layers = dict()
    ppi_train = dict()
    ppi_val = dict()
    ppi_test = dict()

    for f in glob.glob(ppi_dir + "*"): # Expected format of filename: <PPI_DIR>/<CONTEXT>.<suffix>
        
        # Parse name of context
        context = f.split(ppi_dir)[1].split(".")[0]
        
        # Read edgelist
        ppi = nx.read_edgelist(f)
        
        # Relabel PPI nodes
        mapping = {n: idx for idx, n in enumerate(ppi.nodes())}
        ppi_layers[context] = nx.relabel_nodes(ppi, mapping)
        orig_ppi_layers[context] = ppi
        assert nx.is_connected(ppi_layers[context])
        
        # Split into train/val/test
        ppi_train[context], ppi_val[context], ppi_test[context] = split_data(len(ppi_layers[context].edges))
    return orig_ppi_layers, ppi_layers, ppi_train, ppi_val, ppi_test


def create_data(G, train_mask, val_mask, test_mask, node_type, edge_type, feat_type = 'normal', feat_mat = 2048):
    x = torch.zeros(len(G.nodes), feat_mat)
    if feat_type == 'normal': x = torch.normal(x, std=1)
    
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    
    y = torch.ones(edge_index.size(1))
    num_classes = len(torch.unique(y))
    
    node_type = torch.tensor(node_type)
    edge_type = torch.tensor(edge_type)
    
    new_G = Data(x = x, y = y, num_classes = num_classes, edge_index = edge_index, node_type = node_type, edge_attr = edge_type, train_mask = train_mask, val_mask = val_mask, test_mask = test_mask)
    return new_G


def read_data(ppi_dir, metagraph_f, feat_mat):
    
    # Read PPI layers
    orig_ppi_layers, ppi_layers, ppi_train, ppi_val, ppi_test = read_ppi(ppi_dir)
    print("Number of PPI layers:", len(ppi_layers), len(ppi_train), len(ppi_val), len(ppi_test))

    # Read metagraph
    metagraph = nx.read_edgelist(metagraph_f, data=False, delimiter = "\t")
    print("Number of nodes:", len(metagraph.nodes), "Number of edges:", len(metagraph.edges))
    orig_metagraph = metagraph
    print("Number of nodes:", len(metagraph.nodes), "Number of edges:", len(metagraph.edges))
    metagraph_mapping = {n: i for i, n in enumerate(sorted(ppi_layers))}
    metagraph_mapping.update({n: i + len(ppi_layers) for i, n in enumerate(sorted([n for n in metagraph.nodes if "BTO" in n]))})
    assert len(metagraph_mapping) == len(metagraph.nodes), set(metagraph.nodes).difference(set(list(metagraph_mapping.keys())))

    # Set up Data object
    metagraph_nodetype = [0 if "BTO" in n else 1 for n in metagraph_mapping] # Tissue nodes = 0, Cell-type nodes = 1, protein nodes = 2
    metagraph_edgetype = []
    for edges in metagraph.edges:
        if "BTO" in edges[0] and "BTO" in edges[1]: metagraph_edgetype.append(0) # tissue-tissue edge
        elif "BTO" in edges[0] and "BTO" not in edges[1]: metagraph_edgetype.append(1) # tissue-cell edge
        elif "BTO" not in edges[0] and "BTO" in edges[1]: metagraph_edgetype.append(1) # cell-tissue edge
        elif "BTO" not in edges[0] and "BTO" not in edges[1]: metagraph_edgetype.append(2) # cell-cell edge
    tissue_neighbors = {metagraph_mapping[t]: [metagraph_mapping[n] for n in metagraph.neighbors(t)] for t in metagraph.nodes if "BTO" in t}
    metagraph = nx.relabel_nodes(metagraph, metagraph_mapping)
    #metagraph_train, metagraph_val, metagraph_test = split_data(len(metagraph.edges))
    metagraph_mask = torch.ones(len(metagraph.edges), dtype = torch.bool) # Pass in all meta graph edges during training, validation, and test
    metagraph_data = create_data(metagraph, metagraph_mask, metagraph_mask, metagraph_mask, metagraph_nodetype, metagraph_edgetype, 'zeros', feat_mat=feat_mat)

    # Set up PPI Data objects
    ppi_layers = {metagraph_mapping[k]: v for k, v in ppi_layers.items() if k in metagraph_mapping}
    ppi_train = {metagraph_mapping[k]: v for k, v in ppi_train.items() if k in metagraph_mapping}
    ppi_val = {metagraph_mapping[k]: v for k, v in ppi_val.items() if k in metagraph_mapping}
    ppi_test = {metagraph_mapping[k]: v for k, v in ppi_test.items() if k in metagraph_mapping}    
    ppi_data = dict()
    for context, ppi in ppi_layers.items():
        ppi_nodetype = [2] * len(ppi.nodes) # protein nodes = 2
        ppi_edgetype = [3] * len(ppi.edges) # protein-protein edge
        ppi_data[context] = create_data(ppi, ppi_train[context], ppi_val[context], ppi_test[context], ppi_nodetype, ppi_edgetype, feat_mat=feat_mat)

    #  Set up edge attr dict
    edge_attr_dict = {"tissue_tissue": 0, "tissue_cell": 1, "cell_cell": 2, "protein_protein": 3}
    
    return ppi_data, metagraph_data, edge_attr_dict, metagraph_mapping, tissue_neighbors, orig_ppi_layers, orig_metagraph


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


def get_edgetypes():
    ppi_edgetypes = [[3]]
    mg_edgetypes = [[0], [1], [2]]
    return ppi_edgetypes, mg_edgetypes


def get_centerloss_labels(args, celltype_map, ppi_layers):
    center_loss_labels = []
    train_mask = []
    val_mask = []
    test_mask = []
    for celltype, ppi in ppi_layers.items():
        center_loss_labels += [celltype_map[celltype]] * len(ppi.nodes)
    
    center_loss_idx = random.sample(range(len(center_loss_labels)), len(center_loss_labels))
    train_mask = center_loss_idx[ : int(0.8 * len(center_loss_idx))]
    val_mask = center_loss_idx[len(train_mask) : len(train_mask) + int(0.1 * len(center_loss_idx))]
    test_mask = center_loss_idx[len(train_mask) + len(val_mask) : ]
    print("Center loss labels:", Counter(center_loss_labels))
    
    return center_loss_labels, train_mask, val_mask, test_mask
