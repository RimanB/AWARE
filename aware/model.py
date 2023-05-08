# Pytorch 
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
import torch.nn.functional as F

# General
import numpy as np
import math

# Single cell resolution embeddings
from conv import PCTConv, PPIConv

# Debugging flags
metapath_att = True # Default: True
downpool = True # Default: True
# metagraph = True # Default: True
# norm = False # Default: False
apply_ppi_att = True # Default: True


class TrainNet(nn.Module):
    def __init__(self, nfeat, output, num_ppi_relations, num_cci_bto_relations, ppi_data, n_heads, norm=None, dropout = 0.2, metagraph = True, metagraph_relw=False):
        super(TrainNet, self).__init__()

        self.ppi_data = ppi_data
        self.nfeat = nfeat 
        self.output = output 
        self.num_ppi_relations = num_ppi_relations 
        self.num_cci_bto_relations = num_cci_bto_relations 
        self.n_heads = n_heads
        self.dropout = dropout
        self.metagraph = metagraph
        self.metagraph_relw = metagraph_relw

        if self.metagraph_relw:
            self.cci_bto_relw = nn.Parameter(torch.Tensor(num_cci_bto_relations, int(output * n_heads)))
            nn.init.xavier_uniform_(self.cci_bto_relw, gain = nn.init.calculate_gain('relu'))

        # Complete layer #1 of up-/down-pooling
        self.conv1_up = PCTConv(nfeat, num_ppi_relations, num_cci_bto_relations, ppi_data, output, sem_att_channels=8, norm=norm, node_heads=n_heads)
        if downpool: self.conv1_down = PPIConv(output * n_heads, num_ppi_relations, output, ppi_data, sem_att_channels=8, node_heads=n_heads)

        # Complete layer #2 of up-/down-pooling
        self.conv2_up = PCTConv(output * n_heads, num_ppi_relations, num_cci_bto_relations, ppi_data, output, sem_att_channels=8, norm=norm, node_heads=n_heads)
        if downpool: self.conv2_down = PPIConv(output * n_heads, num_ppi_relations, output, ppi_data, sem_att_channels=8, node_heads=n_heads)

        # Normalization layer
        # self.norm = nn.BatchNorm1d(output * n_heads)


    def forward(self, ppi_x, cci_bto_x, ppi_metapaths, cci_bto_metapaths, ppi_edge_index, cci_bto_edge_index, tissue_neighbors):
        
        metagraph = self.metagraph

        ########################################
        # Complete layer #1 of up-/down-pooling
        ########################################

        # Update Protein-Celltype-Tissue
        ppi_x, cci_bto_x = self.conv1_up(ppi_x, cci_bto_x, ppi_metapaths, cci_bto_metapaths, ppi_edge_index, cci_bto_edge_index, tissue_neighbors, apply_ppi_att=apply_ppi_att, metapath_att=metapath_att, metagraph=metagraph, init_cci=True)

        # Update PPI and down-pool CCI-BTO
        if downpool:
            ppi_x = self.conv1_down(ppi_x, ppi_metapaths, cci_bto_x, self.conv1_up.ppi_w, metagraph=metagraph)

        ########################################
        # Apply ReLU, dropout, and normalize
        ########################################
        for celltype, x in ppi_x.items():
            ppi_x[celltype] = F.relu(x)
            ppi_x[celltype] = F.dropout(ppi_x[celltype], p = self.dropout, training = self.training)
            # if norm: ppi_x[celltype] = self.norm(ppi_x[celltype])
        
        if metagraph:
            cci_bto_x = F.relu(cci_bto_x)
            cci_bto_x = F.dropout(cci_bto_x, p = self.dropout, training = self.training)
            # if norm: cci_bto_x = self.norm(cci_bto_x)


        ########################################
        # Complete layer #2 of up-/down-pooling
        ########################################

        # Update Protein-Celltype-Tissue
        ppi_x, cci_bto_x = self.conv2_up(ppi_x, cci_bto_x, ppi_metapaths, cci_bto_metapaths, ppi_edge_index, cci_bto_edge_index, tissue_neighbors, apply_ppi_att=apply_ppi_att, metapath_att=metapath_att, metagraph=metagraph)

        # Update PPI and down-pool CCI-BTO
        if downpool: 
            ppi_x = self.conv2_down(ppi_x, ppi_metapaths, cci_bto_x, self.conv2_up.ppi_w, metagraph=metagraph)

        return ppi_x, cci_bto_x
