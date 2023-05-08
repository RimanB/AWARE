import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
import torch.nn.functional as F

# AWARE
from conv import PCTConv, PPIConv


class AWARE(nn.Module):
    def __init__(self, nfeat, output, num_ppi_relations, num_mg_relations, ppi_data, n_heads, norm, dropout):
        super(AWARE, self).__init__()

        self.ppi_data = ppi_data
        self.nfeat = nfeat 
        self.output = output 
        self.num_ppi_relations = num_ppi_relations 
        self.num_mg_relations = num_mg_relations 
        self.n_heads = n_heads
        self.dropout = dropout

        # Decoder for metagraph
        self.metagraph_relw = nn.Parameter(torch.Tensor(num_mg_relations, int(output * n_heads)))
        nn.init.xavier_uniform_(self.metagraph_relw, gain = nn.init.calculate_gain('relu'))

        # Complete layer #1 of up-/down-pooling
        self.conv1_pct = PCTConv(nfeat, num_ppi_relations, num_mg_relations, ppi_data, output, sem_att_channels = 8, norm = norm, node_heads = n_heads)
        self.conv1_ppi = PPIConv(output * n_heads, num_ppi_relations, output, ppi_data, sem_att_channels = 8, node_heads = n_heads)

        # Complete layer #2 of up-/down-pooling
        self.conv2_pct = PCTConv(output * n_heads, num_ppi_relations, num_mg_relations, ppi_data, output, sem_att_channels = 8, norm = norm, node_heads = n_heads)
        self.conv2_ppi = PPIConv(output * n_heads, num_ppi_relations, output, ppi_data, sem_att_channels = 8, node_heads = n_heads)

    def forward(self, ppi_x, metagraph_x, ppi_edgetypes, mg_edgetypes, ppi_edge_index, mg_edge_index, tissue_neighbors):
        
        ########################################
        # Complete layer #1 of up-/down-pooling
        ########################################

        # Update Protein-Celltype-Tissue
        ppi_x, metagraph_x = self.conv1_pct(ppi_x, metagraph_x, ppi_edgetypes, mg_edgetypes, ppi_edge_index, mg_edge_index, tissue_neighbors, init_cci = True)

        # Update PPI and down-pool metagraph
        ppi_x = self.conv1_ppi(ppi_x, ppi_edgetypes, metagraph_x, self.conv1_pct.ppi_w)

        ########################################
        # Apply ReLU and dropout
        ########################################
        for celltype, x in ppi_x.items():
            ppi_x[celltype] = F.relu(x)
            ppi_x[celltype] = F.dropout(ppi_x[celltype], p = self.dropout, training = self.training)
        
        metagraph_x = F.relu(metagraph_x)
        metagraph_x = F.dropout(metagraph_x, p = self.dropout, training = self.training)

        ########################################
        # Complete layer #2 of up-/down-pooling
        ########################################

        # Update Protein-Celltype-Tissue
        ppi_x, metagraph_x = self.conv2_pct(ppi_x, metagraph_x, ppi_edgetypes, mg_edgetypes, ppi_edge_index, mg_edge_index, tissue_neighbors)

        # Update PPI and down-pool metagraph
        ppi_x = self.conv2_ppi(ppi_x, ppi_edgetypes, metagraph_x, self.conv2_pct.ppi_w)

        return ppi_x, metagraph_x
