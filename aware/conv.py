# Pytorch 
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, BatchNorm, LayerNorm, GraphNorm
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F

# General
import numpy as np
import math


class PCTConv(nn.Module):
    def __init__(self, in_channels, num_ppi_relations, num_metagraph_relations, ppi_data, out_channels, sem_att_channels, norm=None, node_heads=3, tissue_update=100):
        super().__init__()
        
        self.ppi_data = ppi_data
        self.in_channels = in_channels
        self.num_ppi_relations = num_ppi_relations
        self.num_metagraph_relations = num_metagraph_relations
        self.out_channels = out_channels
        self.sem_att_channels = sem_att_channels
        self.node_heads = node_heads
        self.tissue_update = tissue_update
        
        # 1. Cell-type specific PPI layers edge type attention
        self.ppi_node_gats = torch.nn.ModuleList()
        for _ in range(num_ppi_relations):
            self.ppi_node_gats.append(GATConv(in_channels, out_channels, node_heads))

        # 2. Cell-type specific PPI weights
        self.ppi_w = torch.nn.ModuleList()
        for celltype, ppi in ppi_data.items():
            self.ppi_w.append(GATConv(in_channels, out_channels, node_heads))
        
        if norm == 'layernorm':
            self.norm = LayerNorm(out_channels * node_heads)
        elif norm == 'batchnorm':
            self.norm = BatchNorm(out_channels * node_heads)
        elif norm == 'graphnorm':
            self.norm = GraphNorm(out_channels * node_heads)
        else:
            self.norm = None
        
        # 3. Metagraph edge type attention
        self.metagraph_node_gats = torch.nn.ModuleList()
        for _ in range(num_metagraph_relations):
            self.metagraph_node_gats.append(GATConv(out_channels * node_heads, out_channels, node_heads))

        self.W = nn.Parameter(torch.Tensor(1, 1, out_channels * node_heads, sem_att_channels))
        self.b = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))
        self.q = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W)
        zeros(self.b)
        glorot(self.q)

    def _per_data_forward(self, x, edgetypes, node_gats):

        # Calculate node-level attention representations
        out = [gat(x, edgetype) for gat, edgetype in zip(node_gats, edgetypes) if edgetype.shape[1] > 0]
        out = torch.stack(out, dim=1).to(x.device)
    
        # Apply non-linearity
        out = F.relu(out)

        # Aggregate node-level representation using semantic level attention        
        w = torch.sum(self.W * out.unsqueeze(-1), dim=-2) + self.b
        w = torch.tanh(w)
        beta = torch.sum(self.q * w, dim=-1)
        beta = torch.softmax(beta, dim=1)
        z = torch.sum(out * beta.unsqueeze(-1), dim=1)
        return z

    def forward(self, ppi_x, metagraph_x, ppi_edgetypes, metagraph_edgetypes, ppi_edge_index, metagraph_edge_index, tissue_neighbors, init_cci=False):
        
        ########################################
        # Update PPI layers
        ########################################

        if init_cci: # Initialize CCI embeddings using PPI embeddings
            metagraph_x_list = []
            for celltype, x in ppi_x.items(): # Iterate through cell-type specific PPI layers
                
                # Update cell type's PPI embeddings using edge type attention
                ppi_x[celltype] = self._per_data_forward(x, ppi_edgetypes[celltype], self.ppi_node_gats)
                
                metagraph_x_list.append(torch.mean(ppi_x[celltype], 0))
            
            # Concatenate initialized metagraph embeddings
            metagraph_x = torch.stack(metagraph_x_list)
            metagraph_x = torch.cat((metagraph_x, torch.zeros(len(tissue_neighbors), metagraph_x.shape[1]).long().to(metagraph_x.device)))

        ########################################
        # Update metagraph embeddings
        ########################################

        else:
            for celltype, x in ppi_x.items(): # Iterate through cell-type specific PPI layers
                if self.norm is not None:
                    att_ppi = self.norm(x + self.ppi_w[celltype](x, ppi_edgetypes[celltype][0]))
                else:
                    att_ppi = x + self.ppi_w[celltype](x, ppi_edgetypes[celltype][0])
                metagraph_x[celltype, :] += torch.mean(att_ppi, 0)

        # Initialize (or update) tissue embeddings using cell-type embeddings
        for i in range(self.tissue_update):
            for t in sorted(tissue_neighbors):
                metagraph_x[t, :] = torch.mean(metagraph_x[tissue_neighbors[t]], 0)

        # Update metagraph embeddings using edge type attention
        metagraph_x = self._per_data_forward(metagraph_x, metagraph_edgetypes, self.metagraph_node_gats)
        
        return ppi_x, metagraph_x


class PPIConv(nn.Module):
    def __init__(self, in_channels, num_ppi_relations, out_channels, ppi_data, sem_att_channels, node_heads=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_ppi_relations = num_ppi_relations
        self.out_channels = out_channels
        self.sem_att_channels = sem_att_channels
        self.node_heads = node_heads

        self.ppi_node_gats = torch.nn.ModuleList()
        for _ in range(num_ppi_relations):
            self.ppi_node_gats.append(GATConv(in_channels, out_channels, node_heads))

        self.W = nn.Parameter(torch.Tensor(1, 1, out_channels * node_heads, sem_att_channels))
        self.b = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))
        self.q = nn.Parameter(torch.Tensor(1, 1, sem_att_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W)
        zeros(self.b)
        glorot(self.q)

    def _per_data_forward(self, x, edgetypes, node_gats):

        # Calculate node-level attention representations
        out = [gat(x, edgetype) for gat, edgetype in zip(node_gats, edgetypes)]
        out = torch.stack(out, dim=1).to(x.device)
        
        # Apply nonlinear layer
        out = F.relu(out)

        # Aggregate node-level representation using semantic level attention
        w = torch.sum(self.W * out.unsqueeze(-1), dim=-2) + self.b
        w = torch.tanh(w)
        beta = torch.sum(self.q * w, dim=-1)
        beta = torch.softmax(beta, dim=1)
        z = torch.sum(out * beta.unsqueeze(-1), dim=1)

        return z

    def forward(self, ppi_x, ppi_edgetypes, metagraph_x, ppi_w, metagraph):
        
        ########################################
        # Update PPI layers
        ########################################
        
        for celltype, x in ppi_x.items(): # Iterate through cell-type specific PPI layers

            if len(ppi_edgetypes[celltype]) == 0: ppi_x[celltype] = []
            else:
                # Update using edge type attention
                ppi_x[celltype] = self._per_data_forward(x, ppi_edgetypes[celltype], self.ppi_node_gats)

            # Downpool using cell-type embedding
            gamma = (ppi_w[celltype].att_src + ppi_w[celltype].att_dst).flatten(0)  # Different PyG versions use different notations, att_r=att_src, att_l=att_dst.
            ppi_x[celltype] += (metagraph_x[celltype, :] * gamma.to(x.device))

        return ppi_x