import numpy as np
import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from torch import tensor

import scipy.sparse as sp
import dgl
# from layers import GraphormerEncoderLayer
from layers_hmdd32_lpe import GraphormerEncoderLayer, CentralityEncoding, CentralityEncoding_concat
# from layers_cpe import CentralityEncoding_concat
from torch_geometric.utils import degree

from dgl.nn.pytorch import GraphConv


class Graphormer(nn.Module):
    def __init__(self,
                 G,
                 num_layers: int,
                 input_node_dim: int,
                 node_dim: int,
                 output_dim: int,
                 n_heads: int,
                 lpe_dim,
                 cpe_dim
                 ):

        super().__init__()

        self.G = G
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)
        self.num_layers = num_layers
        self.input_node_dim = input_node_dim
        self.node_dim = node_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.lpe_dim = lpe_dim
        self.cpe_dim = cpe_dim  #

        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)
        self.node_in_lin_lpe = nn.Linear(self.input_node_dim + self.lpe_dim, self.node_dim + self.lpe_dim)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout = nn.Dropout(0.5)

        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], input_node_dim, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], input_node_dim, bias=False)
        self.lfc2 = nn.Linear(output_dim * 2, output_dim)
        self.w = nn.Linear(self.node_dim, self.node_dim, bias=True)

        self.predict = nn.Linear(output_dim * 2, 1)
        
        self.predict1 = nn.Linear(output_dim * 2, output_dim)
        self.predict2 = nn.Linear(output_dim, 1)
        
        
        self.predict_lpe = nn.Linear((output_dim + lpe_dim) * 2, 1)

        self.predict_cpe = nn.Linear((output_dim + cpe_dim) * 2, 1)


        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                node_dim=self.node_dim,
                n_heads=self.n_heads,
                lpe_dim=self.lpe_dim,
                cpe_dim=self.cpe_dim
            ) for _ in range(self.num_layers)
        ])

        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

        self.conv1 = GraphConv(self.node_dim, self.node_dim, activation=F.relu)

        self.batch_norm1_h = nn.BatchNorm1d(output_dim, track_running_stats=not False, eps=1e-5,
                                            momentum=0.1)
        self.batch_norm2_h = nn.BatchNorm1d(output_dim, track_running_stats=not False, eps=1e-5,
                                            momentum=0.1)

        self.deg_coef1 = nn.Parameter(torch.zeros(1709, output_dim))
        nn.init.xavier_normal_(self.deg_coef1)
        self.deg_coef2 = nn.Parameter(torch.zeros(1709, output_dim))
        nn.init.xavier_normal_(self.deg_coef2)

        self.w_gt = nn.Linear(self.node_dim, self.node_dim, bias=True)
        self.w_ndls = nn.Linear(self.node_dim, self.node_dim, bias=True)
        self.lfc2 = nn.Linear(output_dim * 2, output_dim)
        self.lfc2_lpe = nn.Linear(output_dim * 2 + lpe_dim, output_dim)
        self.cpefc = nn.Linear(node_dim * 2, node_dim)
        self.lpefc = nn.Linear(node_dim + lpe_dim, node_dim)

    def forward(self, graph, diseases, mirnas, lpe, A_know, feats_ndls,
                training=True):

        self.G.apply_nodes(lambda nodes: {'z': nodes.data['d_sim']}, self.disease_nodes)
        self.G.apply_nodes(lambda nodes: {'z': nodes.data['m_sim']}, self.mirna_nodes)

        feats = self.G.ndata.pop('z')

        num_nodes = feats.shape[0]


        graph.add_self_loop()


        row1 = graph.edges()[0]
        row2 = graph.edges()[1]
        edge_index = torch.vstack((row1, row2))

        deg = degree(index=row1, num_nodes=num_nodes).float()
        log_deg = torch.log(deg + 1)
        log_deg = log_deg.view(num_nodes, 1)



        x = torch.cat((feats, lpe), dim=1)


        for layer in self.layers:  # transformer
            x = layer(x, log_deg)


        x_last = torch.cat((x, feats_ndls), dim=1)
        x = F.relu(self.lfc2_lpe(x_last))


        h = x
        h_diseases = h[diseases]
        h_mirnas = h[mirnas]
        h_concat = torch.cat((h_diseases, h_mirnas), 1)


        predict = self.dropout(F.relu(self.predict1(h_concat)))
        predict_score = torch.sigmoid(self.predict2(predict))
        
        
        
        return predict_score


def laplacian_positional_encoding(g, pos_enc_dim):  #


    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A_dense = g.adjacency_matrix().to_dense()
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy

    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR',
                                    tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()]
    lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()  # 2708*15

    return lap_pos_enc, A_dense

def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x
