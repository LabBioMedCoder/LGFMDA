from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
# from torch_geometric.utils import degree
from torch_geometric.utils import degree


# from graphormer.utils import decrease_to_max_value


def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x



class CentralityEncoding(nn.Module):
    def __init__(self, max_degree: int, node_dim: int, cpe_dim):

        super().__init__()
        # self.max_in_degree = max_in_degree  # 5
        # self.max_out_degree = max_out_degree    # 5

        self.max_degree = max_degree  # 5
        self.node_dim = node_dim
        self.cpe_dim = cpe_dim

        self.z = nn.Parameter(torch.randn((max_degree, cpe_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:


        num_nodes = x.shape[0]  # 节点数量


        return x



class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):

        super().__init__()


        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)
        self.dropout = nn.Dropout(0.5)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                ) -> torch.Tensor:

        query = self.q(query)
        key = self.k(key)
        value = self.v(value)


        a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5  # 注意力值

        a_softmax = torch.softmax(a, dim=-1)

        x = a_softmax.mm(value)

        return x



class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, lpe_dim):

        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self,
                x: torch.Tensor,
                # edge_attr: torch.Tensor,
                # b: torch.Tensor,
                # edge_paths,
                # ptr
                ) -> torch.Tensor:


        return self.linear(  # 这里的x都是特征矩阵
            torch.cat([
                attention_head(x, x, x) for attention_head in self.heads
            ], dim=-1)
        )



class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, n_heads, lpe_dim, cpe_dim):

        super().__init__()

        self.node_dim = node_dim

        self.n_heads = n_heads

        self.lpe_dim = lpe_dim
        self.cpe_dim = cpe_dim

        self.attention = GraphormerMultiHeadAttention(

            dim_in=node_dim + lpe_dim,
            dim_k=node_dim + lpe_dim,
            dim_q=node_dim + lpe_dim,
            num_heads=n_heads,
            lpe_dim=lpe_dim
        )

        self.ln_1 = nn.LayerNorm(node_dim)
        self.ln_1_lpe = nn.LayerNorm(node_dim + lpe_dim)
        self.ln_1_cpe = nn.LayerNorm(node_dim + cpe_dim)  #

        self.ln_2 = nn.LayerNorm(node_dim)
        self.ln_2_lpe = nn.LayerNorm(node_dim + lpe_dim)
        self.ln_2_cpe = nn.LayerNorm(node_dim + cpe_dim)


        self.ff = nn.Linear(node_dim, node_dim)
        self.ff_lpe = nn.Linear(node_dim + lpe_dim, node_dim + lpe_dim)
        self.ff_cpe = nn.Linear(node_dim + cpe_dim, node_dim + cpe_dim)



        self.batch_norm1_h = nn.BatchNorm1d(node_dim, track_running_stats=True, eps=1e-5,
                                            momentum=0.01)
        self.batch_norm2_h = nn.BatchNorm1d(node_dim, track_running_stats=True, eps=1e-5,
                                            momentum=0.01)

        self.batch_norm1_h = nn.BatchNorm1d(node_dim, track_running_stats=True, eps=1e-5,
                                            momentum=0.01, affine=True)
        self.batch_norm2_h = nn.BatchNorm1d(node_dim, track_running_stats=True, eps=1e-5,
                                            momentum=0.01, affine=True)


        self.deg_coef1 = nn.Parameter(torch.zeros(1709, node_dim))
        nn.init.xavier_normal_(self.deg_coef1)
        self.deg_coef2 = nn.Parameter(torch.zeros(1709, node_dim))
        nn.init.xavier_normal_(self.deg_coef2)


        self.FFN_h_layer1 = nn.Linear(node_dim, node_dim * 2)
        self.FFN_h_layer2 = nn.Linear(node_dim * 2, node_dim)

        self.fc = nn.Linear(node_dim * 2, node_dim)
        self.fc_lpe = nn.Linear((node_dim+lpe_dim) * 2, node_dim+lpe_dim)



    def forward(self,x: torch.Tensor, log_deg) -> Tuple[torch.Tensor, torch.Tensor]:

        h0 = x
        h_msa = self.attention(x)
        h_msa = h0 + h_msa  # 残差
        h_logdeg = h_msa * log_deg
        h = torch.cat((h_msa, h_logdeg), dim=1)
        # x_prime = self.ln_1(self.fc(h))
        # x_new = self.ln_2(self.ff(x_prime) + x_prime)

        x_prime = self.ln_1_lpe(self.fc_lpe(h))
        x_new = self.ln_2_lpe(self.ff_lpe(x_prime) + x_prime)

        return x_new


class CentralityEncoding_concat(nn.Module):
    def __init__(self, max_degree: int, node_dim: int, cpe_dim):
        super().__init__()

        self.max_degree = max_degree  # 5
        self.node_dim = node_dim
        self.cpe_dim = cpe_dim

        self.z = nn.Parameter(torch.randn((max_degree+1, cpe_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:

        num_nodes = x.shape[0]


        node_degree = decrease_to_max_value(degree(index=edge_index[0], num_nodes=num_nodes).long(), self.max_degree)

        x = torch.cat((x, self.z[node_degree]), dim=1)


        return x