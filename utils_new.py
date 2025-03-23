import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy import interp
from numpy import interp
from sklearn import metrics
import torch
import torch.nn as nn
import dgl
import scipy.sparse as sp
# from torch_geometric.data import Data
# from torch_geometric.utils.convert import to_networkx
import networkx as nx



def laplacian_positional_encoding(g, pos_enc_dim):

    # Laplacian
    # adjacency_matrix(transpose, scipy_fmt="csr")
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)  #
    A_dense = g.adjacency_matrix().to_dense()
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)  #
    L = sp.eye(g.number_of_nodes()) - N * A * N  #


    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR',
                                    tol=1e-2)  #
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()  # 2708*15

    return lap_pos_enc, A_dense




def load_data_AE_32(directory, random_seed):

    all_associations = pd.read_csv(directory + '/new_adjacency_matrix.csv',
                                   names=['miRNA', 'disease', 'label'])  # 726264 * 3


    known_associations = all_associations.loc[all_associations['label'] == 1]  # 14550 * 3
    unknown_associations = all_associations.loc[all_associations['label'] == 0]  # 711714 * 3

    sample_know_associations = known_associations
    sample_know_associations.reset_index(drop=True, inplace=True)
    samples_know = sample_know_associations.values

    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed,
                                                  axis=0)  # 14550 * 3
    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)
    samples = sample_df.values

    ID = np.loadtxt(directory + '/ID2_792_64.txt')
    IM = np.loadtxt(directory + '/IM_2751_64.txt')

    return ID, IM, samples,samples_know

def build_graph(directory, random_seed, lpedim):

    ID, IM, samples,samples_know = load_data_AE_32(directory, random_seed)  #


    g = dgl.DGLGraph()

    g.add_nodes(ID.shape[0] + IM.shape[0])  #
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64)     # torch.Size([1709])
    node_type[: ID.shape[0]] = 1    #
    g.ndata['type'] = node_type     #

    d_sim = torch.zeros(g.number_of_nodes(), ID.shape[1])
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))
    g.ndata['d_sim'] = d_sim

    m_sim = torch.zeros(g.number_of_nodes(), IM.shape[1])
    m_sim[ID.shape[0]: ID.shape[0] + IM.shape[0], :] = torch.from_numpy(IM.astype('float32'))
    g.ndata['m_sim'] = m_sim
    #
    disease_ids = list(range(1, ID.shape[0] + 1))   # 1-1410
    mirna_ids = list(range(1, IM.shape[0] + 1))     # 1-1160

    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}  #
    mirna_ids_invmap = {id_: i for i, id_ in enumerate(mirna_ids)}  #

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]    #
    sample_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]  #

    g.add_edges(sample_disease_vertices, sample_mirna_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})  #
    g.add_edges(sample_mirna_vertices, sample_disease_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})


    g.readonly()    #

    g_know = dgl.DGLGraph()
    g_know.add_nodes(ID.shape[0] + IM.shape[0])
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
    node_type[: ID.shape[0]] = 1
    g_know.ndata['type'] = node_type
    d_sim = torch.zeros(g_know.number_of_nodes(), ID.shape[1])
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))
    g_know.ndata['d_sim'] = d_sim

    m_sim = torch.zeros(g.number_of_nodes(), IM.shape[1])
    m_sim[ID.shape[0]: ID.shape[0] + IM.shape[0], :] = torch.from_numpy(IM.astype('float32'))
    g_know.ndata['m_sim'] = m_sim


    sample_disease_vertices_know = [disease_ids_invmap[id_] for id_ in samples_know[:, 1]]
    sample_mirna_vertices_know = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples_know[:, 0]]

    g_know.add_edges(sample_disease_vertices_know, sample_mirna_vertices_know,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    g_know.add_edges(sample_mirna_vertices_know, sample_disease_vertices_know,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})

    lpe, A_know = laplacian_positional_encoding(g_know, lpedim)




    return g, g_know,sample_disease_vertices, sample_mirna_vertices, ID, IM, samples, lpe, A_know



def floyd_warshall_source_to_all(G, source, cutoff=None):
    if source not in G:
        raise nx.NodeNotFound("Source {} not in G".format(source))
    edges = {edge: i for i, edge in enumerate(
        G.edges())}  #
    level = 0  # the current level
    nextlevel = {source: 1}  # list of nodes to check at next level
    node_paths = {source: [source]}  # paths dictionary  (paths to key from source)
    edge_paths = {source: []}

    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in G[v]:
                if w not in node_paths:
                    node_paths[w] = node_paths[v] + [w]
                    edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
                    nextlevel[w] = 1
        level = level + 1

        if (cutoff is not None and cutoff <= level):
            break
    return node_paths, edge_paths


def all_pairs_shortest_path(G):
    paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
    node_paths = {n: paths[n][0] for n in paths}
    edge_paths = {n: paths[n][1] for n in paths}
    return node_paths, edge_paths




def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    np.savetxt('mean_fpr_' + '%s.txt' % name, mean_fpr, delimiter='\t')
    np.savetxt('mean_tpr_' + '%s.txt' % name, mean_tpr, delimiter='\t')
    plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Mean AUC: %.4f $\pm$ %.4f' % (mean_auc, auc_std))

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc='lower right')
    plt.savefig(directory + '/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.close()


def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(interp(1 - mean_recall, 1 - recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AP: %.4f' % (i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    np.savetxt('mean_recall_' + '%s.txt' % name, mean_recall, delimiter='\t')
    np.savetxt('mean_precision_' + '%s.txt' % name, mean_precision, delimiter='\t')
    plt.plot(mean_recall, mean_precision, color='BlueViolet', alpha=0.9,
             label='Mean AP: %.4f $\pm$ %.4f' % (mean_prc, prc_std))  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P-R curves')
    plt.legend(loc='lower left')
    plt.savefig(directory + '/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.close()
