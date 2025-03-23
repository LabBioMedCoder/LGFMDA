import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn import metrics
import scipy.sparse as sp
from utils_new import build_graph, weight_reset, plot_auc_curves, plot_prc_curves
# from model import nSGC
# from model_gt import GT
# from model_gt_fang import GT, Graphormer

from model_gt_fang_hmdd32_lpe import Graphormer     # (12.13)
# from model_gt_fang_AE_NDLS import Graphormer     # (12.14)





def Train(directory, epochs, n_classes, in_size, out_dim, dropout, slope, lr, wd, random_seed, cuda):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    context = torch.device('cpu')
    lpedim = 64
    cpedim = 64
    g, g_know, disease_vertices, mirna_vertices, ID, IM, samples, lpe, A_know = build_graph(directory,
                                                                                     random_seed,
                                                                                     lpedim)


    feats_ndls = np.loadtxt(directory + '/epsilon_NoNorm_D-0.5.txt')
    feats_ndls = torch.FloatTensor(feats_ndls)


    samples_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])   # DataFrame 29100*3
    g.to(context)

    auc_result = []
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []
    prc_result = []

    fprs = []
    tprs = []
    precisions = []
    recalls = []

    i = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    for train_idx, test_idx in kf.split(samples[:, 2]):
        i += 1
        print('Training for Fold', i)

        samples_df['train'] = 0
        samples_df['train'].iloc[train_idx] = 1

        train_tensor = torch.from_numpy(samples_df['train'].values.astype('int64'))     # 29100

        edge_data = {'train': train_tensor}

        g.edges[disease_vertices, mirna_vertices].data.update(edge_data)
        g.edges[mirna_vertices, disease_vertices].data.update(edge_data)

        train_eid = g.filter_edges(lambda edges: edges.data['train'])
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)

        eid = g_train.filter_edges(lambda edges: edges.data['label'])
        g_train_new = g_train.edge_subgraph(eid, preserve_nodes=True)


        label_train = g_train.edata['label'].unsqueeze(1)   #
        src_train, dst_train = g_train.all_edges()


        test_eid = g.filter_edges(lambda edges: edges.data['train'] == 0)
        src_test, dst_test = g.find_edges(test_eid)
        label_test = g.edges[test_eid].data['label'].unsqueeze(1)



        print('Training edges:', len(train_eid))
        print('Testing edges:', len(test_eid))
        print('lpe_dim:', lpedim)
        print('lpe的维度(1709行)', len(lpe[0]))


        model = Graphormer(
            G=g,
            num_layers=2,
            input_node_dim=in_size,
            node_dim=64,
            output_dim=out_dim,
            n_heads=4,
            lpe_dim=lpedim,
            cpe_dim=cpedim
        )

        model.apply(weight_reset)
        model.to(context)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        loss = nn.BCELoss()

        for epoch in range(epochs):
            start = time.time()

            model.train()
            with torch.autograd.set_detect_anomaly(True):
                # score_train = model(g_train, src_train, dst_train, lpe, True)
                score_train = model(g_train_new, src_train, dst_train, lpe, A_know, feats_ndls, True)

                # label_train = label_train.view(-1)
                loss_train = loss(score_train, label_train)

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                # score_val = model(g, src_test, dst_test, lpe, True)
                score_val = model(g_know, src_test, dst_test, lpe, A_know, feats_ndls, True)

                # label_test = label_test.view(-1)
                loss_val = loss(score_val, label_test)

            score_train_cpu = np.squeeze(score_train.cpu().detach().numpy())
            score_val_cpu = np.squeeze(score_val.cpu().detach().numpy())
            label_train_cpu = np.squeeze(label_train.cpu().detach().numpy())
            label_val_cpu = np.squeeze(label_test.cpu().detach().numpy())

            train_auc = metrics.roc_auc_score(label_train_cpu, score_train_cpu)
            val_auc = metrics.roc_auc_score(label_val_cpu, score_val_cpu)

            pred_val = [0 if j < 0.5 else 1 for j in score_val_cpu]
            acc_val = metrics.accuracy_score(label_val_cpu, pred_val)
            pre_val = metrics.precision_score(label_val_cpu, pred_val)
            recall_val = metrics.recall_score(label_val_cpu, pred_val)
            f1_val = metrics.f1_score(label_val_cpu, pred_val)

            end = time.time()



            if (epoch + 1) % 10 == 0:

                print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(),
                      'Val Loss: %.4f' % loss_val.cpu().detach().numpy(),
                      'Acc: %.4f' % acc_val, 'Pre: %.4f' % pre_val, 'Recall: %.4f' % recall_val, 'F1: %.4f' % f1_val,
                      'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc, 'Time: %.2f' % (end - start))

        model.eval()
        with torch.no_grad():

            score_test = model(g_know, src_test, dst_test, lpe, A_know, feats_ndls, True)
            # score_test = model(g, src_test, dst_test, True)

        score_test_cpu = np.squeeze(score_test.cpu().detach().numpy())
        label_test_cpu = np.squeeze(label_test.cpu().detach().numpy())

        fpr, tpr, thresholds = metrics.roc_curve(label_test_cpu, score_test_cpu)
        precision, recall, _ = metrics.precision_recall_curve(label_test_cpu, score_test_cpu)
        test_auc = metrics.auc(fpr, tpr)
        test_prc = metrics.auc(recall, precision)

        pred_test = [0 if j < 0.5 else 1 for j in score_test_cpu]
        acc_test = metrics.accuracy_score(label_test_cpu, pred_test)
        pre_test = metrics.precision_score(label_test_cpu, pred_test)
        recall_test = metrics.recall_score(label_test_cpu, pred_test)
        f1_test = metrics.f1_score(label_test_cpu, pred_test)

        print('Fold: ', i, 'Test acc: %.4f' % acc_test, 'Pre: %.4f' % pre_test,
              'Recall: %.4f' % recall_test, 'F1: %.4f' % f1_test, 'PRC: %.4f' % test_prc,
              'AUC: %.4f' % test_auc)



        auc_result.append(test_auc)
        acc_result.append(acc_test)
        pre_result.append(pre_test)
        recall_result.append(recall_test)
        f1_result.append(f1_test)
        prc_result.append(test_prc)

        fprs.append(fpr)
        tprs.append(tpr)
        precisions.append(precision)
        recalls.append(recall)
        filepath1 = 'fprs.txt'
        filepath2 = 'tprs.txt'
        filepath3 = 'precisions.txt'
        filepath4 = 'recalls.txt'
    print('OK !')
    print('-----------------------------------------------------------------------------------------------')
    print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
          'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))
    return fprs, tprs, auc_result, precisions, recalls, prc_result


import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    fprs, tprs, auc, precisions, recalls, prc = Train(directory='data_new3.2',
                                                      # directory='new_data',
                                                      epochs=450,
                                                      # epochs=5,
                                                      n_classes=64,
                                                      # in_size=64,
                                                      in_size=64,
                                                      # out_dim=64,
                                                      out_dim=64,
                                                      dropout=0.5,
                                                      slope=0.2,
                                                      lr=0.001,
                                                      wd=5e-3,
                                                      random_seed=1225,
                                                      cuda=True)
    #plot_auc_curves(fprs, tprs, auc, directory='roc_result', name='test_auc_hmdd32_64dlpe')
    #plot_prc_curves(precisions, recalls, prc, directory='roc_result', name='test_prc_newdata_hmdd32_64dlpe')

