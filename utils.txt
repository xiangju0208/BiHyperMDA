import torch as t
import torch.nn.functional as F
import numpy as np


def get_co_occurrence(sim_or_association, args):
    # 进行降噪
    row_quantile = t.quantile(sim_or_association, args.quantile, dim=1, interpolation='nearest', keepdim=True)
    sim_or_association = t.where(sim_or_association >= row_quantile, 1.0, 0.0)
    sim_square = sim_or_association @ sim_or_association
    return sim_square


def get_hypergraph_feat(all_hyper_edge_feat_list):
    hyper_graph_feat_list = []
    for hyper_edge_feat in all_hyper_edge_feat_list:
        hyper_edge_feat_1 = t.max(hyper_edge_feat, dim=0).values
        # hyper_edge_feat_2 = t.mean(hyper_edge_feat, dim=0)
        # hyper_edge_feat = hyper_edge_feat_1 + hyper_edge_feat_2
        hyper_edge_feat = hyper_edge_feat_1
        hyper_graph_feat_list.append(hyper_edge_feat)
    return t.vstack(hyper_graph_feat_list)


def get_all_hyper_edge_feat_list(hypergraph, feat_matrix):
    all_hyper_edge_feat_list = []
    row_index, col_index = t.where(hypergraph)
    unique_rows, counts = t.unique_consecutive(row_index, return_counts=True)
    count = 0
    for cou in counts:
        col_index_slice = col_index[count:count + cou]
        result = feat_matrix[col_index_slice]
        all_hyper_edge_feat_list.append(result)
        count = count + cou
    return all_hyper_edge_feat_list


def get_all_hyper_edge_feat_list_1(hypergraph, feat_matrix):
    all_hyper_edge_feat_list = []
    for i in range(hypergraph.shape[0]):
        result = hypergraph[i, :].unsqueeze(1) * feat_matrix
        mask = t.any(result, dim=1)
        all_hyper_edge_feat_list.append(result[mask])
    return all_hyper_edge_feat_list


def get_edge_index(sim):
    return t.nonzero(sim).T


def new_miRNA_or_disease_experiment_sample(md_copy, test_index):
    pos_samples_index = get_edge_index(md_copy)
    pos_num = pos_samples_index.shape[1]
    pos_samples_index_shuffle = pos_samples_index[:, t.randperm(pos_num)]
    add = t.zeros_like(md_copy)
    add[test_index] = 1
    md_copy = md_copy + add
    neg_samples_index = t.nonzero(md_copy == 0).T
    neg_samples_index_shuffle = neg_samples_index[:, t.randperm(neg_samples_index.shape[1])]
    neg_samples_index_shuffle = neg_samples_index_shuffle[:, :pos_num]
    train_labels = t.hstack(
        (t.ones(pos_samples_index_shuffle.shape[1]), t.zeros(neg_samples_index_shuffle.shape[1]))).to(md_copy.device)
    train_samples = t.hstack((pos_samples_index_shuffle, neg_samples_index_shuffle)).to(md_copy.device)
    return train_samples, train_labels