import math

import torch
import torch.nn as nn
from utils import *
from torch_geometric.nn import GCNConv, GATv2Conv

class construct_hypergraph(nn.Module):
    def __init__(self, args):
        super(construct_hypergraph, self).__init__()
        self.top_k = args.k1
        self.sim_hypergraph = construct_hypergraph_with_sim_or_asso(self.top_k)
        self.co_occurrence_hypergraph = construct_hypergraph_with_co_occurrence(args)

    def forward(self, sim_or_association):
        sim_hypergraph = self.sim_hypergraph(sim_or_association)
        co_occurrence_hypergraph = self.co_occurrence_hypergraph(sim_or_association)
        return sim_hypergraph, co_occurrence_hypergraph


class construct_hypergraph_with_sim_or_asso(nn.Module):
    def __init__(self, top_k):
        super(construct_hypergraph_with_sim_or_asso, self).__init__()
        self.top_k = top_k

    # 需要检查数量k是否合理
    def forward(self, sim_or_association):
        hyper_graph = t.zeros(sim_or_association.shape[0], sim_or_association.shape[1],
                              device=sim_or_association.device)
        if self.top_k > 591 and sim_or_association.shape[0] == 591:
            self.top_k = 591
        src = t.ones(sim_or_association.shape[0], self.top_k, device=sim_or_association.device)
        _, index = t.topk(sim_or_association, self.top_k)
        hyper_graph = t.scatter(hyper_graph, dim=1, index=index, src=src)
        # hyper_graph = hyper_graph * sim_or_association
        return hyper_graph


class construct_hypergraph_with_co_occurrence(nn.Module):
    def __init__(self, args):
        super(construct_hypergraph_with_co_occurrence, self).__init__()
        self.args = args

    def forward(self, sim_or_association):
        hyper_graph = t.zeros(sim_or_association.shape[0], sim_or_association.shape[1],
                              device=sim_or_association.device)
        if self.args.k > 591 and sim_or_association.shape[0] == 591:
            self.args.k = 591
        src = t.ones(sim_or_association.shape[0], self.args.k, device=sim_or_association.device)
        sim_square_or_asso = get_co_occurrence(sim_or_association, self.args)
        _, index = t.topk(sim_square_or_asso, self.args.k)
        hyper_graph = t.scatter(hyper_graph, dim=1, index=index, src=src)
        return hyper_graph


class hypergraph_conv(nn.Module):
    def __init__(self, args):
        super(hypergraph_conv, self).__init__()

        self.m_sim_gcn = nn.ModuleList()
        self.d_sim_gcn = nn.ModuleList()
        self.relu = nn.ModuleList()

        self.layer_relu = nn.ModuleList()
        self.m_layer_gcn = nn.ModuleList()
        self.d_layer_gcn = nn.ModuleList()

        for _ in range(args.layer):
            self.m_layer_gcn.append(hypergraph_GCN_conv_2(args))
            self.d_layer_gcn.append(hypergraph_GCN_conv_2(args))
            self.layer_relu.append(nn.ReLU())

        for _ in range(args.hyper_gcn_layer2):
            self.m_sim_gcn.append(hypergraph_GCN_conv_3(args, 2 * 853))
            self.d_sim_gcn.append(hypergraph_GCN_conv_3(args, 2 * 591))
            self.relu.append(nn.ReLU())

    def forward(self,
                m_hg=None,
                m_feat_list=None,
                d_hg=None, d_feat_list=None):
        m_hierarchical_mes_passing_list = []
        d_hierarchical_mes_passing_list = []
        m_hypergraph_feat_list = [m_feat_list.copy(), m_feat_list.copy()]
        d_hypergraph_feat_list = [d_feat_list.copy(), d_feat_list.copy()]

        m_some_view_feat = get_hypergraph_feat(
            get_all_hyper_edge_feat_list(m_hg, m_hypergraph_feat_list[0][0]))

        d_some_view_feat = get_hypergraph_feat(
            get_all_hyper_edge_feat_list(d_hg, d_hypergraph_feat_list[0][0]))

        m_hierarchical_mes_passing_list.append(m_some_view_feat)
        d_hierarchical_mes_passing_list.append(d_some_view_feat)

        m_hierarchical_mes_passing_list_copy = m_hierarchical_mes_passing_list.copy()
        d_hierarchical_mes_passing_list_copy = d_hierarchical_mes_passing_list.copy()

        for count, (m_multi_view_gcn, d_multi_view_gcn, relu) in enumerate(zip(self.m_sim_gcn, self.d_sim_gcn,
                                                                               self.relu)):
            m_some_view_feat = m_multi_view_gcn(m_hg.T, m_hypergraph_feat_list[1][count]) + \
                               m_hypergraph_feat_list[1][count]
            m_some_view_feat = relu(m_some_view_feat)
            d_some_view_feat = d_multi_view_gcn(d_hg.T, d_hypergraph_feat_list[1][count]) + \
                               d_hypergraph_feat_list[1][count]
            d_some_view_feat = relu(d_some_view_feat)
            m_hypergraph_feat_list[1].append(m_some_view_feat)
            d_hypergraph_feat_list[1].append(d_some_view_feat)

        count = 0
        for m_multi_view_gcn, d_multi_view_gcn, relu in zip(self.m_layer_gcn, self.d_layer_gcn,
                                                            self.layer_relu):
            m_some_view_feat = m_multi_view_gcn(m_hg,
                                                m_hierarchical_mes_passing_list_copy[count]) + \
                               m_hierarchical_mes_passing_list_copy[count]
            m_some_view_feat = relu(m_some_view_feat)
            d_some_view_feat = d_multi_view_gcn(d_hg,
                                                d_hierarchical_mes_passing_list_copy[count]) + \
                               d_hierarchical_mes_passing_list_copy[count]
            d_some_view_feat = relu(d_some_view_feat)
            m_hierarchical_mes_passing_list_copy.append(m_some_view_feat)
            d_hierarchical_mes_passing_list_copy.append(d_some_view_feat)
            count = count + 1

        return m_hypergraph_feat_list[1][-1], d_hypergraph_feat_list[1][-1], m_hierarchical_mes_passing_list_copy[-1], \
               d_hierarchical_mes_passing_list_copy[-1]


class hypergraph_GCN_conv_2(nn.Module):
    def __init__(self, args):
        super(hypergraph_GCN_conv_2, self).__init__()
        self.W = nn.Linear(args.proj, args.proj)

    def forward(self, hypergraph, feat):
        D_e = t.sum(hypergraph, dim=1)
        D_v = t.sum(hypergraph, dim=0)
        D_e_1 = t.sqrt(t.diag(1 / D_e))
        D_v_1 = t.diag(1 / D_v)
        D_v_1 = t.where(t.isinf(D_v_1), 1, D_v_1)
        feat = self.W(D_e_1 @ hypergraph @ D_v_1 @ hypergraph.T @ D_e_1 @ feat)
        return feat


class hypergraph_GCN_conv_3(nn.Module):
    def __init__(self, args, num):
        super(hypergraph_GCN_conv_3, self).__init__()
        self.W = nn.Linear(args.proj, args.proj)
        self.W1 = nn.Linear(num, num, bias=False)

    def forward(self, hypergraph_transpose, feat):
        D_e = t.sum(hypergraph_transpose, dim=0)
        D_v = t.sum(hypergraph_transpose, dim=1)
        D_e_1 = t.diag(1 / D_e)
        D_v_1 = t.sqrt(t.diag(1 / D_v))
        feat = self.W(self.W1(D_v_1 @ hypergraph_transpose) @ D_e_1 @ hypergraph_transpose.T @ D_v_1 @ feat)
        return feat


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.MLP1 = nn.Linear(2 * args.proj, args.proj)
        self.MLP2 = nn.Linear(args.proj, args.proj // 2)
        self.MLP3 = nn.Linear(args.proj // 2, args.proj // 4)
        self.MLP4 = nn.Linear(args.proj // 4, args.proj // 8)
        self.MLP5 = nn.Linear(args.proj // 8, 1)

    def forward(self, sample_pairs):
        result = self.MLP1(sample_pairs)
        result = self.MLP2(result)
        result = self.MLP3(result)
        if result.shape[1] == 1:
            return result
        result = self.MLP4(result)
        if result.shape[1] == 1:
            return result
        return self.MLP5(result)


class Model(nn.Module):
    def __init__(self, kernel_matrix_edge_index, args, m_d_copy):
        super(Model, self).__init__()
        self.m_f = kernel_matrix_edge_index['m_f']
        self.m_s = kernel_matrix_edge_index['m_s']
        self.m_g = kernel_matrix_edge_index['m_g']
        self.d_f = kernel_matrix_edge_index['d_f']
        self.d_s = kernel_matrix_edge_index['d_s']
        self.d_g = kernel_matrix_edge_index['d_g']

        self.m_kernel_list = [self.m_f, self.m_s, self.m_g]
        self.d_kernel_list = [self.d_f, self.d_s, self.d_g]

        self.m_sim = t.stack(self.m_kernel_list).mean(dim=0)
        self.d_sim = t.stack(self.d_kernel_list).mean(dim=0)

        self.m_proj = nn.Linear(self.m_f.shape[1], args.proj)
        self.d_proj = nn.Linear(self.d_f.shape[1], args.proj)

        self.m_f_m_d = self.m_f @ m_d_copy
        self.m_s_m_d = self.m_s @ m_d_copy
        self.m_g_m_d = self.m_g @ m_d_copy

        self.m_f_degree = 1 / t.sum(self.m_f, dim=1)
        self.m_s_degree = 1 / t.sum(self.m_s, dim=1)
        self.m_g_degree = 1 / t.sum(self.m_g, dim=1)

        self.A_m_f = self.m_f_degree.unsqueeze(1) * self.m_f_m_d
        self.A_m_s = self.m_s_degree.unsqueeze(1) * self.m_s_m_d
        self.A_m_g = self.m_g_degree.unsqueeze(1) * self.m_g_m_d

        self.d_f_m_d = m_d_copy @ self.d_f
        self.d_s_m_d = m_d_copy @ self.d_s
        self.d_g_m_d = m_d_copy @ self.d_g

        self.d_f_degree = 1 / t.sum(self.d_f, dim=1)
        self.d_s_degree = 1 / t.sum(self.d_s, dim=1)
        self.d_g_degree = 1 / t.sum(self.d_g, dim=1)

        self.A_d_f = self.d_f_m_d * self.d_f_degree.unsqueeze(0)
        self.A_d_s = self.d_s_m_d * self.d_s_degree.unsqueeze(0)
        self.A_d_g = self.d_g_m_d * self.d_g_degree.unsqueeze(0)

        self.m_A_list = [self.A_m_f, self.A_m_s, self.A_m_g]
        self.d_A_list = [self.A_d_f, self.A_d_s, self.A_d_g]

        self.construct_hypergraph = construct_hypergraph(args)

        self.hypergraph_conv = hypergraph_conv(args)

        self.m_edge_index = []
        self.d_edge_index = []

        self.m_edge_index.append(get_edge_index(self.m_sim))
        self.d_edge_index.append(get_edge_index(self.d_sim))

        self.MLP = MLP(args)
        self.mm = nn.Linear(args.proj * 2, args.proj)
        self.dd = nn.Linear(args.proj * 2, args.proj)

    def forward(self, samples):
        m_feat_list = []
        d_feat_list = []

        m_feat = self.m_proj(self.m_sim)
        m_feat_list.append(m_feat)
        sim_hg, co_hg = self.construct_hypergraph(self.m_sim)
        # m_hg = (sim_hg.bool() | co_hg.bool()).float()
        m_hg = t.vstack((sim_hg, co_hg))

        d_feat = self.d_proj(self.d_sim)
        d_feat_list.append(d_feat)
        sim_hg, co_hg = self.construct_hypergraph(self.d_sim)

        d_hg = t.vstack((sim_hg, co_hg))


        final_m_feat3, final_d_feat3, m_passing_feat, d_passing_feat = self.hypergraph_conv(
            m_hg=m_hg,
            m_feat_list=m_feat_list.copy(),
            d_hg=d_hg,
            d_feat_list=d_feat_list.copy())

        m_passing_feat_split_1 = m_passing_feat[:853]
        m_passing_feat_split_2 = m_passing_feat[853:]
        d_passing_feat_split_1 = d_passing_feat[:591]
        d_passing_feat_split_2 = d_passing_feat[591:]
        m_passing_feat = t.max(t.stack((m_passing_feat_split_1, m_passing_feat_split_2)), dim=0).values
        d_passing_feat = t.max(t.stack((d_passing_feat_split_1, d_passing_feat_split_2)), dim=0).values
        
        final_m_feat = self.mm(t.hstack(
            (m_passing_feat, final_m_feat3)))
        final_d_feat = self.dd(t.hstack(
            (d_passing_feat, final_d_feat3)))

        sample_pairs = t.hstack((final_m_feat[samples[0]], final_d_feat[samples[1]]))

        
        return 0.1 * t.max(t.stack((self.m_A_list[0][samples[0], samples[1]].unsqueeze(1),
                                    self.m_A_list[1][samples[0], samples[1]].unsqueeze(1),
                                    self.m_A_list[2][samples[0], samples[1]].unsqueeze(1),
                                    self.d_A_list[0][samples[0], samples[1]].unsqueeze(1),
                                    self.d_A_list[1][samples[0], samples[1]].unsqueeze(1),
                                    self.d_A_list[2][samples[0], samples[1]].unsqueeze(1))),
                           dim=0).values + 0.9 * t.sigmoid(
            self.MLP(sample_pairs))