import numpy as np
import os
import torch as t


def get_miRNA_disease_kernel(args):
    path = args.path
    device = args.device
    kernel_matrix_edge_index = {}

    "miRNA functional sim"
    m_func_sim = np.loadtxt(os.path.join(path, 'm_fs.csv'), delimiter=',', dtype=float)
    m_func_sim = t.tensor(m_func_sim, device=device).to(t.float32)

    "miRNA sequence sim"
    m_seq_sim = np.loadtxt(os.path.join(path, 'm_ss.csv'), delimiter=',', dtype=float)
    m_seq_sim = t.tensor(m_seq_sim, device=device).to(t.float32)

    "miRNA GIP sim"
    m_gip_sim = np.loadtxt(os.path.join(path, 'm_gs.csv'), delimiter=',', dtype=float)
    m_gip_sim = t.tensor(m_gip_sim, device=device).to(t.float32)

    "disease functional sim"
    d_func_sim = np.loadtxt(os.path.join(path, 'd_fs.csv'), delimiter=',', dtype=float)
    d_func_sim = t.tensor(d_func_sim, device=device).to(t.float32)

    "disease semantic sim"
    d_sem_sim = np.loadtxt(os.path.join(path, 'd_ss.csv'), delimiter=',', dtype=float)
    d_sem_sim = t.tensor(d_sem_sim, device=device).to(t.float32)

    "disease GIP sim"
    d_gip_sim = np.loadtxt(os.path.join(path, 'd_gs.csv'), delimiter=',', dtype=float)
    d_gip_sim = t.tensor(d_gip_sim, device=device).to(t.float32)

    "miRNA associate disease"
    m_d = np.loadtxt(os.path.join(path, 'm_d.csv'), delimiter=',', dtype=float)
    m_d = t.tensor(m_d, device=device).to(t.float32)

    kernel_matrix_edge_index['m_d'] = m_d
    kernel_matrix_edge_index['m_f'] = m_func_sim
    kernel_matrix_edge_index['m_s'] = m_seq_sim
    kernel_matrix_edge_index['m_g'] = m_gip_sim
    kernel_matrix_edge_index['d_f'] = d_func_sim
    kernel_matrix_edge_index['d_s'] = d_sem_sim
    kernel_matrix_edge_index['d_g'] = d_gip_sim

    return kernel_matrix_edge_index


def get_positive_negative_samples(kernel_matrix_edge_index):
    m_d = kernel_matrix_edge_index['m_d']
    t.manual_seed(42)
    pos_samples_index = t.where(m_d == 1)
    n = pos_samples_index[0].shape[0]
    pos_samples_index = t.stack(pos_samples_index)
    pos_samples_index_shuffled = pos_samples_index[:, t.randperm(n)]

    neg_samples_index = t.where(m_d == 0)
    n = neg_samples_index[0].shape[0]
    neg_samples_index = t.stack(neg_samples_index)
    neg_samples_index_shuffled = neg_samples_index[:, t.randperm(n)]
    return pos_samples_index_shuffled, neg_samples_index_shuffled