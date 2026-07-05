import gc

from model import *
from getdata import *
from sklearn.model_selection import KFold
from sklearn import metrics
from param import *
import pandas as pd

def train_valid_test(args):
    kernel_matrix_edge_index = get_miRNA_disease_kernel(args)
    pos_samples_index_shuffled, neg_samples_index_shuffled = get_positive_negative_samples(kernel_matrix_edge_index)
    neg_samples_index_shuffled = neg_samples_index_shuffled[:, :pos_samples_index_shuffled.shape[1]]
    pos_samples_index_shuffled = pos_samples_index_shuffled.T
    neg_samples_index_shuffled = neg_samples_index_shuffled.T
    t.manual_seed(42)
    k_fold = KFold(n_splits=args.kfolds, shuffle=True, random_state=42)
    train_idx = []
    test_idx = []
    limit = [512]
    for train_index, test_index in k_fold.split(pos_samples_index_shuffled):
        train_idx.append(train_index)
        test_idx.append(test_index)
    for lim in limit:
        auc_list = []
        pre_list = []
        acc_list = []
        f1_list = []
        recall_list = []
        aupr_list = []
        args.proj = lim
        for i in range(args.kfolds):

            train_pos_samples_80_percent = pos_samples_index_shuffled[train_idx[i]]
            train_neg_samples_80_percent = neg_samples_index_shuffled[train_idx[i]]

            train_pos_samples_80_percent = train_pos_samples_80_percent.T
            train_neg_samples_80_percent = train_neg_samples_80_percent.T

            test_pos_samples_20_percent = pos_samples_index_shuffled[test_idx[i]]
            test_neg_samples_20_percent = neg_samples_index_shuffled[test_idx[i]]

            test_pos_samples_20_percent = test_pos_samples_20_percent.T
            test_neg_samples_20_percent = test_neg_samples_20_percent.T

            train_samples = t.hstack((train_pos_samples_80_percent, train_neg_samples_80_percent)).to(args.device)
            test_samples = t.hstack((test_pos_samples_20_percent, test_neg_samples_20_percent)).to(args.device)
            train_labels = t.hstack(
                (t.ones(train_pos_samples_80_percent.shape[1]), t.zeros(train_neg_samples_80_percent.shape[1]))).to(
                args.device)
            test_labels = t.hstack(
                (t.ones(test_pos_samples_20_percent.shape[1]), t.zeros(test_neg_samples_20_percent.shape[1]))).to(
                args.device)

            m_d = kernel_matrix_edge_index['m_d']
            m_d_copy = m_d.clone()
            m_d_copy[test_pos_samples_20_percent[0], test_pos_samples_20_percent[1]] = 0

            print(
                f'######################  Fold {i + 1} of {args.kfolds}##########################')

            model = Model(kernel_matrix_edge_index, args, m_d_copy).to(args.device)
            optimizer = t.optim.AdamW(params=model.parameters(), weight_decay=1e-4, lr=args.lr)
            loss_fn = t.nn.BCELoss()
            for l in range(args.epoch):
                model.train()
                optimizer.zero_grad()
                result = model(train_samples)
                loss = loss_fn(t.flatten(result), train_labels)
                loss.backward()
                optimizer.step()
                print('epoch {:03d} train_loss {:.8f}  '.format(
                    l + 1, loss.item()))
                gc.collect()
            model.eval()
            with t.no_grad():
                y_pre = model(test_samples)
                y_pre = t.flatten(y_pre)
                y_pre = y_pre.cpu().numpy()
                y_real = test_labels.cpu().numpy()
                auc = metrics.roc_auc_score(y_real, y_pre)
                auc_list.append(auc)
                precision_u, recall_u, thresholds_u = metrics.precision_recall_curve(y_real, y_pre)
                aupr = metrics.auc(recall_u, precision_u)
                aupr_list.append(aupr)
                y_score = np.where(y_pre >= 0.625, 1, 0)
                acc = metrics.accuracy_score(y_real, y_score)
                acc_list.append(acc)
                f1 = metrics.f1_score(y_real, y_score)
                f1_list.append(f1)
                recall = metrics.recall_score(y_real, y_score)
                recall_list.append(recall)
                precision = metrics.precision_score(y_real, y_score)
                pre_list.append(precision)
                print(' fold {:01d} auc {:.4f} pre {:.4f} acc {:.4f} recall {:.4f} f1 {:.4f} aupr {:.4f}'.format(
                    i + 1,
                    auc,
                    precision,
                    acc,
                    recall,
                    f1,
                    aupr))
        print(f'all auc:{auc_list}')
        print(f'all precision:{pre_list}')
        print(f'all acc:{acc_list}')
        print(f'all recall:{recall_list}')
        print(f'all f1:{f1_list}')
        print(f'all aupr:{aupr_list}')
        arr_auc = np.array(auc_list)
        arr_pre = np.array(pre_list)
        arr_acc = np.array(acc_list)
        arr_recall = np.array(recall_list)
        arr_f1 = np.array(f1_list)
        arr_aupr = np.array(aupr_list)
        averages_auc = np.round(np.mean(arr_auc, axis=0), 4)
        averages_pre = np.round(np.mean(arr_pre, axis=0), 4)
        averages_acc = np.round(np.mean(arr_acc, axis=0), 4)
        averages_recall = np.round(np.mean(arr_recall, axis=0), 4)
        averages_f1 = np.round(np.mean(arr_f1, axis=0), 4)
        averages_aupr = np.round(np.mean(arr_aupr, axis=0), 4)
        print(
            'final auc {:.4f} pre {:.4f} acc {:.4f} recall {:.4f} f1 {:.4f} aupr {:.4f}'.format(averages_auc,
                                                                                                averages_pre,
                                                                                                averages_acc,
                                                                                                averages_recall,
                                                                                                averages_f1,
                                                                                                averages_aupr))