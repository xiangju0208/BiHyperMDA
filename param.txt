import argparse
import torch as t

def parse_args():
    parser = argparse.ArgumentParser()
    device = t.device('cuda')
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--path', type=str, default='./HMDD')

    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)

    parser.add_argument('--kfolds', type=int, default=5)

    # The k means we pick up top k nodes
    parser.add_argument('--k', type=int, default=100)

    parser.add_argument('--k1', type=int, default=200)

    parser.add_argument('--proj', type=int, default=512)

    parser.add_argument('--quantile', type=float, default=0.1)

    parser.add_argument('--layer', type=int, default=1)

    parser.add_argument('--hyper_gcn_layer2', type=int, default=1)
   
    return parser.parse_args()