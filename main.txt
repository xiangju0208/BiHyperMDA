import warnings
from param import *
from train import *

if __name__ == '__main__':
    args = parse_args()
    warnings.filterwarnings("ignore")
    train_valid_test(args)