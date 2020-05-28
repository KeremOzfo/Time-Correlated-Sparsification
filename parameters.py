import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # dataset related
    parser.add_argument('--dataset_name', type=str, default='mnist', help='mnist, fmnist, cifar10')
    parser.add_argument('--nn_name', type=str, default='mnist', help='mnist, fmnist, simplecifar, resnet18')
    parser.add_argument('--dataset_dist', type=str, default='iid', help='distribution of dataset; iid or non_iid')

    # Federated params
    parser.add_argument('--num_client', type=int, default=10, help='number of clients')
    parser.add_argument('--bs', type=int, default=32, help='batchsize')
    parser.add_argument('--num_epoch', type=int, default=300, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.4, help='learning_rate')
    parser.add_argument('--layer_wise_spars', type=bool, default=True, help='include layer-wise sparsification')
    parser.add_argument('--sparsity_window', type=int, default=100, help='largest grad entry is chosen within this window')
    parser.add_argument('--lr_change', type=list, default=[150, 250], help='determines the at which epoch lr will decrease')
    # Quantization params
    parser.add_argument('--quantization', type=bool, default=True,help='apply quantization or not')
    parser.add_argument('--num_groups', type=int, default=16, help='add up the error weights or not')
    parser.add_argument('--denominator', type=float, default=1.2, help='divide groups by this')
    parser.add_argument('--avg_all', type=bool, default=True, help='apply quantization for all params')
    parser.add_argument('--all_avg_iter', type=int, default=25, help='at which iter make all avg')
    args = parser.parse_args()
    return args