import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # dataset related
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='mnist, fmnist, cifar10')
    parser.add_argument('--nn_name', type=str, default='resnet18', help='mnist, fmnist, simplecifar, resnet18')
    parser.add_argument('--dataset_dist', type=str, default='iid', help='distribution of dataset; iid or non_iid')

    # Federated params
    parser.add_argument('--num_client', type=int, default=20, help='number of clients')
    parser.add_argument('--bs', type=int, default=32, help='batchsize')
    parser.add_argument('--sparse_type',type=str,default='t',help='types of sparsification')
    parser.add_argument('--num_epoch', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.2, help='learning_rate')
    parser.add_argument('--eval_period', type=int, default=10, help='evaluation period')
    parser.add_argument('--layer_wise_spars', type=bool, default=True, help='include layer-wise sparsification')
    parser.add_argument('--sparsity_window', type=int, default=100, help='largest grad entry is chosen within this window')
    args = parser.parse_args()
    return args