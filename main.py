from train_funcs import *
import numpy as np
from parameters import *
import torch

device = torch.device("cpu")
args = args_parser()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device ',device, ' Dataset ',args.dataset_dist,' Model ',args.nn_name )
    args = args_parser()
    results = []
    for i in range(1):
        accs = train(args, device)
        if args.num_client ==1:
            np.save('Bencmark-NN-' + args.nn_name + '--' + str(i),
                    accs)
        else:
            if args.sparse_type == 'topk':
                np.save('topk' + args.nn_name + '-datadist-' + args.dataset_dist + '--numW-' + str(args.num_client)+'--'+str(i),
                        accs)
            else:
                np.save('modifiedTimeCorralated-NN-'+args.nn_name+'-numW-'+str(args.num_client)+'--'+str(i),accs)