from train_funcs import *
import numpy as np
from parameters import *
import torch
import datetime

device = torch.device("cpu")
args = args_parser()

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('device ',device, ' Dataset ',args.dataset_dist,' Model ',args.nn_name )
    args = args_parser()
    results = []
    x = datetime.datetime.now()
    date = x.strftime('%b') + '-' + str(x.day)
    for i in range(2):
        accs = train(args, device)
        np.save(date+'-TimeCorralated-NN-'+args.nn_name+'-numW-'+str(args.num_client)+'startingLR-'+str(args.lr)+'--'+str(i),accs)