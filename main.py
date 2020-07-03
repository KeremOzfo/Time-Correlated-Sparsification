from train_funcs import *
import numpy as np
from parameters import *
import torch
import random
import datetime
import os

device = torch.device("cpu")
args = args_parser()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    simulation_ID = int(random.uniform(1,999))
    print('device:',device)
    args = args_parser()
    for arg in vars(args):
       print(arg, ':', getattr(args, arg))
    results = []
    x = datetime.datetime.now()
    date = x.strftime('%b') + '-' + str(x.day)
    newFile = date + '-sim_ID-' + str(simulation_ID)
    if not os.path.exists(os.getcwd() + '/Results'):
        os.mkdir(os.getcwd() + '/Results')
    n_path = os.path.join(os.getcwd(), 'Results', newFile)
    for i in range(5):
        accs = train(args, device)
        if i == 0:
            os.mkdir(n_path)
            f = open(n_path + '/simulation_Details.txt', 'w+')
            f.write('simID = ' + str(simulation_ID) + '\n')
            f.write('############## Args ###############' + '\n')
            for arg in vars(args):
                line = str(arg) + ' : ' + str(getattr(args, arg))
                f.write(line + '\n')
            f.write('############ Results ###############' + '\n')
            f.close()
        if args.quantization:
            s_loc = date+'-Q-TimeCorralated'+'-numW-'+str(args.num_client)+'startingLR-'+str(args.lr)+'--'+str(i)
            s_loc = os.path.join(n_path,s_loc)
            np.save(s_loc,accs)
        else:
            s_loc = date + '-TimeCorralated' + '-startingLR-' + str(
                args.lr) + '-LWS-'+str(args.layer_wise_spars) +'-WorkerLWS-'+str(args.worker_LWS) +'ErrorDecay2-'+str(args.errorDecay)+ str(i)
            s_loc = os.path.join(n_path,s_loc)
            np.save(s_loc, accs)
        f = open(n_path + '/simulation_Details.txt', 'a+')
        f.write('Trial ' + str(i) + ' results at ' + str(accs[args.num_epoch]) + '\n')
        f.close()