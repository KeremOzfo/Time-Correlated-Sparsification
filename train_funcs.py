import torch
from torch.utils.data import DataLoader
# custom modules
import data_loader as dl
from nn_classes import *
import server_functions as sf
import math
from parameters import *
import time
import numpy as np
from tqdm import tqdm


def evaluate_accuracy(model, testloader, device):
    """Calculates the accuracy of the model"""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)


    net_users = [get_net(args).to(device) for u in range(num_client)]

    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay = 1e-4) for cl in range(num_client)]
    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)
    schedulers = [torch.optim.lr_scheduler.StepLR(optimizers[cl], step_size=30, gamma=0.1) for cl in range(num_client)]

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    modelsize = sf.count_parameters(net_ps)
    errorCorCof = 1
    layer_types = []
    for p in net_ps.named_parameters():
        names = p[0]
        layer_types.append(names.split('.'))
    errors = []
    accuracys = []
    ps_model_mask = torch.ones(modelsize).to(device)
    sf.initialize_zero(net_ps)
    currentLR = args.lr
    for cl in range(num_client):
        errors.append(torch.zeros(modelsize).to(device))
    runs = math.ceil(N_s/(args.bs*num_client))

    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    for epoch in tqdm(range(args.num_epoch)):
        if epoch == args.errDecayVals[0] and args.errorDecay is True:
            errorCorCof = args.errDecayVals[1]
        if args.warmUp and epoch < 5:
            for cl in range(num_client):
                for param_group in optimizers[cl].param_groups:
                    if epoch ==0:
                        param_group['lr'] = 0.1
                    else:
                        lr_change = (args.lr - 0.1) / 4
                        param_group['lr'] = (lr_change * epoch) + 0.1
        if epoch in args.lr_change:
            for cl in range(num_client):
                sf.adjust_learning_rate(optimizers[cl], epoch, args.lr_change, args.lr)
        currentLR = sf.get_LR(optimizers[0])

        for run in range(runs):

            for cl in range(num_client):

                trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                         shuffle=True)
                for data in trainloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizers[cl].zero_grad()
                    predicts = net_users[cl](inputs)
                    loss = criterions[cl](predicts, labels)
                    loss.backward()
                    optimizers[cl].step()
                    break
            ps_model_flat = sf.get_model_flattened(net_ps, device)
            ps_model_dif = torch.zeros_like(ps_model_flat)
            for cl in range(num_client):
                model_flat = sf.get_model_flattened(net_users[cl], device)
                model_flat.add_(errors[cl] * currentLR * errorCorCof)
                difmodel = (model_flat.sub(ps_model_flat)).to(device)
                difmodel_clone = torch.clone(difmodel).to(device)

                if not (args.warmUp and epoch < 5):
                    if args.layer_wise_spars and args.worker_LWS:
                        sf.sparse_timeC_alt(difmodel,args.sparsity_window,10,args.lws_sparsity_w,ps_model_mask,ind_pairs,device)
                    else:
                        sf.sparse_timeC(difmodel, args.sparsity_window, 10, ps_model_mask, device)

                    if args.quantization:
                        sf.groups(difmodel,args.num_groups,args.denominator,device)
                    errors[cl] = (difmodel_clone.sub(difmodel)) / currentLR

                ps_model_dif.add_(difmodel/num_client)
            ps_model_flat.add_(ps_model_dif)
            topk = math.ceil(ps_model_dif.nelement() / args.sparsity_window)
            ind = torch.topk(ps_model_dif.abs(), k=topk, dim=0)[1]
            if not (args.warmUp and epoch < 5):
                if args.layer_wise_spars:
                    ps_model_mask = sf.sparse_special_mask(ps_model_flat, args.sparsity_window, args.lws_sparsity, ind_pairs, device)
                else:
                    ps_model_mask *= 0
                    ps_model_mask[ind] = 1

            sf.make_model_unflattened(net_ps, ps_model_flat, net_sizes, ind_pairs)

            [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

            '''
            if run %10 == 0: ##debug
                acc = evaluate_accuracy(net_ps, testloader, device)
                print('accuracy:', acc * 100)
                break
            '''

        acc = evaluate_accuracy(net_ps, testloader, device)
        accuracys.append(acc * 100)
        print('accuracy:',acc*100,)
    return accuracys