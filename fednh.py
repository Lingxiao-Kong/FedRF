"""
federated learning with different aggregation strategy on benchmark exp.
"""
import sys, os
import yaml

from fednh_baseline.flbase.strategies.CReFF import CReFFClient, CReFFServer
from fednh_baseline.flbase.strategies.Ditto import DittoClient, DittoServer
from fednh_baseline.flbase.strategies.FedBABU import FedBABUClient, FedBABUServer
from fednh_baseline.flbase.strategies.FedNH import FedNHClient, FedNHServer
from fednh_baseline.flbase.strategies.FedPer import FedPerClient, FedPerServer
from fednh_baseline.flbase.strategies.FedROD import FedRODClient, FedRODServer
from fednh_baseline.flbase.strategies.FedRep import FedRepClient, FedRepServer
from fednh_baseline.flbase.utils import setup_clients, setup_clients4ours

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
from torch import nn, optim
import time
import copy
from nets.models_old import DigitModel
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils import data_utils


def prepare_data(args):
    # Prepare data
    transform_mnist = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_svhn = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_usps = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_synth = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mnistm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # MNIST
    mnist_trainset = data_utils.DigitsDataset(data_path="data/MNIST", channels=1, percent=args.percent, train=True,
                                              transform=transform_mnist)
    mnist_testset = data_utils.DigitsDataset(data_path="data/MNIST", channels=1, percent=args.percent, train=False,
                                             transform=transform_mnist)

    # SVHN
    svhn_trainset = data_utils.DigitsDataset(data_path='data/SVHN', channels=3, percent=args.percent, train=True,
                                             transform=transform_svhn)
    svhn_testset = data_utils.DigitsDataset(data_path='data/SVHN', channels=3, percent=args.percent, train=False,
                                            transform=transform_svhn)

    # USPS
    usps_trainset = data_utils.DigitsDataset(data_path='data/USPS', channels=1, percent=args.percent, train=True,
                                             transform=transform_usps)
    usps_testset = data_utils.DigitsDataset(data_path='data/USPS', channels=1, percent=args.percent, train=False,
                                            transform=transform_usps)

    # Synth Digits
    synth_trainset = data_utils.DigitsDataset(data_path='data/SynthDigits/', channels=3, percent=args.percent,
                                              train=True, transform=transform_synth)
    synth_testset = data_utils.DigitsDataset(data_path='data/SynthDigits/', channels=3, percent=args.percent,
                                             train=False, transform=transform_synth)

    # MNIST-M
    mnistm_trainset = data_utils.DigitsDataset(data_path='data/MNIST_M/', channels=3, percent=args.percent,
                                               train=True, transform=transform_mnistm)
    mnistm_testset = data_utils.DigitsDataset(data_path='data/MNIST_M/', channels=3, percent=args.percent,
                                              train=False, transform=transform_mnistm)

    mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
    svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch, shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
    usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch, shuffle=True)
    usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
    synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch, shuffle=True)
    synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False)
    mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch, shuffle=True)
    mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False)

    train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
    test_loaders = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]

    return train_loaders, test_loaders


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to make a log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--percent', type=float, default=0.1, help='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--iters', type=int, default=100, help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=5,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='fedbn', help='fedavg | fedprox | fedbn')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type=str, default='../checkpoint/digits', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')
    parser.add_argument('--gpu', type=str, default='1', help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')
    parser.add_argument('--yamlfile', type=str, default='./fednh_baseline/fednh.yaml', help='Configuration file.')
    parser.add_argument('--strategy', default='CReFF', type=str, help='strategy FL')
    parser.add_argument('--num_clients', default=5, type=int, help='number of clients')
    parser.add_argument('--num_rounds', default=200, type=int, help='number of communication rounds')
    parser.add_argument('--participate_ratio', default=1.0, type=float, help='participate ratio')
    parser.add_argument('--beta', default='1.0', type=str, help='Dirichlet Distribution parameter')    # overwrite with inputs
    parser.add_argument('--num_classes_per_client', default=None, type=int, help='pathological non-iid parameter')
    parser.add_argument('--num_shards_per_client', default=None, type=int, help='pathological non-iid parameter fedavg simulation')
    parser.add_argument('--global_seed', default=2022, type=int, help='Global random seed.')# optimizer
    parser.add_argument('--optimizer', default='SGD', type=str, help='Optimizer')
    parser.add_argument('--client_lr', default=1e-2, type=float, help='client side initial learning rate')
    parser.add_argument('--client_lr_scheduler', default='stepwise', type=str, help='client side learning rate update strategy')
    parser.add_argument('--sgd_momentum', default=0.0, type=float, help='sgd momentum')
    parser.add_argument('--sgd_weight_decay', default=1e-5, type=float, help='sgd weight decay')
    parser.add_argument('--use_sam', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='Use SAM optimizer')
    parser.add_argument('--no_norm', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='Use group/batch norm or not')
    parser.add_argument('--FedNH_server_adv_prototype_agg', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='FedNH server adv agg')
    parser.add_argument('--FedNH_client_adv_prototype_agg', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='FedNH client adv agg')
    parser.add_argument('--FedNH_smoothing', default=0.9, type=float, help='moving average parameters')
    parser.add_argument('--FedROD_hyper_clf', default=True, type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='FedRod phead uses hypernetwork')
    parser.add_argument('--FedROD_phead_separate', default=True, type=lambda x: (str(x).lower()
                        in ['true', '1', 'yes']), help='FedROD phead separate train')
    parser.add_argument('--FedProto_lambda', default=0.1, type=float, help='FedProto local penalty lambda')
    parser.add_argument('--FedRep_head_epochs', default=10, type=int, help='FedRep local epochs to update head')
    parser.add_argument('--FedBABU_finetune_epoch', default=5, type=int, help='FedBABU local epochs to finetune')
    parser.add_argument('--Ditto_lambda', default=0.75, type=float, help='penalty parameter for Ditto')
    parser.add_argument('--CReFF_num_of_fl_feature', default=100, type=int, help='num of federated feature per class')
    parser.add_argument('--CReFF_match_epoch', default=100, type=int, help='epoch used to minmize gradient matching loss')
    parser.add_argument('--CReFF_crt_epoch', default=300, type=int, help='epoch used to retrain classifier')
    parser.add_argument('--CReFF_lr_net', default=0.01, type=float, help='lr for head')
    parser.add_argument('--CReFF_lr_feature', default=0.1, type=float, help='lr for feature')
    args = parser.parse_args()

    #Device setup
    torch.cuda.set_device(int(args.gpu))
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print('Device:', device)

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # Experiment folder setup
    exp_folder = 'federated_digits'

    args.save_path = os.path.join(args.save_path, exp_folder)

    log = args.log
    if log:
        log_path = os.path.join('../logs/digits/', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path, '{}.log'.format(args.mode)), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    iters: {}\n'.format(args.iters))
        logfile.write('    wk_iters: {}\n'.format(args.wk_iters))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))
    print(args.save_path)

    criterion = nn.CrossEntropyLoss()

    # prepare the data
    train_loaders, test_loaders = prepare_data(args)

    # name of each client dataset
    datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']

    # federated setting
    client_num = len(datasets)
    client_weights = [1 / client_num for i in range(client_num)]
    # 加载参数
    with open(args.yamlfile, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    # parse the default setting
    server_config = config['server_config']
    client_config = config['client_config']



    server_config['strategy'] = args.strategy
    server_config['num_clients'] = args.num_clients
    server_config['num_rounds'] = args.num_rounds
    server_config['participate_ratio'] = args.participate_ratio
    # server_config['partition'] = args.partition
    server_config['beta'] = args.beta
    server_config['num_classes_per_client'] = args.num_classes_per_client
    server_config['num_shards_per_client'] = args.num_shards_per_client
    client_config['num_rounds'] = args.num_rounds
    client_config['global_seed'] = args.global_seed
    client_config['optimizer'] = args.optimizer
    client_config['client_lr'] = args.client_lr
    client_config['client_lr_scheduler'] = args.client_lr_scheduler
    client_config['sgd_momentum'] = args.sgd_momentum
    client_config['sgd_weight_decay'] = args.sgd_weight_decay
    client_config['use_sam'] = args.use_sam
    client_config['no_norm'] = args.no_norm

    if args.strategy == 'FedNH':
        ClientCstr, ServerCstr = FedNHClient, FedNHServer
        server_config['FedNH_smoothing'] = args.FedNH_smoothing
        server_config['FedNH_server_adv_prototype_agg'] = args.FedNH_server_adv_prototype_agg
        client_config['FedNH_client_adv_prototype_agg'] = args.FedNH_client_adv_prototype_agg
    elif args.strategy == "FedPer":
        ClientCstr, ServerCstr = FedPerClient, FedPerServer
    elif args.strategy == 'FedROD':
        ClientCstr, ServerCstr = FedRODClient, FedRODServer
        client_config['FedROD_hyper_clf'] = args.FedROD_hyper_clf
        client_config['FedROD_phead_separate'] = args.FedROD_phead_separate
    elif args.strategy == 'FedRep':
        ClientCstr, ServerCstr = FedRepClient, FedRepServer
        client_config['FedRep_head_epochs'] = args.FedRep_head_epochs
    elif args.strategy == 'FedBABU':
        ClientCstr, ServerCstr = FedBABUClient, FedBABUServer
        client_config['FedBABU_finetune_epoch'] = args.FedBABU_finetune_epoch
    elif args.strategy == 'Ditto':
        ClientCstr, ServerCstr = DittoClient, DittoServer
        client_config['Ditto_lambda'] = args.Ditto_lambda
    elif args.strategy == 'CReFF':
        ClientCstr, ServerCstr = CReFFClient, CReFFServer
        server_config['CReFF_match_epoch'] = args.CReFF_match_epoch
        server_config['CReFF_crt_epoch'] = args.CReFF_crt_epoch
        server_config['CReFF_lr_net'] = args.CReFF_lr_net
        server_config['CReFF_lr_feature'] = args.CReFF_lr_feature
    else:
        raise ValueError("Invalid strategy!")

    client_config_lst = [client_config for i in range(client_num)]

    # TODO 初始化client模型，记得删除上面的初始化
    clients_dict = setup_clients4ours(ClientCstr, train_loaders, test_loaders, criterion,
                                 client_config_lst, device,client_num,
                                 server_config=server_config,
                                 beta=server_config['beta'],
                                 num_classes_per_client=server_config['num_classes_per_client'],
                                 num_shards_per_client=server_config['num_shards_per_client'],
                                 )

    # TODO 初始化server模型
    server = ServerCstr(server_config, clients_dict, exclude=server_config['exclude'],
                        server_side_criterion=criterion, global_testset=test_loaders, global_trainset=train_loaders,
                        client_cstr=ClientCstr, server_side_client_config=client_config,
                        server_side_client_device=device)

    # path = './result/febnh'
    server.run(filename=args.save_path + 'fednh_best_global_model.pkl', use_wandb=False, global_seed=args.global_seed)
    server.save(filename=args.save_path + 'fednh_final_server_obj.pkl', keep_clients_model=args.keep_clients_model)




