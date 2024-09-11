"""
federated learning with different aggregation strategy on benchmark exp.
"""
# Importing necessary libraries and modules
import sys, os
import torch.nn.functional as F

# Getting the base path of the project and adding it to the system path
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

# Importing torch and neural network modules
import torch
from torch import nn, optim

# Importing time, copy, and custom modules
import time
import copy
from nets.models_old import DigitModel, BasicBlock, ResNet18

# Importing argument parsing module
import argparse

# Importing numpy and torchvision modules
import numpy as np
import torchvision
import torchvision.transforms as transforms

# Importing custom data utility functions
from utils import data_utils


# Function to prepare data loaders for different datasets
def prepare_data(args):
    # Define transformations for different datasets

    # MNIST transformation
    transform_mnist = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # SVHN transformation
    transform_svhn = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # USPS transformation
    transform_usps = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Synth Digits transformation
    transform_synth = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # MNIST-M transformation
    transform_mnistm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Define datasets for each dataset type
    mnist_trainset = data_utils.DigitsDataset(data_path="data/MNIST", channels=1, percent=args.percent, train=True,
                                              transform=transform_mnist)
    mnist_testset = data_utils.DigitsDataset(data_path="data/MNIST", channels=1, percent=args.percent, train=False,
                                             transform=transform_mnist)
    svhn_trainset = data_utils.DigitsDataset(data_path='data/SVHN', channels=3, percent=args.percent, train=True,
                                             transform=transform_svhn)
    svhn_testset = data_utils.DigitsDataset(data_path='data/SVHN', channels=3, percent=args.percent, train=False,
                                            transform=transform_svhn)
    usps_trainset = data_utils.DigitsDataset(data_path='data/USPS', channels=1, percent=args.percent, train=True,
                                             transform=transform_usps)
    usps_testset = data_utils.DigitsDataset(data_path='data/USPS', channels=1, percent=args.percent, train=False,
                                            transform=transform_usps)
    synth_trainset = data_utils.DigitsDataset(data_path='data/SynthDigits/', channels=3, percent=args.percent,
                                              train=True, transform=transform_synth)
    synth_testset = data_utils.DigitsDataset(data_path='data/SynthDigits/', channels=3, percent=args.percent,
                                             train=False, transform=transform_synth)
    mnistm_trainset = data_utils.DigitsDataset(data_path='data/MNIST_M/', channels=3, percent=args.percent, train=True,
                                               transform=transform_mnistm)
    mnistm_testset = data_utils.DigitsDataset(data_path='data/MNIST_M/', channels=3, percent=args.percent, train=False,
                                              transform=transform_mnistm)

    # Create data loaders for each dataset
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

    # Aggregate train and test loaders for each dataset!!!!!!!!!!!
    train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
    test_loaders = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]

    return train_loaders, test_loaders


# Function to train a model

# todo 在这个里面传入一个servermodel，用这个中间层的输出作为本地model的对应层的输入
def train(model, train_loader, optimizer, loss_fun, client_num, device, output_layer, server_model, train_epoch=-1):
    # Set the model to training mode
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    # Iterate over the training loader
    train_iter = iter(train_loader)

    for step in range(len(train_iter)):
        # Zero the gradients
        optimizer.zero_grad()
        # Get the inputs and labels
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        # Forward pass
        output = model(x, -1)
        loss = loss_fun(output, y)
        # Backward pass
        # Compute loss
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        # Compute accuracy
        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()

        # if train_epoch != 0:
        #     # 上面这个output也需要计算损失
        #     # 对中间层的输出计算损失
        #     # Zero the gradients
        #     optimizer.zero_grad()
        #     server_model_middle_output = server_model(x, output_layer)  # TODO 应该把这个也设置成server_model的输入
        #     input_layer = output_layer + 1
        #     middleoutput = model(server_model_middle_output, -1, input_layer)
        #
        #     middle_loss = loss_fun(middleoutput, y)
        #     # Backward pass
        #     middle_loss.backward()
        #     loss_all += middle_loss.item()
        #     optimizer.step()
        #     # Compute accuracy
        #     pred = middleoutput.data.max(1)[1]
        #     correct += pred.eq(y.view(-1)).sum().item()

    # Return average loss and accuracy
    return loss_all / len(train_iter) * 2, correct / num_data * 2


# Function to train a model with FedProx
def train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x, -1)

        loss = loss_fun(output, y)

        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if step > 0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += args.mu / 2. * w_diff
        #############################################################################

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all / len(train_iter), correct / num_data


# Function to test a model
def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        output = model(data, -1)

        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()

    return test_loss / len(test_loader), correct / len(test_loader.dataset)

def test4Proto(model, test_loader, loss_fun, device,idx):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []

    for data, target in test_loader:
        data = data.to(device).float()
        label_g = target
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        output = model(data, -1)

        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()
        for i in range(len(target)):
            file_path = './' + args.alg + '_inputs.npy'
            if os.path.isfile(file_path):
                existing_data = np.load(file_path)
                input = []
                input.append(data[i, :].cpu().detach().numpy())
                npInput = np.array(input)
                combined_data = np.concatenate((existing_data, npInput))
                np.save(file_path, combined_data)
            else:
                input = []
                input.append(data[i,:].cpu().detach().numpy())
                npInput = np.array(input)
                np.save('./' + args.alg + '_inputs.npy', npInput)

            file_path = './' + args.alg + '_y.npy'
            if os.path.isfile(file_path):
                existing_data = np.load(file_path)
                targetY = []
                targetY.append(target[i].item())
                npTarget = np.array(targetY)
                combined_data = np.concatenate((existing_data, npTarget))
                np.save(file_path, combined_data)
            else:
                targetY = []
                targetY.append(target[i].item())
                npTarget = np.array(targetY)
                np.save('./' + args.alg + '_y.npy', npTarget)

    return test_loss / len(test_loader), correct / len(test_loader.dataset)

# 用来测试所有数据集在server上面的性能
def test_for_server(model, test_loaders, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []
    test_loaders_len = 0
    test_loaders_dataset_len = 0

    for test_loader in test_loaders:

        test_loaders_len += len(test_loader)
        test_loaders_dataset_len += len(test_loader.dataset)
        for data, target in test_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            targets.append(target.detach().cpu().numpy())

            output = model(data, -1)

            test_loss += loss_fun(output, target).item()
            pred = output.data.max(1)[1]

            correct += pred.eq(target.view(-1)).sum().item()

    return test_loss / test_loaders_len, correct / test_loaders_dataset_len


# Key function for communication and aggregation
def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


# Main function
if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to make a log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--percent', type=float, default=0.5, help='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--iters', type=int, default=100, help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=5,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='fedavg', help='fedavg | fedprox | fedbn')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type=str, default='../checkpoint/digits', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')
    parser.add_argument('--gpu', type=str, default='3', help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')
    parser.add_argument('--output_layer', type=int, default=1, help='output layer for forward')
    parser.add_argument('--alg', type=str, default='fedavg', help="algorithms")

    args = parser.parse_args()

    # Device setup
    torch.cuda.set_device(int(args.gpu))
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # Seed setup for reproducibility
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print('Device:', device)

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

    # Instantiate server model and define loss function
    # Instantiate server model and define loss function
    # server_model = DigitModel().to(device)
    server_model = ResNet18(BasicBlock, [2, 2, 2, 2], 1, 10)  # 7 is the numbr of classes
    # if torch.cuda.device_count() > 1:
    #     print("We use",torch.cuda.device_count(), "GPUs")
    #     server_model = nn.DataParallel(server_model)   # to use the multiple GPUs
    server_model.to(device)

    loss_fun = nn.CrossEntropyLoss()

    # Prepare data loaders for training and testing
    train_loaders, test_loaders = prepare_data(args)

    # Define datasets for each client
    datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']

    # Federated setting
    client_num = len(datasets)
    client_weights = [1 / client_num for i in range(client_num)]
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    # Testing pretrained models
    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('snapshots/digits/{}'.format(args.mode.lower()))
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        else:
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(server_model, test_loader, loss_fun, device)
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        exit(0)

    # Resuming training
    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        resume_iter = int(checkpoint['a_iter']) + 1
        print('Resume training from epoch {}'.format(resume_iter))
    else:
        resume_iter = 0

    all_start = time.time()

    # Start training
    for a_iter in range(resume_iter, args.iters):
        start = time.time()
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log: logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))

            for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                test_loader = test_loaders[client_idx]
                if args.mode.lower() == 'fedprox':
                    if a_iter > 0:
                        train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device)
                    else:
                        train(model, train_loader, optimizer, loss_fun, client_num, device, args.output_layer,
                              server_model)
                        # 聚合前client训练一次的输出结果
                        train_loss, train_acc = test(model, train_loader, loss_fun, device)
                        print('聚合前client本地更新结果:  {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(
                            datasets[client_idx], train_loss, train_acc))
                        test_loss, test_acc = test(models[client_idx], test_loader, loss_fun, device)
                        print('聚合前client本地更新结果:  {:<11s}| Test Loss: {:.4f} | Test Acc: {:.4f}'.format(
                            datasets[client_idx], test_loss, test_acc))

                else:
                    train(model, train_loader, optimizer, loss_fun, client_num, device, args.output_layer, server_model,
                          a_iter)
                    # 聚合前client训练一次的输出结果
                    train_loss, train_acc = test(model, train_loader, loss_fun, device)
                    print('聚合前client本地更新结果:  {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(
                        datasets[client_idx], train_loss, train_acc))
                    test_loss, test_acc = test(models[client_idx], test_loader, loss_fun, device)
                    print('聚合前client本地更新结果:  {:<11s}| Test Loss: {:.4f} | Test Acc: {:.4f}'.format(
                        datasets[client_idx], test_loss, test_acc))

        # Aggregation 聚合
        server_model, models = communication(args, server_model, models, client_weights)
        # server聚合之后的结果输出
        server_train_loss, server_train_acc = test_for_server(server_model, train_loaders, loss_fun, device)
        print('聚合后server： {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format("all", server_train_loss,
                                                                                     server_train_acc))
        server_test_loss, server_test_acc = test_for_server(server_model, test_loaders, loss_fun, device)
        print('聚合后server： {:<11s}| Test Loss: {:.4f} | Test Acc: {:.4f}'.format("all", server_test_loss,
                                                                                   server_test_acc))

        # Report after aggregation 聚合后client的输出结果
        for client_idx in range(client_num):
            model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
            train_loss, train_acc = test(model, train_loader, loss_fun, device)
            print(
                '聚合后client： {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], train_loss,
                                                                                       train_acc))
            if args.log:
                logfile.write(
                    ' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx], train_loss,
                                                                                train_acc))

        # Start testing
        for test_idx, test_loader in enumerate(test_loaders):
            if test_idx < 5:
                if a_iter == 99:
                    test_loss, test_acc = test4Proto(models[test_idx], test_loader, loss_fun, device,
                                                     test_idx)
                else:
                    test_loss, test_acc = test(models[test_idx], test_loader, loss_fun, device)
                print('聚合后client： {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx],
                                                                                             test_loss, test_acc))
                if args.log:
                    logfile.write(
                        ' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_loss,
                                                                                    test_acc))

        end = time.time()
        print('\nTrain time for epoch #%d : %f second' % (a_iter, end - start))

    all_end = time.time()
    print('\nTrain time for all #%d : %f second' % (a_iter, all_end - all_start))
    # Save checkpoint
    print(' Saving checkpoints to {}...'.format(SAVE_PATH))
    if args.mode.lower() == 'fedbn':
        torch.save({
            'model_0': models[0].state_dict(),
            'model_1': models[1].state_dict(),
            'model_2': models[2].state_dict(),
            'model_3': models[3].state_dict(),
            'model_4': models[4].state_dict(),
            'server_model': server_model.state_dict(),
        }, SAVE_PATH)
    else:
        torch.save({
            'server_model': server_model.state_dict(),
        }, SAVE_PATH)

    if log:
        logfile.flush()
        logfile.close()
