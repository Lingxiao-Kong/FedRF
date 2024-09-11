"""
维护一个local_model_list用来存放客户端的模型，每次训练的时候加载
全局的模型不再需要，
计算的acc，应该是acc_list_l
local_model初始化，还是使用自己的模型初始化
"""
# Importing necessary libraries and modules
import sys, os
import torch.nn.functional as F

from utils.proto_utils import proto_aggregation, count_loss2, agg_func, count_all_protos

# Getting the base path of the project and adding it to the system path
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

# Importing torch and neural network modules
import torch
from torch import nn, optim

# Importing time, copy, and custom modules
import time
import copy
from nets.models import DigitModel,BasicBlock,ResNet18

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
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # USPS transformation
    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Synth Digits transformation
    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # MNIST-M transformation
    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Define datasets for each dataset type
    mnist_trainset     = data_utils.DigitsDataset(data_path="data/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist)
    mnist_testset      = data_utils.DigitsDataset(data_path="data/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist)
    svhn_trainset      = data_utils.DigitsDataset(data_path='data/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
    svhn_testset       = data_utils.DigitsDataset(data_path='data/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn)
    usps_trainset      = data_utils.DigitsDataset(data_path='data/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps)
    usps_testset       = data_utils.DigitsDataset(data_path='data/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps)
    synth_trainset     = data_utils.DigitsDataset(data_path='data/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=transform_synth)
    synth_testset      = data_utils.DigitsDataset(data_path='data/SynthDigits/', channels=3, percent=args.percent,  train=False, transform=transform_synth)
    mnistm_trainset    = data_utils.DigitsDataset(data_path='data/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm)
    mnistm_testset     = data_utils.DigitsDataset(data_path='data/MNIST_M/', channels=3, percent=args.percent,  train=False, transform=transform_mnistm)


    # Create data loaders for each dataset
    mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
    mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
    svhn_train_loader  = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
    svhn_test_loader   = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
    usps_train_loader  = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
    usps_test_loader   = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
    synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True)
    synth_test_loader  = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False)
    mnistm_train_loader= torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True)
    mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False)

    # Aggregate train and test loaders for each dataset!!!!!!!!!!!
    train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
    test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]

    return train_loaders, test_loaders

# Function to train a model

#todo 在这个里面传入一个servermodel，用这个中间层的输出作为本地model的对应层的输入
def train(args,model, train_loader, optimizer, loss_fun, global_protos , device  ):
    # Set the model to training mode
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    # Iterate over the training loader
    train_iter = iter(train_loader)
    # add proto
    agg_protos_label = {}

    for step in range(len(train_iter)):
        # Zero the gradients
        # optimizer.zero_grad()
        model.zero_grad();
        # Get the inputs and labels
        x, y = next(train_iter)
        label_g = y
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        # Forward pass
        output , protos = model(x)   # 这部分的proto是client

        loss1 = loss_fun(output, y)
        # 改成count_loss2
        loss_mse = nn.MSELoss()
        if len(global_protos) == 0:
            loss2 = 0 * loss1
        else:
            proto_new = copy.deepcopy(protos.data)
            i = 0
            for label in label_g:
                if label.item() in global_protos.keys():
                    proto_new[i, :] = global_protos[label.item()][0].data
                i += 1
            loss2 = loss_mse(proto_new, protos)

        loss = loss1 + loss2 *  args.ld
        # Backward pass
        # Compute loss
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
        # Compute accuracy
        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()

        for i in range(len(y)):
            if label_g[i].item() in agg_protos_label:
                agg_protos_label[label_g[i].item()].append(protos[i, :])
            else:
                agg_protos_label[label_g[i].item()] = [protos[i, :]]

    # Return average loss and accuracy
    return loss_all/len(train_iter), correct/num_data , agg_protos_label


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

        output , other = model(data,-1)

        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()

    return test_loss/len(test_loader), correct /len(test_loader.dataset)

# 用来测试所有数据集在server上面的性能,用客户端的平均结果来表示
def test_for_server(models, test_loaders, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []
    test_loaders_len=0
    test_loaders_dataset_len=0
    for idx in range(len(test_loaders)):
        test_loaders_len += len(test_loaders[idx])
        test_loaders_dataset_len += len(test_loaders[idx].dataset)
        for data, target in test_loaders[idx]:
            data = data.to(device).float()
            target = target.to(device).long()
            targets.append(target.detach().cpu().numpy())

            output , other = models[idx](data,-1)

            test_loss += loss_fun(output, target).item()
            pred = output.data.max(1)[1]

            correct += pred.eq(target.view(-1)).sum().item()

    return test_loss / test_loaders_len, correct / test_loaders_dataset_len
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

        output , proto = model(data, -1)

        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()
        for i in range(len(target)):
            file_path = './' + args.alg + str(idx)+ '_protos.npy'
            if os.path.isfile(file_path):
                existing_data = np.load(file_path)
                x = []
                x.append(proto[i, :].cpu().detach().numpy())
                npProto = np.array(x)
                combined_data = np.concatenate((existing_data, npProto))
                np.save(file_path, combined_data)
            else:
                x = []
                x.append(proto[i,:].cpu().detach().numpy())
                npProto = np.array(x)
                np.save('./' + args.alg + str(idx) + '_protos.npy', npProto)

            file_path = './' + args.alg + str(idx)+ '_y.npy'
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
                np.save('./' + args.alg + str(idx)+ '_y.npy', npTarget)
            # if label_g[i].item() in agg_protos_label[idx]:
            #     agg_protos_label[idx][label_g[i].item()].append(proto[i, :])
            # else:
            #     np.save('./' + args.alg + '_protos.npy', proto[i,:])

    return test_loss / len(test_loader), correct / len(test_loader.dataset)

# Main function
if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help ='whether to make a log')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size')
    parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='fedproto', help='fedavg | fedprox | fedbn')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint/digits', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--gpu', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')
    parser.add_argument('--ld', type=int, default=1, help='loss2的权重')
    parser.add_argument('--alg', type=str, default='fedproto', help="algorithms")

    args = parser.parse_args()

    # Device setup
    torch.cuda.set_device(int(args.gpu))
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # Seed setup for reproducibility
    seed= 1
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
        logfile = open(os.path.join(log_path,'{}.log'.format(args.mode)), 'a')
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
    # Instantiate server model and define loss function
    server_model = ResNet18(BasicBlock, [2, 2, 2, 2], 1, 10) #7 is the numbr of classes

    # loss_fun = nn.NLLLoss()
    loss_fun = nn.CrossEntropyLoss()

    # Prepare data loaders for training and testing
    train_loaders, test_loaders = prepare_data(args)

    # Define datasets for each client
    datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']

    # Federated setting
    client_num = len(datasets)
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    # Testing pretrained models
    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('snapshots/digits/{}'.format(args.mode.lower()))
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower()=='fedbn':
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
        if args.mode.lower()=='fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        resume_iter = int(checkpoint['a_iter']) + 1
        print('Resume training from epoch {}'.format(resume_iter))
    else:
        resume_iter = 0
    # 开始添加 proto
    global_protos = []

    all_start = time.time()

    # Start training
    for a_iter in range(resume_iter, args.iters):
        # 开始添加 proto
        local_protos = {}

        # 因为代码的逻辑和proto的有一点点差别，因此，另外维护一个字典
        local_protos_dic = {}

        start = time.time()
        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        #TODO 修改wk_iters和client的执行顺序，以和fedproto保持一致
        for client_idx in range(client_num):
            print("============ Train epoch {} ============".format(client_idx + a_iter * args.wk_iters))
            model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
            test_loader = test_loaders[client_idx]

            for wi in range(args.wk_iters):
                if args.log: logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))
                loss_all, correct, protos = train(args, model, train_loader, optimizer, loss_fun, global_protos, device)

                agg_protos = agg_func(protos)
                local_protos[client_idx] = agg_protos

                # 聚合前client训练一次的输出结果
                train_loss, train_acc = test(model, train_loader, loss_fun, device)
                print('聚合前client本地更新结果:  {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(
                    datasets[client_idx], train_loss, train_acc))
                test_loss, test_acc = test(models[client_idx], test_loader, loss_fun, device)
                print('聚合前client本地更新结果:  {:<11s}| Test Loss: {:.4f} | Test Acc: {:.4f}'.format(
                    datasets[client_idx], test_loss, test_acc))

        # update global weights
        global_protos = proto_aggregation(local_protos)
        # 采用febproto之后，不再进行模型的聚合，以及模型的更新

        #server聚合之后的结果输出
        server_train_loss, server_train_acc = test_for_server(models, train_loaders, loss_fun, device)
        print('聚合后server： {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format("all" ,server_train_loss, server_train_acc))
        server_test_loss, server_test_acc = test_for_server(models, test_loaders, loss_fun, device)
        print('聚合后server： {:<11s}| Test Loss: {:.4f} | Test Acc: {:.4f}'.format("all" ,server_test_loss, server_test_acc))


        # Report after aggregation 聚合后client的输出结果
        for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                train_loss, train_acc = test(model, train_loader, loss_fun, device)
                print('聚合后client： {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx] ,train_loss, train_acc))
                if args.log:
                    logfile.write(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx] ,train_loss, train_acc))

        # Start testing
        for test_idx, test_loader in enumerate(test_loaders):

            if test_idx < 5:
                if a_iter == 99:
                    global_proto_np = []
                    global_y_np = []
                    for key in global_protos.keys():
                        item = global_protos[key][0].cpu().detach().numpy()
                        # 保存全局聚合的proto和对应的label
                        global_proto_np.append(item)
                        global_y_np.append(key)
                    global_proto_np = np.array(global_proto_np)
                    global_y_np = np.array(global_y_np)
                    np.save('./' + args.alg + '_global_protos.npy', global_proto_np)
                    np.save('./' + args.alg + '_global_labels.npy', global_y_np)
                    test_loss, test_acc = test4Proto(models[test_idx], test_loader, loss_fun, device,test_idx)
                else:
                    test_loss, test_acc = test(models[test_idx], test_loader, loss_fun, device)
                print('聚合后client： {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss, test_acc))
                if args.log:
                    logfile.write(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_loss, test_acc))

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
