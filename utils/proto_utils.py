import numpy as np
import torch
from torch import nn, optim
import copy

def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label



def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

# 计算添加了proto后的loss2
def count_loss2(loss1,protos,labels,global_protos_1):
    loss_mse = nn.MSELoss()
    if len(global_protos_1) == 0:
        loss2 = 0 * loss1
    else:
        proto_new = copy.deepcopy(protos.data)
        i = 0
        for label in labels:
            if label.item() in global_protos_1.keys():
                proto_new[i, :] = global_protos_1[label.item()][0].data
            i += 1
        loss2 = loss_mse(proto_new, protos)
    return loss2

def count_all_protos(protos,local_protos_dic,client_idx):
    if client_idx in local_protos_dic:
        for proto in protos:
            if proto in local_protos_dic[client_idx]:
                local_protos_dic[client_idx][proto] = local_protos_dic[client_idx][proto] + protos[proto]
            else:
                local_protos_dic[client_idx][proto] = protos[proto]
    else:
        local_protos_dic[client_idx] = protos
    return local_protos_dic[client_idx]


def save_protos(args, local_model_list, testloaders,num_clients,device):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    # criterion = nn.NLLLoss().to(device)

    agg_protos_label = {}
    for idx in range(num_clients):
        agg_protos_label[idx] = {}
        model = local_model_list[idx]
        # model.to(device)
        testloader = testloaders[idx]

        model.eval()
        test4Proto(testloader,model,agg_protos_label,device,idx)

    x = []
    y = []
    d = []
    for i in range(num_clients):
        for label in agg_protos_label[i].keys():
            for proto in agg_protos_label[i][label]:
                if args.device == 'cuda':
                    tmp = proto.cpu().detach().numpy()
                else:
                    tmp = proto.detach().numpy()
                x.append(tmp)
                y.append(label)
                d.append(i)

    x = np.array(x)
    y = np.array(y)
    d = np.array(d)
    np.save('./' + args.alg + '_protos.npy', x)
    np.save('./' + args.alg + '_labels.npy', y)
    np.save('./' + args.alg + '_idx.npy', d)

    print("Save protos and labels successfully.")

def test4Proto(testloader,model,agg_protos_label,device,idx):
    for images, labels in testloader:
        # model.zero_grad()
        label_g = labels
        images = images.to(device).float()
        labels = labels.to(device).long()
        # Forward pass
        output, protos = model(images)  # 这部分的proto是client

        # Compute accuracy
        pred = output.data.max(1)[1]

        for i in range(len(labels)):
            if label_g[i].item() in agg_protos_label[idx]:
                agg_protos_label[idx][label_g[i].item()].append(protos[i, :])
            else:
                agg_protos_label[idx][label_g[i].item()] = [protos[i, :]]