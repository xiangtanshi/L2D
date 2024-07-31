import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from PIL import Image
from utils.dataset_class import *
from utils.resnet import *
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='parameter setting for training the siamese network.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--d', type=int, default=18)
parser.add_argument('--device', type=int, default=1)

args =  parser.parse_args()


def softmax(logit):
    return torch.exp(logit)/torch.sum(torch.exp(logit))


def CL_dists(network=None, dataloader=None, dists = None, dists_=None, device=None):

    positive_prediction = 0
    total = 0

    for data in tqdm(dataloader):
        images,labels = data
        images, labels = images.to(device),labels.to(device)
        results,_ = network(images)
        probability = softmax(results)

        _,predicted = torch.max(results.data,1)
        correc = (predicted==labels).sum()
        positive_prediction += correc
        total += labels.size(0) 

        if probability[0][predicted]>0.9:
            dists['a'] += 1
            dists_['a'] += correc.item()
        elif probability[0][predicted]>0.8:
            dists['b'] += 1
            dists_['b'] += correc.item()
        elif probability[0][predicted]>0.7:
            dists['c'] += 1
            dists_['c'] += correc.item()
        elif probability[0][predicted]>0.6:
            dists['d'] += 1
            dists_['d'] += correc.item()
        elif probability[0][predicted]>0.5:
            dists['e'] += 1
            dists_['e'] += correc.item()
        else:
            dists['f'] += 1
            dists_['f'] += correc.item()

    accu = float(positive_prediction)/total
    return accu


def main(args):

    device = torch.device('cuda:{}'.format(args.device))
    if args.d == 50:
            net = ResNet(num_classes=17, block=Bottleneck, layers=[3, 4, 6, 3])
            net.load_state_dict(torch.load('./models/resnet50/nico-normal-0.pkl')) 
    elif args.d == 18:
        net = ResNet(num_classes=17, block=BasicBlock, layers=[2, 2, 2, 2])
        net.load_state_dict(torch.load('./models/resnet18/nico-normal-0.pkl')) 

    # record the percentage of samples in each confidence level
    # a: p>0.9, b: 0.9~0.8, c: 0.8~0.7, d: 0.7~0.6, e: 0.6~0.5, f: <0.5
    name_list = ['a','b','c','d','e','f']
    dist_num = {'a':0,'b':0,'c':0,'d':0,'e':0, 'f':0}
    dist_correct = {'a':0,'b':0,'c':0,'d':0,'e':0, 'f':0}
 

    net = net.to(device)
    net.eval()

    loader = DataLoader(dataset=test1, batch_size=1, shuffle=False)
    accuracy = CL_dists(dataloader=loader,network=net,dists=dist_num,dists_=dist_correct,device=device)
    print(accuracy)
    
    val = [v for v in dist_num.values()]
    acc = [dist_correct[name]/dist_num[name] for name in name_list]
    # plt.bar(name_list, val, fc='y')
    # plt.xlabel('Confidence Level')
    # plt.ylabel('Number')
    # plt.savefig('./results/original_num.jpg')

    plt.bar(name_list, acc, fc='g')
    plt.xlabel('Confidence Level')
    plt.ylabel('Accuracy')
    plt.savefig('./results/original_acc.jpg')


if __name__ == '__main__':
    main(args)
