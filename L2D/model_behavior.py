'''
There are two things that is recorded here:
1. Sample distribution in datasets(train/val/test) according to the Confidence Level that are assigned to them by the trained resnet model.

'''

import torch
import torch.nn as nn
from PIL import Image
from utils.dataset_class import *
from utils.resnet import *
import matplotlib.pyplot as plt
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,0,1,2'

class Args:
    def __init__(self,datasets=a_ani_test,classes=10,paths='./models/aux/trinet_ani_adam/trinet3_res.pkl'):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        print(self.device)
        self.datasets = datasets
        self.classes = classes
        self.paths = paths

def softmax(logit):
    return torch.exp(logit)/torch.sum(torch.exp(logit))


def CL_dists(network=None, dataloader=None, dists = None, device=None):

    positive_prediction = 0
    total = 0

    for data in dataloader:
        images,labels = data
        images, labels = images.to(device),labels.to(device)
        results,_ = network(images)
        probability = softmax(results)

        _,predicted = torch.max(results.data,1)
        positive_prediction += (predicted==labels).sum()
        total = labels.size(0) + total

        if probability[0][predicted]>0.9:
            dists['a'] += 1
        elif probability[0][predicted]>0.8:
            dists['b'] += 1
        elif probability[0][predicted]>0.7:
            dists['c'] += 1
        elif probability[0][predicted]>0.6:
            dists['d'] += 1
        elif probability[0][predicted]>0.5:
            dists['e'] += 1
        else:
            dists['f'] += 1

    accu = float(positive_prediction)/total
    return accu

def true_cls_rank(network=None, dataloader=None,dists=None, device=None):
    pass
    

def main(args):

    device = args.device
    classes = args.classes
    model_path =  args.paths

    # record the percentage of samples in each confidence level
    # a: p>0.9, b: 0.9~0.8, c: 0.8~0.7, d: 0.7~0.6, e: 0.6~0.5, f: <0.5
    name_list = ['a','b','c','d','e','f']
    distribution = {'a':0,'b':0,'c':0,'d':0,'e':0, 'f':0} 

    net = ResNet(num_classes=classes)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    net.eval()

    loader = DataLoader(dataset=args.datasets, batch_size=1, shuffle=False)
    accuracy = CL_dists(dataloader=loader,network=net,dists=distribution,device=device)
    print(accuracy)
    
    val = [v for v in distribution.values()]
    plt.bar(name_list, val, fc='b')
    plt.xlabel('Confidence Level')
    plt.ylabel('Number')
    plt.savefig('./result_records/CL_test.jpg')



if __name__ == '__main__':
    args = Args()
    main(args)
