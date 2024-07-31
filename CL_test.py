# implementing counterfactual comparision with the help of the siamese net

from utils.dataset_class import *
from triplet_class import *
from utils.resnet import *
from torch.utils.data import  DataLoader
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='parameter setting for training the siamese network.')
parser.add_argument('--d', type=int, default=18)
parser.add_argument('--device', type=int, default=4)
parser.add_argument('--optim', type=str, default='sgd')

args =  parser.parse_args()

if args.d == 18:
    cls_net = ResNet(num_classes=17, block=BasicBlock, layers=[2, 2, 2, 2])
    cls_net.load_state_dict(torch.load('./models/resnet18/nico-normal-0-{}.pkl'.format(args.optim))) 
    retro_net = ResNet(num_classes=17, block=BasicBlock, layers=[2, 2, 2, 2])
    retro_net.load_state_dict(torch.load('./models/resnet18/nico-siamese-0-200-{}.pkl'.format(args.optim)))  
elif args.d == 50:
    cls_net = ResNet(num_classes=17, block=Bottleneck, layers=[3, 4, 6, 3])
    cls_net.load_state_dict(torch.load('./models/resnet50/nico-normal-0-{}.pkl'.format(args.optim)))
    retro_net = ResNet(num_classes=17, block=Bottleneck, layers=[3, 4, 6, 3])
    retro_net.load_state_dict(torch.load('./models/resnet50/nico-siamese-0-400-{}.pkl'.format(args.optim)))

device = torch.device('cuda:{}'.format(args.device))
cls_net = cls_net.to(device)
retro_net = retro_net.to(device)
cls_net.eval()
retro_net.eval()

test_loader1 = DataLoader(dataset=test1, batch_size=1, shuffle=False)
test_loader2 = DataLoader(dataset=consensus_test, batch_size=1, shuffle=False)

def softmax(logit):
    return torch.exp(logit)/torch.sum(torch.exp(logit))
    
def pattern():

    iter1 = iter(test_loader1)
    loaded_result = np.loadtxt('./results/nico-{}-result.txt'.format(args.d), delimiter='  ')
    keys = ['>0.99', '>0.95', '>0.9', '0.9~0.8', '0.8~0.7', '0.7~0.6','0.6~0.5', '<0.5']
    p_right = defaultdict(int)
    p_total = defaultdict(int)
    p_faith_right = defaultdict(int)
    p_faith_total = defaultdict(int)
    p_unfaith_right = defaultdict(int)
    p_unfaith_total = defaultdict(int)
    for key in keys:
        p_right[key]=0; p_total[key]=0; p_faith_right[key]=0; p_unfaith_right[key]=0; p_faith_total[key]=0; p_unfaith_total[key]=0

    f_total = 0
    f_true = 0

    for idx in tqdm(range(len(iter1))):
        _, label = next(iter1)
        probability = loaded_result[idx][0:17]
        consensus = loaded_result[idx][17:]
        pred = probability.argmax()
        c_pred = consensus.argmax()


        if  probability[pred] > 0.99:
            p_total['>0.99'] += 1
            if pred == label:
                p_right['>0.99'] += 1
            if c_pred == pred:
                p_faith_total['>0.99'] += 1
                if label == pred:
                    p_faith_right['>0.99'] += 1
 
        elif probability[pred] >0.95:
            p_total['>0.95'] += 1
            if pred == label:
                p_right['>0.95'] += 1
            if c_pred == pred:
                p_faith_total['>0.95'] += 1
                if label == pred:
                    p_faith_right['>0.95'] += 1

        elif probability[pred] > 0.9:
            p_total['>0.9'] += 1
            if pred == label:
                p_right['>0.9'] += 1
            if c_pred == pred:
                p_faith_total['>0.9'] += 1
                if label == pred:
                    p_faith_right['>0.9'] += 1

        elif probability[pred] > 0.8:
            p_total['0.9~0.8'] += 1
            if pred == label:
                p_right['0.9~0.8'] += 1
            if c_pred == pred:
                p_faith_total['0.9~0.8'] += 1
                if label == pred:
                    p_faith_right['0.9~0.8'] += 1

        elif probability[pred] > 0.7:
            p_total['0.8~0.7'] += 1
            if pred == label:
                p_right['0.8~0.7'] += 1
            if c_pred == pred:
                p_faith_total['0.8~0.7'] += 1
                if label == pred:
                    p_faith_right['0.8~0.7'] += 1

        elif probability[pred] > 0.6:
            p_total['0.7~0.6'] += 1
            if pred == label:
                p_right['0.7~0.6'] += 1
            if c_pred == pred:
                p_faith_total['0.7~0.6'] += 1
                if label == pred:
                    p_faith_right['0.7~0.6'] += 1
 
        elif probability[pred] > 0.5:
            p_total['0.6~0.5'] += 1
            if pred == label:
                p_right['0.6~0.5'] += 1
            if c_pred == pred:
                p_faith_total['0.6~0.5'] += 1
                if label == pred:
                    p_faith_right['0.6~0.5'] += 1

        else:
            p_total['<0.5'] += 1
            if pred == label:
                p_right['<0.5'] += 1
            if c_pred == pred:
                p_faith_total['<0.5'] += 1
                if label == pred:
                    p_faith_right['<0.5'] += 1

        if pred == label:
            f_true += 1
        f_total += 1
    
    for key in keys:
        p_unfaith_right[key] = p_right[key] - p_faith_right[key]
        p_unfaith_total[key] = p_total[key] - p_faith_total[key]

    print("p_right:\n",p_right)
    print("p_total:\n",p_total)
    print("p_faith_right:\n",p_faith_right)
    print('P_faith_total:\n',p_faith_total)
    print("p_unfaith_right:\n",p_unfaith_right)
    print('P_unfaith_total:\n',p_unfaith_total)

    print('f_true:{},f_total:{},accuracy:{}'.format(f_true,f_total,f_true/f_total))

    for key in keys:
        print('cl:{},accuracy:{}'.format(key,p_right[key]/p_total[key]))
        print('cl:{},accuracy:{}'.format(key,p_faith_right[key]/p_faith_total[key]))
        print('cl:{},accuracy:{}'.format(key,p_unfaith_right[key]/p_unfaith_total[key]))
        print('')

    

def CL_testing():

    # record the faithful accuracy and unfaithful accuracy based on the range of max_prob

    iter1 = iter(test_loader1)
    iter2 = iter(test_loader2)
    total = len(test_loader1)

    keys = ['>0.99', '>0.95', '>0.9', '0.9~0.8', '0.8~0.7', '0.7~0.6','0.6~0.5', '<0.5']
    p_right = defaultdict(int)
    p_total = defaultdict(int)
    p_faith_right = defaultdict(int)
    p_faith_total = defaultdict(int)
    p_unfaith_right = defaultdict(int)
    p_unfaith_total = defaultdict(int)
    for key in keys:
        p_right[key]=0; p_total[key]=0; p_faith_right[key]=0; p_unfaith_right[key]=0; p_faith_total[key]=0; p_unfaith_total[key]=0

    pred_result = np.zeros((total,34))

    for idx in tqdm(range(total)):

        img,label = next(iter1)
        imgs = next(iter2)
        img = img.to(device)
        label = label.to(device)
        imgs = torch.stack(imgs,dim=0)
        imgs = imgs.squeeze(1)
        imgs = imgs.to(device)

        logit,_ = cls_net(img)
        # logit,_ = args.cls_net(imgs[0])
        probability = softmax(logit)
        # order = torch.argsort(probability, dim=1, descending=True)
        pred = probability.argmax()

        _,feats = retro_net(imgs) 
        feats = F.normalize(feats,p=2,dim=1)

        cosdist = F.cosine_similarity(feats[0:1],feats[1:],dim=1)

        mean = (cosdist[0:17] + cosdist[17:34] + cosdist[34:51])/3

        final_pred = mean.argmax()

        
        pred_result[idx,0:17] = probability[0].detach().cpu().numpy()
        pred_result[idx,17:] = mean.detach().cpu().numpy()

        # fusion on hard samples
        # if probability[0,pred]<0.9 and pred != final_pred:
        #     f_total += 1
        #     fused = probability.cpu()+ 1.0 * mean
        #     f_pred = fused.argmax()
        #     if f_pred == label:
        #         f_true += 1

        # effectiveness of uncertainty 
        if  probability[0,pred] > 0.99:
            p_total['>0.99'] += 1
            if pred == label:
                p_right['>0.99'] += 1
            if final_pred == pred:
                p_faith_total['>0.99'] += 1
                if label == pred:
                    p_faith_right['>0.99'] += 1
        elif probability[0,pred] >0.95:
            p_total['>0.95'] += 1
            if pred == label:
                p_right['>0.95'] += 1
            if final_pred == pred:
                p_faith_total['>0.95'] += 1
                if label == pred:
                    p_faith_right['>0.95'] += 1
        elif probability[0,pred] > 0.9:
            p_total['>0.9'] += 1
            if pred == label:
                p_right['>0.9'] += 1
            if final_pred == pred:
                p_faith_total['>0.9'] += 1
                if label == pred:
                    p_faith_right['>0.9'] += 1
        elif probability[0,pred] > 0.8:
            p_total['0.9~0.8'] += 1
            if pred == label:
                p_right['0.9~0.8'] += 1
            if final_pred == pred:
                p_faith_total['0.9~0.8'] += 1
                if label == pred:
                    p_faith_right['0.9~0.8'] += 1
        elif probability[0,pred] > 0.7:
            p_total['0.8~0.7'] += 1
            if pred == label:
                p_right['0.8~0.7'] += 1
            if final_pred == pred:
                p_faith_total['0.8~0.7'] += 1
                if label == pred:
                    p_faith_right['0.8~0.7'] += 1
        elif probability[0,pred] > 0.6:
            p_total['0.7~0.6'] += 1
            if pred == label:
                p_right['0.7~0.6'] += 1
            if final_pred == pred:
                p_faith_total['0.7~0.6'] += 1
                if label == pred:
                    p_faith_right['0.7~0.6'] += 1
        elif probability[0,pred] > 0.5:
            p_total['0.6~0.5'] += 1
            if pred == label:
                p_right['0.6~0.5'] += 1
            if final_pred == pred:
                p_faith_total['0.6~0.5'] += 1
                if label == pred:
                    p_faith_right['0.6~0.5'] += 1
        else:
            p_total['<0.5'] += 1
            if pred == label:
                p_right['<0.5'] += 1
            if final_pred == pred:
                p_faith_total['<0.5'] += 1
                if label == pred:
                    p_faith_right['<0.5'] += 1
    for key in keys:
        p_unfaith_right[key] = p_right[key] - p_faith_right[key]
        p_unfaith_total[key] = p_total[key] - p_faith_total[key]
        

    print("p_right:\n",p_right)
    print("p_total:\n",p_total)
    print("p_faith_right:\n",p_faith_right)
    print('P_faith_total:\n',p_faith_total)
    print("p_unfaith_right:\n",p_unfaith_right)
    print('P_unfaith_total:\n',p_unfaith_total)

    for key in keys:
        print('cl:{},accuracy:{}'.format(key,p_right[key]/p_total[key]))
        print('cl:{},accuracy:{}'.format(key,p_faith_right[key]/p_faith_total[key]))
        print('cl:{},accuracy:{}'.format(key,p_unfaith_right[key]/p_unfaith_total[key]))
        print('')

    # store the result
    np.savetxt('./results/nico-{}-{}-result.txt'.format(args.d,args.optim), pred_result, fmt='%.6f', delimiter='  ')


if __name__ == '__main__':
        
    CL_testing()
    # pattern()
