# implementing counterfactual comparision with the help of the siamese net

from utils.dataset_class import *
from utils.triplet_class import *
from utils.resnet import *
from torch.utils.data import  DataLoader
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
 


class Args:
    def __init__(self, device = torch.device('cuda:3'), model = '0.pkl' ,trinet = 'trinet3_res.pkl', target_set = [a_ani_valid, a_ani_test, a_valid_counter, a_test_counter],
                    cls_num = 10, optimz = 'adam/', sets = 'animal/', mods = 'resnet/', datas = 'valid', datatype = 'animals', strs = 'ani/'):

        '''
        model: the trained classification model
        trinet: siamese net
        datas: valid or test
        target_set: original val/test dataset and the corresponding counterfactual dataset
        the rest parameters are used to determine the path of model and trinet
        '''

        self.device = device
        self.cls_num = cls_num
        self.datas = datas
        self.datatype = datatype
        self.strs = strs

        # loader1 corresponds to original dataset, loader 2 corresponds to counterfactual dataset
        self.valid_loader1 = DataLoader(dataset=target_set[0],batch_size=1,shuffle=False,num_workers=1)                 
        self.valid_loader2 = DataLoader(dataset=target_set[2],batch_size=(1+cls_num)*4,shuffle=False,num_workers=4) 
        self.test_loader1 = DataLoader(dataset=target_set[1],batch_size=1,shuffle=False,num_workers=1)                   
        self.test_loader2 = DataLoader(dataset=target_set[3],batch_size=(cls_num+1)*4,shuffle=False,num_workers=4)   

        cls_net = ResNet(num_classes=cls_num)
        cls_net.load_state_dict(torch.load('./models/aux/' + optimz + sets + mods + model))     
        self.cls_net = cls_net.to(device)
        self.cls_net.eval()

        retro_net = ResNet(num_classes=cls_num)
        retro_net.load_state_dict(torch.load('./models/aux/trinet_' + strs[0:3] + '_' + optimz +trinet))    
        self.retro_net = retro_net.to(device)
        self.retro_net.eval()


def softmax(logit):
    return torch.exp(logit)/torch.sum(torch.exp(logit))


def sorting(logits,args):
    # rearrange the class index according to their probability such that order[0] is the top 1 class, order[1] is the top 2 class, etc.
    order = []
    num = args.cls_num
    logit = torch.zeros(1,num)
    for i in range(num):
        if logits.shape == (1,num):
            logit[0,i] = logits[0,i]
        else:
            logit[0,i] = logits[i]
    for i in range(num):
        order.append(logit.argmax())
        logit[0,order[-1]] = -100
    return order


def pattern(args):
    if args.datas == 'test':
        loader1 = args.test_loader1     
        loader2 = args.test_loader2     
    else:
        loader1 = args.valid_loader1
        loader2 = args.valid_loader2

    iter1 = iter(loader1)  
    iter2 = iter(loader2)  
    total = len(loader1)  
    bar = tqdm(total=total)

    file = open('./Cgn/cgn_data/counterfactual_data/logit/' + args.strs + args.datas + '_logit.txt','w')   
    contents = []

    # file1 = open('order.txt','w')
    positive = 0.0

    for i in range(total):

        bar.update()
        img,label = next(iter1)
        imgs,_ = next(iter2)
        img = img.to(args.device)
        imgs = imgs.to(args.device)

        logit,_ = args.cls_net(img)
        # logit,_ = args.cls_net(imgs[args.cls_num:args.cls_num+1])
        logit = logit.cpu() 
        probability = softmax(logit)
        orders = sorting(probability,args)
        pred = orders[0]
        if pred == label:
            positive += 1

        _,feats = args.retro_net(imgs) 
        for j in range(4*(args.cls_num+1)):                              
            feats[j] = feats[j]/torch.norm(feats[j])

        cosdist = []
        for k in range(4):
            cosdist.append(feats[(args.cls_num+1)*k:(args.cls_num+1)*(k+1)-1,:]@feats[(args.cls_num+1)*(k+1)-1,:].T)              
            cosdist[k] = cosdist[k].cpu()

        patterns = torch.zeros(1,2*args.cls_num)

        mean = (cosdist[0] + cosdist[1] + cosdist[2] + cosdist[3])/4
        m_pred = mean.argmax()

        # class label
        for s in range(args.cls_num):                                 
            if orders[s] == label:
                new_label = s

        # if probability[0,pred] > 0.9 and m_pred == pred:
        if False:
            pass
        else:
            contents.append('/data/dengx/counterfactual/' + args.datatype + '/logit/' + args.datas + '/{}.npy${}\n'.format(i,new_label))
        
            for k in range(args.cls_num):
                patterns[0,k] = probability[0,orders[k]]
                patterns[0,k+args.cls_num] = mean[orders[k]] 

            patterns = patterns.detach().numpy()
            np.save('/data/dengx/counterfactual/' + args.datatype + '/logit/' + args.datas + '/{}.npy'.format(i),patterns)

            # file1.write(str(orders)+ str(patterns[0]) +'\n')

    
    bar.close()
    file.writelines(contents)
    file.close()
    # file1.close()
    accuracy = positive/total
    print(accuracy)

    
def CL_testing(args):

    # record the faithful accuracy and unfaithful accuracy based on the range of max_prob

    if args.datas == 'test':
        iter1 = iter(args.test_loader1)
        iter2 = iter(args.test_loader2)
        total = len(args.test_loader1)
    else:
        raise ValueError('not expected data.')

    p_right = {'>0.99':0, '>0.95':0, '>0.9':0, '0.9~0.8':0, '0.8~0.7':0, '0.7~0.6':0,'0.6~0.5':0, '<0.5':0 }
    p_total = {'>0.99':0, '>0.95':0, '>0.9':0, '0.9~0.8':0, '0.8~0.7':0, '0.7~0.6':0,'0.6~0.5':0, '<0.5':0 }
    p_faith_right = {'>0.99':0, '>0.95':0, '>0.9':0, '0.9~0.8':0, '0.8~0.7':0, '0.7~0.6':0,'0.6~0.5':0, '<0.5':0 }
    p_faith_total = {'>0.99':0, '>0.95':0, '>0.9':0, '0.9~0.8':0, '0.8~0.7':0, '0.7~0.6':0,'0.6~0.5':0, '<0.5':0 }

    f_total = 0
    f_true = 0

    bar = tqdm(total=total)

    for i in range(total):

        bar.update()
        img,label = next(iter1)
        imgs,_ = next(iter2)
        img = img.to(args.device)
        imgs = imgs.to(args.device)

        # logit,_ = args.cls_net(img)
        logit,_ = args.cls_net(imgs[args.cls_num:args.cls_num+1])
        probability = softmax(logit)
        order = sorting(probability, args)
        pred = order[0]

        _,feats = args.retro_net(imgs) 
        for j in range(4*(args.cls_num+1)):                              
            feats[j] = feats[j]/torch.norm(feats[j])

        cosdist = []
        for k in range(4):
            cosdist.append(feats[(args.cls_num+1)*k:(args.cls_num+1)*(k+1)-1,:]@feats[(args.cls_num+1)*(k+1)-1,:].T)              
            cosdist[k] = cosdist[k].cpu() 

        mean = (cosdist[0] + cosdist[1] + cosdist[2] + cosdist[3])/4

        final_pred = mean.argmax()

        # fusion on hard samples
        if probability[0,pred]<0.9 and pred != final_pred:
            f_total += 1
            fused = probability.cpu()+ 1.0 * mean
            f_pred = fused.argmax()
            if f_pred == label:
                f_true += 1

        # cl test
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
        
    bar.close()

    print("p_right:\n",p_right)
    print("p_total:\n",p_total)
    print("p_faith_right:\n",p_faith_right)
    print('P_faith_total:\n',p_faith_total)

    print('f_true:{},f_total:{}'.format(f_true,f_total))


def pattern1(args):
    if args.datas == 'test':
        loader1 = args.test_loader1     
        loader2 = args.test_loader2     
    else:
        loader1 = args.valid_loader1
        loader2 = args.valid_loader2

    iter1 = iter(loader1)  
    iter2 = iter(loader2)  
    total = len(loader1)  
    bar = tqdm(total=total)

    file = open('./Cgn/cgn_data/counterfactual_data/logit/' + args.strs + args.datas + '_logit.txt','w')   
    contents = []

    positive = 0.0

    for i in range(total):

        bar.update()
        img,label = next(iter1)
        imgs,_ = next(iter2)
        img = img.to(args.device)
        imgs = imgs.to(args.device)

        logit,_ = args.cls_net(img)
        # logit,_ = args.cls_net(imgs[args.cls_num:args.cls_num+1])
        logit = logit.cpu() 
        probability = softmax(logit)
        orders = sorting(probability,args)
        pred = orders[0]
        if pred == label:
            positive += 1

        _,feats = args.retro_net(imgs) 
        for j in range(4*(args.cls_num+1)):                              
            feats[j] = feats[j]/torch.norm(feats[j])

        cosdist = []
        for k in range(4):
            cosdist.append(feats[(args.cls_num+1)*k:(args.cls_num+1)*(k+1)-1,:]@feats[(args.cls_num+1)*(k+1)-1,:].T)              
            cosdist[k] = cosdist[k].cpu()

        patterns = torch.zeros(1,2*args.cls_num)

        mean = (cosdist[0] + cosdist[1] + cosdist[2] + cosdist[3])/4

        contents.append('/data/dengx/counterfactual/' + args.datatype + '/logit/' + args.datas + '/{}.npy${}\n'.format(i,label[0]))
    
        for k in range(args.cls_num):
            patterns[0,k] = probability[0,k]
            patterns[0,k+args.cls_num] = mean[k] 

        patterns = patterns.detach().numpy()
        np.save('/data/dengx/counterfactual/' + args.datatype + '/logit/' + args.datas + '/{}.npy'.format(i),patterns)

    
    bar.close()
    file.writelines(contents)
    file.close()
    accuracy = positive/total
    print(accuracy)

if __name__ == '__main__':

    optimiz1 = 'adam/'
    m1,m2,m3 = 'resnet/','rsc/','dsl/'

    target = 'animals'

    if target == 'animals':
        args = Args(datatype=target, cls_num=10, sets='animal/', strs='ani/', target_set = [a_ani_valid, a_ani_test, a_valid_counter, a_test_counter],
                device=torch.device("cuda:1"), model='0.pkl', optimz=optimiz1, datas='test',
                trinet='trinet3_dsl.pkl', mods=m3)
    else:
        args = Args(datatype=target, cls_num=8, sets='vehicle/', strs='vel/', target_set = [a_vel_valid, a_vel_test, v_valid_counter, v_test_counter],
                device=torch.device("cuda:1"), model='0.pkl', optimz=optimiz1, datas='test',
                trinet='trinet3_res.pkl', mods=m1)
        
    # CL_testing(args)

    args.datas = 'valid'
    pattern1(args)
    args.datas = 'test'
    pattern1(args)


