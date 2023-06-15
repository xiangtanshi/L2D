import torch
import torch.nn as nn


def loss(prob, label):
    loss = nn.CrossEntropyLoss()
    return loss(prob, label)

def lossc (inputs, target, weight) :
    '''
    :param inputs: output of classifier in Network, i.e. the logit
    :param target: true label
    :param weight: learned weight
    :return:
    '''
    #loss = nn.NLLLoss (reduction='none')
    #return loss (inputs, target).view (1, -1).mm (weight).view (1)
    loss = nn.CrossEntropyLoss(reduction='none')
    return loss(inputs,target).view(1,-1).mm(weight).view(1)

def lossb (cfeaturec, weight, cfs) :
    '''
    for batch confounder balancing, target is to learn the weight W for a batch
    :param cfeaturec: extracted feature of image batch
    :param weight: expected weight for batch samples
    :param cfs: length of cfeaturec
    :return: loss
    '''

    cfeatureb = (cfeaturec.sign () + 1).sign ()    #calculating the indicator matrix: I
    mfeatureb = 1 - cfeatureb
    #loss = Variable (torch.FloatTensor ([0]).cuda ())
    loss = torch.tensor([0.0]).cuda()

    for p in range (cfs) :      #successively regard each element in feature as treatment
        if p == 0 :
            cfeaturer = cfeaturec [:, 1 : cfs]
        elif p == cfs - 1 :
            cfeaturer = cfeaturec [:, 0 : cfs - 1]
        else :
            cfeaturer = torch.cat ((cfeaturec [:, 0 : p], cfeaturec [:, p + 1 : cfs]), 1)
        
        if cfeatureb [:, p : p + 1].t ().mm (weight).view (1).data [0] != 0 or mfeatureb [:, p : p + 1].t ().mm (weight).view (1).data [0] != 0 :
            if cfeatureb [:, p : p + 1].t ().mm (weight).view (1).data [0] == 0 :
                loss += (cfeaturer.t().mm (mfeatureb [:, p : p + 1] * weight) / mfeatureb [:, p : p + 1].t ().mm (weight)).pow (2).sum (0).view (1)
            elif mfeatureb [:, p : p + 1].t ().mm (weight).view (1).data [0] == 0 :
                loss += (cfeaturer.t().mm (cfeatureb [:, p : p + 1] * weight) / cfeatureb [:, p : p + 1].t ().mm (weight)).pow (2).sum (0).view (1)
            else :
                loss += (cfeaturer.t().mm (cfeatureb [:, p : p + 1] * weight) / cfeatureb [:, p : p + 1].t ().mm (weight) -
                         cfeaturer.t().mm (mfeatureb [:, p : p + 1] * weight) / mfeatureb [:, p : p + 1].t ().mm (weight)).pow (2).sum (0).view (1)

    return loss


def lossq (cfeatures) :
    #quantization loss for feature binarization
    return - cfeatures.pow (2).sum (1).mean (0).view (1)

def lossn (cfeatures) :
    return cfeatures.mean (0).pow (2).mean (0).view (1)

