import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def sd(x):
    return np.std(x, axis=0, ddof=1)

def sd_gpu(x):
    return torch.std(x, dim=0)

def normalize_gpu(x):
    x = F.normalize(x, p=1, dim=1)
    return x

def normalize(x):
    mean = np.mean(x, axis=0)
    std = sd(x)
    std[std == 0] = 1
    x = (x - mean) / std
    return x

def random_fourier_features_gpu(x, w=None, b=None, num_f=None, sigma=None, seed=None):
    '''
    Done: generate fourier featrues for X(N, d) all at once
    TODO: seed, if the seed is all the same
    TODO: try normalize, have all the tensors devided by length
    '''
    if num_f is None:
        num_f = 1
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        w = 1 / sigma * (torch.randn(size=(num_f, c)))
        b = 2 * np.pi * torch.rand(size=(r, num_f))
        b = b.repeat((n, 1, 1))

    Z = torch.sqrt(torch.tensor(2.0 / num_f).cuda())

    mid = torch.matmul(x.cuda(), w.t().cuda())
    mid = mid + b.cuda()
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0].cuda()
    mid *= np.pi / 2.0

    Z = Z * (torch.cos(mid).cuda() + torch.sin(mid).cuda())
    return Z

def lossc(inputs, target, weight):
    loss = nn.NLLLoss(reduce=False)
    return loss(inputs, target).view(1, -1).mm(weight).view(1)


def lossb_expect(cfeaturec, weight, cfs, num_f, sample_rate):
    
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f).cuda()
    loss = Variable(torch.FloatTensor([0]).cuda())
    weight = weight.cuda()
    for i in range(cfeaturecs.size()[-1]):

        cfeaturec = cfeaturecs[:, :, i]
        weight_matrix = torch.diag(weight.view(weight.size()[0]))
        conduct_matrix = cfeaturec.t().mm(weight_matrix)
        conduct_vector_matrix = cfeaturec.t().mm(weight)
        conduct_vector = conduct_vector_matrix.view(-1)
        conduct_cr_matrix = conduct_matrix.mm(cfeaturec)
        conduct_cr_vector = conduct_vector_matrix[:, 0]
        random_points = np.random.rand(cfs)
        for p in range(cfs):
            if random_points[p] > sample_rate:
                continue

            conduct_matric_mid = conduct_cr_matrix.clone()
            conduct_matric_mid[:, p] = 0
            conduct_vector_mid = conduct_cr_vector.clone()
            conduct_vector_mid[p] = 0

            loss += (conduct_matric_mid[p, :] -
                       conduct_vector[p] * conduct_vector_mid).pow(2).sum(0).view(1)

    return loss


def lossq(cfeatures, cfs):
    return - cfeatures.pow(2).sum(1).mean(0).view(1) / cfs


def lossn(cfeatures):
    return cfeatures.mean(0).pow(2).mean(0).view(1)
