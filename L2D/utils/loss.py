import torch
from tqdm import tqdm


def triplet_loss(a,p,n,alpha):
    '''
    a,p,n denotes anchor,positive,negative
    calculate the loss with cosine distance
    output cos(a,p),cos(a,n),max(cos(a,n)-cos(a,p)+alpha, 0)
    '''
    bs = len(a)
    D_ap = []
    D_an = []
    loss = 0
    for i in range(bs):
        anchor = a[i]/torch.norm(a[i])
        positive = p[i]/torch.norm(p[i])
        negative = n[i]/torch.norm(n[i])
        ap = anchor.dot(positive)
        an = anchor.dot(negative)
        D_ap.append(ap)
        D_an.append(an)
    
    loss = [max(D_an[i] - D_ap[i] + alpha, 0) for i in range(bs)]

    ave_loss = sum(loss)/bs
    ave_dap = sum(D_ap)/bs
    ave_dan = sum(D_an)/bs

    return ave_loss, ave_dap, ave_dan

def contrastive_loss(a,p,n):

    bs = len(a)
    loss = 0
    for i in range(bs):
        anchor = a[i]/torch.norm(a[i])
        positive = p[i]/torch.norm(p[i])
        negative = n[i]/torch.norm(n[i])
        ap = anchor.dot(positive)
        an = anchor.dot(negative)
        loss -= torch.log(torch.exp(ap)/(torch.exp(ap)+torch.exp(an)))
    loss /= bs
    return loss


def cos_dist(a,b):
    dist = []
    for i in range(len(a)):
        ai = a[i]/torch.norm(a[i])
        bi = b[i]/torch.norm(b[i])
        dist.append(ai.dot(bi))
    return dist

def tri_test(trinet=None,dataloader=None,device=None):

    total = 0
    right = 0
    bar = tqdm(total=500)
    dataitr = iter(dataloader)
    for i in range(500):

        a,p,n = next(dataitr)
        a1 = a.to(device)
        p1 = p.to(device)
        n1 = n.to(device)

        _,anchor = trinet(a1)
        _,positive = trinet(p1)
        _,negative = trinet(n1)

        total += 10
        dap = cos_dist(anchor,positive)
        dan = cos_dist(anchor,negative)
        for j in range(10):
            if dap[j]>=dan[j] + 0.0001:
                right += 1
        bar.update()
    bar.close()
    acc = float(right)/total
    return acc