import torch
from tqdm import tqdm
import torch.nn.functional as F

def loss(prob, label):
    loss = torch.nn.CrossEntropyLoss()  # input is the raw logit, the log softmax operations are implemented in itself
    return loss(prob, label)

def triplet_loss(a,p,n,alpha):
    '''
    a,p,n denotes anchor,positive,negative
    calculate the loss with cosine distance
    output cos(a,p),cos(a,n),max(cos(a,n)-cos(a,p)+alpha, 0)
    '''
    bs = len(a)
    S_ap = []
    S_an = []
    loss = 0
    for i in range(bs):
        anchor = a[i]/torch.norm(a[i])
        positive = p[i]/torch.norm(p[i])
        negative = n[i]/torch.norm(n[i])
        ap = anchor.dot(positive)
        an = anchor.dot(negative)
        S_ap.append(ap)
        S_an.append(an)
    
    loss = [max(S_an[i] - S_ap[i] + alpha, 0) for i in range(bs)]

    ave_loss = sum(loss)/bs
    ave_dap = sum(S_ap)/bs
    ave_dan = sum(S_an)/bs

    return ave_loss, ave_dap, ave_dan

def triplet_loss_(a, p, n, alpha):
    '''
    a,p,n denotes anchor,positive,negative
    calculate the loss with cosine distance
    output cos(a,p),cos(a,n),max(cos(a,n)-cos(a,p)+alpha, 0)
    '''
    bs = a.shape[0]
    
    # Normalize the vectors
    a_norm = F.normalize(a, p=2, dim=1)
    p_norm = F.normalize(p, p=2, dim=1)
    n_norm = F.normalize(n, p=2, dim=1)
    
    # Compute cosine similarities
    S_ap = torch.sum(a_norm * p_norm, dim=1)
    S_an = torch.sum(a_norm * n_norm, dim=1)
    
    # Compute loss
    loss = torch.clamp(S_an - S_ap + alpha, min=0)
    
    ave_loss = torch.mean(loss)
    ave_dap = torch.mean(S_ap)
    ave_dan = torch.mean(S_an)

    return ave_loss, ave_dap, ave_dan



def cos_dist(a,b):
    dist = []
    for i in range(len(a)):
        ai = a[i]/torch.norm(a[i])
        bi = b[i]/torch.norm(b[i])
        dist.append(ai.dot(bi))
    return dist

def cos_dist_(a, b):
    a_norm = a / torch.norm(a, dim=1, keepdim=True)
    b_norm = b / torch.norm(b, dim=1, keepdim=True)
    return torch.sum(a_norm * b_norm, dim=1)


def tri_test(trinet=None,dataloader=None,device=None,num=50):


    right = 0
    dataitr = iter(dataloader)
    for i in tqdm(range(int(10000/num))):

        a,p,n = next(dataitr)
        a1 = a.to(device)
        p1 = p.to(device)
        n1 = n.to(device)

        _,anchor = trinet(a1)
        _,positive = trinet(p1)
        _,negative = trinet(n1)

        dap = cos_dist(anchor,positive)
        dan = cos_dist(anchor,negative)
        for j in range(num):
            if dap[j]>=dan[j] + 0.0001:
                right += 1

    acc = float(right)/10000
    return acc

def tri_test_(trinet, dataloader, device):
    total = 50*200  # 500 batches * 10 samples per batch
    right = 0
    margin = 0.0001

    with torch.no_grad():
        for a, p, n in tqdm(dataloader, total=200):
            a, p, n = a.to(device), p.to(device), n.to(device)

            _, anchor = trinet(a)
            _, positive = trinet(p)
            _, negative = trinet(n)

            dap = cos_dist_(anchor, positive)
            dan = cos_dist_(anchor, negative)

            right += torch.sum(dap >= dan + margin).item()

    acc = right / total
    return acc
