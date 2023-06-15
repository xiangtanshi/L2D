import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.modules.activation import LeakyReLU
import torchvision.models as models
import random 
import os
import numpy as np

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')

def to_device(data,device):
    return data.to(device)

def seed_torch(seed=1001):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



class  Pattern (nn.Module) :
    '''
    revision network
    '''

    def __init__(self,classes=8) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=200,kernel_size=int(classes),stride=int(classes),padding=0)
        self.bn = nn.BatchNorm1d(num_features=200*2,momentum=0.95)
        self.fc1 = nn.Linear(in_features=200*2,out_features=100)
        self.fc2 = nn.Linear(in_features=100,out_features=classes)
       
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def forward(self, input_batch, batchsize):
        out = self.conv1(input_batch).view(batchsize,-1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out1 = self.fc2(out)       
        return out1


def main():

    model = Pattern(classes=10)
    model_dict = model.state_dict()

    torch.save(model_dict,'./models/aux/pattern_ani.pkl')

    model = Pattern(classes=8)
    model_dict = model.state_dict()

    torch.save(model_dict,'./models/aux/pattern_vel.pkl')
    

if __name__ == '__main__':
    main()

