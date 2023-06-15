from torch import optim
from utils.loss import *
from utils.dataset_class import *
from utils.triplet_class import *
from utils.resnet import *
from utils.Convnet import *
from torch.utils.data import  DataLoader
import torch
import numpy as np
from tqdm import tqdm


class Args:
    def __init__(self, seed = 0, lr = 1e-4, bs = 32, momentum = 0.9, weight_decay = 5e-4, clss = 10, 
                    model_path = './models/aux/adam/animal/resnet/0.pkl', optimizer = 'adam', data = [a_train_triplets,a_test_triplets]):

        if torch.cuda.is_available():
            self.device = torch.device('cuda:1')
        else:
            self.device = torch.device('cpu')

        self.seed = seed
        self.learning_rate = lr
        self.batchsize = bs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.classes = clss
        self.model_path = model_path
        self.optimizer = optimizer
        self.data = data
        

class Trainer:

    def __init__(self, args):

        self.args = args

        # load the models with weights that are pretrained on imagenet
        model = ResNet(num_classes=args.classes)
        model.load_state_dict(torch.load(self.args.model_path))
        self.model = to_device(model,args.device)

        if args.optimizer == 'sgd':
            self.optimizer1 = optim.SGD(self.model.parameters(), args.learning_rate,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            self.optimizer1 = optim.Adam(self.model.parameters(),lr=args.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer1,step_size=14,gamma=0.1)


    def model_adjusting(self):
        # training the siamese network
        count = 0
        alpha = 0.4

        triplet_loader = DataLoader(dataset=self.args.data[0],batch_size=self.args.batchsize,shuffle=True,num_workers=2)
        train__loader = DataLoader(dataset=self.args.data[0],batch_size=10,shuffle=True,num_workers=2)
        test__loader = DataLoader(dataset=self.args.data[1],batch_size=10,shuffle=True,num_workers=2)

        self.model.eval()
        tri_acc1 = tri_test(self.model,train__loader,self.args.device)
        tri_acc2 = tri_test(self.model,test__loader,self.args.device)
        print('initi train tri_acc:{},init test tri_acc:{}'.format(tri_acc1,tri_acc2))

        for a,p,n in triplet_loader:
            count += 1
            self.model.train()
                
            a,p,n = a.to(self.args.device),p.to(self.args.device),n.to(self.args.device)
            self.optimizer1.zero_grad()

            _,anchor = self.model(a)
            _,positive = self.model(p)
            _,negative = self.model(n)

            loss,d_ap,d_an = triplet_loss(anchor,positive,negative,alpha)
            if loss!=0:
                loss.backward()
                self.optimizer1.step()
            else:
                print("loss=0")

            if count% 20 == 0:
                print('loss:{},dap:{},dan:{}\n'.format(loss,d_ap,d_an))
            
            if count%100 == 0:
                self.model.eval()
                tri_acc1 = tri_test(self.model,test__loader,self.args.device)
                tri_acc2 = tri_test(self.model,train__loader,self.args.device)
                print('accuracy on train:{}, accuracy on test:{}\n'.format(tri_acc2,tri_acc1))
                torch.save(self.model.state_dict(),'./models/inference/trinet{}.pkl'.format(int(count)))


if __name__ == '__main__':

    # args = Args(seed=1001,lr=1e-4, bs=32,clss=10,model_path='./models/aux/adam/animal/dsl/0.pkl', optimizer='adam', 
    #             data=[a_train_triplets,a_test_triplets])

    args = Args(seed=1001,lr=1e-4, bs=32,clss=8,model_path='./models/aux/adam/vehicle/dsl/0.pkl', optimizer='adam', 
                data=[v_train_triplets,v_test_triplets])
    trainer = Trainer(args)

    trainer.model_adjusting()