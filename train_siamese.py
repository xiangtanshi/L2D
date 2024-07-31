from torch import optim
from utils.loss import *
from utils.dataset_class import *
from triplet_class import *
from utils.resnet import *
from utils.Convnet import *
from torch.utils.data import  DataLoader
import torch
import numpy as np
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='parameter setting for training the siamese network.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--d', type=int, default=18)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--eval_num', type=int, default=50, help='the number of triplets tested in one batch')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--optim', type=str, default='adam')
args =  parser.parse_args()


class Trainer:

    def __init__(self, args):

        self.args = args
        if self.args.d == 50:
            model = ResNet(num_classes=17, block=Bottleneck, layers=[3, 4, 6, 3])
            model.load_state_dict(torch.load('./models/resnet50/nico-normal-0-{}.pkl'.format(args.optim))) 
        elif self.args.d == 18:
            model = ResNet(num_classes=17, block=BasicBlock, layers=[2, 2, 2, 2])
            model.load_state_dict(torch.load('./models/resnet18/nico-normal-0-{}.pkl'.format(args.optim))) 

        self.model = to_device(model,args.device)

        # it is interesting that the siamese network cannot be trained with sgd, only adam is suitable for the triplet loss
        if args.optimizer == 'sgd':
            self.optimizer1 = optim.SGD(self.model.parameters(), args.learning_rate,
                                    momentum=0.9, weight_decay=5e-4)
        else:
            self.optimizer1 = optim.Adam(self.model.parameters(),lr=args.learning_rate)


    def model_adjusting(self):
        seed_torch(seed=self.args.seed)
        # training the siamese network
        count = 0
        alpha = 0.4

        triplet_loader = DataLoader(dataset=siamese_train,batch_size=self.args.batchsize,shuffle=True,num_workers=4)
        train__loader = DataLoader(dataset=siamese_train,batch_size=self.args.eval_num,shuffle=True,num_workers=2)
        test__loader = DataLoader(dataset=siamese_test,batch_size=self.args.eval_num,shuffle=True,num_workers=2)

        self.model.eval()
        tri_acc1 = tri_test(self.model,train__loader,self.args.device,self.args.eval_num)
        tri_acc2 = tri_test(self.model,test__loader,self.args.device,self.args.eval_num)
        print('initi train tri_acc:{},init test tri_acc:{}'.format(tri_acc1,tri_acc2))

        for a,p,n in tqdm(triplet_loader):
            count += 1
            self.model.train()
                
            a,p,n = a.to(self.args.device),p.to(self.args.device),n.to(self.args.device)
            self.optimizer1.zero_grad()

            _,anchor = self.model(a)
            _,positive = self.model(p)
            _,negative = self.model(n)

            loss,d_ap,d_an = triplet_loss_(anchor,positive,negative,alpha)
            if loss!=0:
                loss.backward()
                self.optimizer1.step()
            else:
                print("loss=0")

            if count% 20 == 1:
                print('loss:{},dap:{},dan:{}\n'.format(loss,d_ap,d_an))
            
            if count%100 == 0:
                self.model.eval()
                tri_acc1 = tri_test(self.model,test__loader,self.args.device,self.args.eval_num)
                tri_acc2 = tri_test(self.model,train__loader,self.args.device,self.args.eval_num)
                print('accuracy on train:{}, accuracy on test:{}\n'.format(tri_acc2,tri_acc1))

                if self.args.d == 50:
                    torch.save(self.model.state_dict(),'./models/resnet50/nico-siamese-{}-{}-{}.pkl'.format(self.args.seed,count,self.args.optim))
                elif self.args.d == 18:
                    torch.save(self.model.state_dict(),'./models/resnet18/nico-siamese-{}-{}-{}.pkl'.format(self.args.seed,count,self.args.optim))


if __name__ == '__main__':


    trainer = Trainer(args)

    trainer.model_adjusting()