from torch import optim
from utils.loss1 import *
from utils.dataset_class import *
from utils.triplet_class import *
from utils.resnet import *
from utils.Convnet import *
from torch.utils.data import  DataLoader
import torch
import numpy as np
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Args:
    def __init__(self, seed = 0, lr = 1e-3, bs = 128, momentum = 0.9, weight_decay = 5e-4, ep = 20, clss = 8, 
                model_path = './models/resnet/nico_vehicle.pkl', optimizer = 'adam', 
                data = [a_vel_train, a_vel_valid, a_vel_test], load_flag = True):

        self.seed = seed
        self.learning_rate = lr
        self.batchsize = bs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epoch = ep
        self.classes = clss
        self.model_path = model_path
        self.optimizer = optimizer
        self.data = data
        self.load_flag = load_flag
        

class Trainer:

    def __init__(self, args):

        self.args = args

        # load the models with weights that are pretrained on imagenet
        model = ResNet(num_classes=args.classes)
        model.load_state_dict(torch.load(self.args.model_path))
        self.model = model.cuda()
        # self.model = nn.DataParallel(self.model)
        

        if args.optimizer == 'sgd':
            self.optimizer1 = optim.SGD(self.model.parameters(), args.learning_rate,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            self.optimizer1 = optim.Adam(self.model.parameters(),lr=args.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer1,step_size=14,gamma=0.1)

        # load the data when training
        if args.load_flag:

            self.train_loader = DataLoader(dataset=args.data[0], batch_size=args.batchsize, shuffle=True, num_workers=4)
            self.val_loader = DataLoader(dataset=args.data[1], batch_size=args.batchsize, shuffle=True, num_workers=4)
            self.test_loader = DataLoader(dataset=args.data[2], batch_size=args.batchsize, shuffle=True, num_workers=4)
            self.test_loaders = {'val':self.val_loader,'test':self.test_loader}
            self.results = {'val': torch.zeros(self.args.epoch), 'test': torch.zeros(self.args.epoch)}

    def testing(self, loader):
        positive_prediction = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                images,labels = data
                images, labels = images.cuda(), labels.cuda()
                logits,_ = self.model(images)
                _,pred = logits.max(dim=1)
                positive_prediction += torch.sum(pred == labels)
                total += labels.size(0)
        
        acc = float(positive_prediction)/total
        return acc


    def training(self, method = 'normal'):

        seed_torch(seed=self.args.seed)
        bar = tqdm(total=self.args.epoch)
        for itr in range(self.args.epoch):
            self.model.train()
            
            for images,labels in self.train_loader:

                
                if method == 'normal':
                    # for ResNet-18, CGN
                    images, labels = images.cuda(), labels.cuda()
                    self.optimizer1.zero_grad()
                    logits,_ = self.model(images)
                    Loss = loss(logits, labels)
                    Loss.backward()
                    self.optimizer1.step()

                elif method == 'rsc':
                    # for rsc 
                
                    images, labels = images.cuda(), labels.cuda()
                    self.optimizer1.zero_grad()

                    images_flip = torch.flip(images,(3,)).detach().clone()
                    images = torch.cat((images,images_flip))
                    labels = torch.cat((labels,labels))

                    logits,_ = self.model(images,labels,True,itr)
                    Loss = loss(logits,labels)
                    Loss.backward()
                    self.optimizer1.step()
                
                else:

                    # for cnbb
                
                    batch_size = labels.size(0)
                    self.optimizer1.zero_grad()
                    images, labels = images.cuda(), labels.cuda()
                    logits, features = self.model(images)

                    Weight = torch.ones((batch_size,1))/batch_size
                    updation = 0
                    while(updation<2):
                        Weight.requires_grad = False
                        Weight.clamp_(0.0001)
                        Weight = Weight/sum(Weight)
                        Weight.requires_grad = True
                        weight = Weight.cuda()

                        loss_b1 = lossb(features.data,weight,512)
                        norm = 1e4*torch.norm(weight)**2
                        loss_b = loss_b1 + norm
                        loss_b.backward()
                        grad = max(torch.abs(Weight.grad))

                        rate = 0.01/float(grad)
                        optimizer2 = optim.SGD([Weight], lr=rate)
                        optimizer2.step()
                        optimizer2.zero_grad()
                        updation += 1
                    
                    Weight.requires_grad = False
                    Weight.clamp_(0.0001)
                    Weight = Weight/sum(Weight)
                    weight = Weight.cuda()

                    loss_c = lossc(logits,labels,weight)
                    # loss_q = 0.0001*lossq(features)
                    loss_p = loss_c # + loss_q
                    loss_p.backward()
                    self.optimizer1.step()

            # eval the model after a total epoch
            self.model.eval()
            with torch.no_grad():
                for phase, loader in self.test_loaders.items():
                    acc = self.testing(loader)
                    print(acc)
                    self.results[phase][itr] = acc
            self.scheduler.step()
            bar.update()

            if itr>14:
                if isinstance(self.model, nn.DataParallel) or isinstance(self.model, nn.parallel.DistributedDataParallel):
                    # multipul gpu
                    torch.save(self.model.module.state_dict(),'./models/inference/model_3_{}.pkl'.format(int(itr)))
                else:
                    torch.save(self.model.state_dict(),'./models/inference/model_3_{}.pkl'.format(int(itr)))

        bar.close()
        best_index = self.results['val'].argmax()
        print(self.results)
        print('best val %g in epoch %g, corresponding test %g, best test %g' % (self.results['val'][best_index],
        best_index, self.results['test'][best_index],self.results['test'].max()))


if __name__ == '__main__':

    args = Args(seed=1001,bs=128, clss=8, optimizer='adam', data=[a_vel_train, a_vel_valid, a_vel_test])

    # train classifiers
    model_path='./models/resnet/nico_vehicle.pkl'
    args.lr = 1e-3
    trainer = Trainer(args)
    trainer.training(method='rsc')
