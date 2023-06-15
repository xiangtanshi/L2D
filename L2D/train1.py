from torch import optim
from utils.loss import *
from utils.dataset_class import *
# from utils.triplet_class import train_triplets
from utils.resnet import *
from utils.Convnet import *
from torch.utils.data import  DataLoader
from torchvision.utils import make_grid
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class Args:
    def __init__(self):

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        self.learning_rate = 1e-3
        self.batchsize = 128
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.epoch = 4
        self.classes = 10
        

class Trainer:

    def __init__(self, args):

        self.args = args

        model = ResNet(num_classes=args.classes)
        model.load_state_dict(torch.load('./models/resnet/nico_animal.pkl'))

        # three ways of runing models on gpu: 1 gpu; multi parallel; multi-distributed parallel

        # self.model = to_device(model,args.device)

        model = nn.DataParallel(model)
        self.model = model.cuda()

        # torch.distributed.init_process_group(backend='nccl',init_method='tcp://localhost:23456',rank=0,world_size=1)
        # self.model = model.cuda()
        # self.model = nn.parallel.DistributedDataParallel(self.model)

        if isinstance(self.model,nn.parallel.DistributedDataParallel):
            print('true')

        self.optimizer1 = optim.SGD(self.model.parameters(), args.learning_rate,
                                 momentum=args.momentum, weight_decay=args.weight_decay)
        # self.optimizer1 = optim.Adam(self.model.parameters(),lr=args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer1,step_size=14,gamma=0.1)

        self.train_loader = DataLoader(dataset=a_ani_train, batch_size=args.batchsize, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(dataset=a_ani_valid, batch_size=args.batchsize, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(dataset=a_ani_test, batch_size=args.batchsize, shuffle=False, num_workers=4)
        self.test_loaders = {'val':self.val_loader,'test':self.test_loader}
        # self.tri_train_loader = DataLoader(dataset=train_triplets, batch_size=args.batchsize,shuffle=True,num_workers=4)
        self.results = {'val': torch.zeros(self.args.epoch), 'test': torch.zeros(self.args.epoch)}

    def testing(self, loader):
        positive_prediction = 0
        total = 0
        for data in loader:
            images,labels = data
            # images, labels = images.to(self.args.device), labels.to(self.args.device)
            images, labels = images.cuda(), labels.cuda()
            logits,_ = self.model(images)
            _,pred = logits.max(dim=1)
            positive_prediction += torch.sum(pred == labels)
            total += labels.size(0)
        
        acc = float(positive_prediction)/total
        return acc

    def training(self):

        # writer = SummaryWriter('result_records/runs')
        count = 0
        seed_torch(seed=1001)
        bar = tqdm(total=self.args.epoch)
        for itr in range(self.args.epoch):
            self.model.train()
            
            # for ccl learning
            '''
            for a,p,n,labels in self.tri_train_loader:
                
                a,p,n,labels = a.to(self.args.device),p.to(self.args.device),n.to(self.args.device),labels.to(self.args.device)
                self.optimizer1.zero_grad()
                logits,_ = self.model(a)
                loss_cls = loss(logits,labels)

                _,anchor = self.model(a)
                _,positive = self.model(p)
                _,negative = self.model(n)
                loss_contrast = contrastive_loss(anchor,positive,negative)
                Loss = loss_cls + 0.1*loss_contrast
                Loss.backward()
                self.optimizer1.step()      
            '''
            
            for images,labels in self.train_loader:

                # for normal model
                
                # images, labels = images.to(self.args.device), labels.to(self.args.device)
                images, labels = images.cuda(), labels.cuda()
                self.optimizer1.zero_grad()
                logits,_ = self.model(images)
                Loss = loss(logits, labels)
                Loss.backward()
                self.optimizer1.step()
                print(Loss)

                # writer.add_scalar('traning loss',Loss,count)
                count += 1
                

                # for rsc 
                
                # images, labels = images.to(self.args.device), labels.to(self.args.device)
                # self.optimizer1.zero_grad()

                # images_flip = torch.flip(images,(3,)).detach().clone()
                # images = torch.cat((images,images_flip))
                # labels = torch.cat((labels,labels))

                # logits,_ = self.model(images,labels,True,itr,args.device)
                # Loss = loss(logits,labels)
                # Loss.backward()
                # self.optimizer1.step()
                

                # for cnbb
                
                # batch_size = labels.size(0)
                # self.optimizer1.zero_grad()
                # images, labels = images.to(self.args.device), labels.to(self.args.device)
                # logits, features = self.model(images)

                # Weight = torch.ones((batch_size,1))/batch_size
                # updation = 0
                # while(updation<2):
                #     Weight.requires_grad = False
                #     Weight.clamp_(0.0001)
                #     Weight = Weight/sum(Weight)
                #     Weight.requires_grad = True
                #     weight = Weight.to(self.args.device)

                #     loss_b1 = lossb(features.data,weight,512,self.args.device)
                #     norm = 1e3*torch.norm(weight)**2
                #     loss_b = loss_b1 + norm
                #     loss_b.backward()
                #     grad = max(torch.abs(Weight.grad))

                #     rate = 0.006/float(grad)
                #     optimizer2 = optim.SGD([Weight], lr=rate)
                #     optimizer2.step()
                #     optimizer2.zero_grad()
                #     updation += 1
                
                # Weight.requires_grad = False
                # Weight.clamp_(0.0001)
                # Weight = Weight/sum(Weight)
                # weight = Weight.to(self.args.device)

                # loss_c = lossc(logits,labels,weight)
                # loss_q = 0.0001*lossq(features)
                # loss_p = loss_c + loss_q
                # loss_p.backward()
                # self.optimizer1.step()
                
            # img_grid = make_grid(images)
            # writer.add_image('minibatch',img_grid)
            # writer.add_graph(self.model,images)

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
                    torch.save(self.model.module.state_dict(),'./models/model_1_{}.pkl'.format(int(itr)))
                else:
                    torch.save(self.model.state_dict(),'./models/model_1_{}.pkl'.format(int(itr)))
   
            
        # writer.close()
        bar.close()
        best_index = self.results['val'].argmax()
        print(self.results)
        print('best val %g in epoch %g, corresponding test %g, best test %g' % (self.results['val'][best_index],
        best_index, self.results['test'][best_index],self.results['test'].max()))


if __name__ == '__main__':

    args = Args()
    trainer = Trainer(args)
    trainer.training()
