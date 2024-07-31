from torch import optim
from utils.dataset_class import *
from utils.loss import *
from utils.resnet import *
from utils.Convnet import *
from torch.utils.data import  DataLoader
import argparse
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='the hyperparamers for training a resnet classifier on NICO.')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--bs', type=int, default=128, help='training batchsize')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--ep', type=int, default=20, help='training epoch, 20 for resnet-18 and 10 for resnet-50')
parser.add_argument('--clss', type=int, default=17, help='number of classes')
parser.add_argument('--flag', type=int, default=1)
parser.add_argument('--d', type=int, default=18, help='choose resent18 or resnet50')
parser.add_argument('--method', type=str, default='normal', help='training paradigm', choices=['normal','rsc','adv'])
parser.add_argument('--optim', type=str, default='sgd', help='optimizer used for training')

args = parser.parse_args()
        

class Trainer:

    def __init__(self, args):

        self.args = args
        # load the models with weights that are pretrained on imagenet
        if self.args.d == 50:
            model = ResNet(num_classes=17, block=Bottleneck, layers=[3, 4, 6, 3])
            model.load_state_dict(torch.load('./models/resnet50/nico.pkl')) 
        elif self.args.d == 18:
            model = ResNet(num_classes=17, block=BasicBlock, layers=[2, 2, 2, 2])
            model.load_state_dict(torch.load('./models/resnet18/nico.pkl')) 

        self.model = model.cuda()
        
        if args.optim == 'sgd':
            self.optimizer1 = optim.SGD(self.model.parameters(), args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            self.optimizer1 = optim.Adam(self.model.parameters(),lr=args.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer1,step_size=14,gamma=0.1)

        # load the data when training
        if args.flag == 1:
            self.train_loader = DataLoader(dataset=train1, batch_size=args.bs, shuffle=True, num_workers=4)
            self.val_loader = DataLoader(dataset=val1, batch_size=args.bs, shuffle=False, num_workers=4)
            self.test_loader = DataLoader(dataset=test1, batch_size=args.bs, shuffle=False, num_workers=4)
            print('data prepared, the number of samples in the train, val, test tests are {},{},{}.'.format(len(train1),len(val1),len(test1)))
        elif args.flag == 2:
            self.train_loader = DataLoader(dataset=train2, batch_size=args.bs, shuffle=True, num_workers=4)
            self.val_loader = DataLoader(dataset=val2, batch_size=args.bs, shuffle=False, num_workers=4)
            self.test_loader = DataLoader(dataset=test2, batch_size=args.bs, shuffle=False, num_workers=4)
            print('data prepared, the number of samples in the train, val, test tests are {},{},{}.'.format(len(train2),len(val2),len(test2)))

        self.test_loaders = {'train': self.train_loader, 'val':self.val_loader, 'test':self.test_loader}
        self.results = {'train': torch.zeros(self.args.ep), 'val': torch.zeros(self.args.ep), 'test': torch.zeros(self.args.ep)}

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

    def training(self):

        seed_torch(seed=self.args.seed)
        
        for itr in range(self.args.ep):
            self.model.train()
            
            for images,labels in tqdm(self.train_loader, desc=f"Epoch {itr+1}/{self.args.ep}"):

                
                if self.args.method == 'normal':
                    # for ERM
                    images, labels = images.cuda(), labels.cuda()
                    self.optimizer1.zero_grad()
                    logits,_ = self.model(images)
                    Loss = loss(logits, labels)
                    Loss.backward()
                    self.optimizer1.step()

                elif self.args.method == 'rsc':
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
                
                elif self.args.method == 'adv':
                    # for deep ensembling
                    images, labels = images.cuda(), labels.cuda()
                    self.optimizer1.zero_grad()
                    images.requires_grad_(True)
                    logits,_ = self.model(images)
                    Loss = loss(logits, labels)
                    Loss.backward()

                    # adversarial training
                    epsilon = 0.01
                    adversarial_images = images + epsilon * images.grad.sign()
                    logits_adv,_ = self.model(adversarial_images)
                    Loss_adv = loss(logits_adv, labels)
                    Loss_adv.backward()
                    
                    self.optimizer1.step()

            if self.args.optim == 'adam':
                self.scheduler.step()
            # eval the model after a total epoch
            self.model.eval()
            with torch.no_grad():
                for phase, loader in self.test_loaders.items():
                    acc = self.testing(loader)
                    self.results[phase][itr] = acc
                print('epoch:{},train acc={}, val acc={}, test acc={}'.format(itr+1,self.results['train'][itr],self.results['val'][itr],self.results['test'][itr]))

        if self.args.d == 50:
            torch.save(self.model.state_dict(),'./models/resnet50/nico-{}-{}-{}.pkl'.format(self.args.method, self.args.seed, self.args.optim))
        elif self.args.d == 18:
            torch.save(self.model.state_dict(),'./models/resnet18/nico-{}-{}-{}.pkl'.format(self.args.method, self.args.seed, self.args.optim))

        best_index = self.results['val'].argmax()
        print(self.results)
        print('best val %g in epoch %g, corresponding test %g, best test %g' % (self.results['val'][best_index],
        best_index, self.results['test'][best_index],self.results['test'].max()))

    def evaluating(self):

        if self.args.method == 'adv':
            seeds = [0,1001,2001,3001,4001]
            result = []
            for i in range(5):
                if self.args.d == 50:
                    model = ResNet(num_classes=17, block=Bottleneck, layers=[3, 4, 6, 3])
                    model.load_state_dict(torch.load('./models/resnet50/nico-adv-{}.pkl'.format(seeds[i]))) 
                elif self.args.d == 18:
                    model = ResNet(num_classes=17, block=BasicBlock, layers=[2, 2, 2, 2])
                    model.load_state_dict(torch.load('./models/resnet18/nico-adv-{}.pkl'.format(seeds[i]))) 
                model = model.cuda()
                model.eval()
                tmp_logit = torch.zeros(14564,17)
                tmp_label = torch.zeros(14564)
                idx = 0
                with torch.no_grad():
                    for data in self.test_loader:
                        images, labels = data
                        images = images.cuda()
                        logits,_ = model(images)
                        logits.cpu()
                        tmp_logit[idx:int(idx+len(labels)),:] = logits
                        tmp_label[idx:int(idx+len(labels))] = labels
                        idx += len(labels)
                result.append(tmp_logit)
            ave_logit = torch.sum(torch.stack(result), dim=0)
            _,pred = ave_logit.max(dim=1)
            positive_prediction = torch.sum(pred == tmp_label)
            total = tmp_label.size(0)
            acc = float(positive_prediction)/total
            print('Accuracy on test set:{}'.format(acc))
                


if __name__ == '__main__':

    trainer = Trainer(args)
    trainer.training()
    # trainer.evaluating()
