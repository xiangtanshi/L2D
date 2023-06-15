from torch import optim
from utils.loss1 import *
from utils.logit_dataset import *
from utils.Convnet import *
from torch.utils.data import  DataLoader
import torch


class Args:
    def __init__(self, seed = 1013, lr = 1e-2, bs = 64, ep = 20, clss = 10, hard = False,
                    model_path = './models/inference/pattern_ani.pkl', data = [a_valid_logit,a_test_logit]):

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.seed = seed
        self.learning_rate = lr
        self.batchsize = bs
        self.epoch = ep
        self.classes = clss
        self.model_path = model_path
        self.logits = data
        self.hard = hard      # set as True when need to calculating accuracy on hard samples
        self.results = {'val': torch.zeros(self.epoch), 'test': torch.zeros(self.epoch)}



def accuracy(args, network=None, dataloader=None):
   
    positive_prediction = 0
    total = 0
    f2t = 0
    t2f = 0
    f2f = 0

    if args.hard:
        h_f2t = 0
        h_t2f = 0
        h_positive = 0
        h_total = 0
        

    for index, item in enumerate(dataloader):
        # item[0] is 1Dim logits
        item[0] = item[0].squeeze(1)
        item[0] = item[0].to(args.device)
        item[1] = item[1].to(args.device)
        results = network(item[0],1)
        pred = results.argmax()

        faith = item[0][0,0,args.classes] == max(item[0][0,0,args.classes:])

        if faith:
            # do not change the original prediction
            if item[1] == 0:
                positive_prediction += 1
        else:
            # predict with the revision net
            if item[1] == pred:
                positive_prediction += 1

                # positive correction
                if pred != 0:
                    f2t += 1
                    
            else:
                if item[1] == 0 and pred != 0:
                    # negative revision
                    t2f += 1
                
                elif item[1] != 0 and pred != 0:
                    # revise from a false prediction to another one
                    f2f += 1
                
        total += 1

        if args.hard:
            # calculate accuracy on hard samples
            if item[0][0,0,0]<0.9 and (not faith):

                h_total += 1
                if item[1] == pred:
                    h_positive += 1
                    if pred != 0:
                        h_f2t += 1
                if item[1] == 0 and pred != 0:
                    h_t2f += 1
                    
                                
    accu = positive_prediction/total
    
    if args.hard:
        hard_accu = h_positive/h_total
        print("f2t:{},t2f{},improve:{},h_acc:{}".format(h_f2t,h_t2f,(h_f2t-h_t2f)/h_total,hard_accu))

    print('f2t:{},t2f:{}'.format(f2t,t2f), end=',')
    return accu


def main(args = None):
    '''
    train the small network which serves the purpose of fusion of logits from true data and counterfactual features
    '''
    seed_torch(args.seed)

    target = args.logits

    # load the data
    test_loader = DataLoader(dataset=target[1], batch_size=args.batchsize, shuffle=True, num_workers=4)    
    valid_loader = DataLoader(dataset=target[0], batch_size=args.batchsize, shuffle=True, num_workers=4)   

    valid_loader_ = DataLoader(dataset=target[0], batch_size=1, shuffle=False, num_workers=1)    
    test_loader_ = DataLoader(dataset=target[1], batch_size=1, shuffle=False, num_workers=1)   

    device = args.device

    net = Pattern(classes=args.classes)
    net.load_state_dict(torch.load(args.model_path))                         
    net = to_device(net,device)

    if args.hard:
        net.eval()
        accuracy(args,net,test_loader_)
        return
    
    optimizer1 = optim.Adam(net.parameters(),lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer1,step_size=15,gamma=0.1)

    for itr in range(args.epoch):

        for features, label in valid_loader:
            #check the batchsize
            batch_size = label.shape[0]
            net.train()
            optimizer1.zero_grad()

            #the transform operation will add an additional dimension, squeeze this dimension first
            features = features.squeeze(1)
            features, label = to_device(features,device), to_device(label,device)
            probs = net(features,batch_size)

            Loss = loss(probs,label)
            L = Loss 
            
            L.backward()
            optimizer1.step()

        net.eval()

        print("epoch:{}".format(itr),end='  ')
        Accuracy = accuracy(args,net,valid_loader_)
        args.results['val'][itr] = Accuracy
        print('valid_accuracy:{}'.format(Accuracy),end=',')

        Accuracy= accuracy(args,net,test_loader_)
        args.results['test'][itr] = Accuracy
        print('test_accuracy:{}'.format(Accuracy))\

        scheduler.step()
        torch.save(net.state_dict(),'./models/inference/revision{}.pkl'.format(itr))  

    print(args.results)


if __name__ == '__main__':
    args = Args(seed=1001,clss=10,hard=False,model_path='./models/inference/pattern_ani.pkl',data=[a_valid_logit,a_test_logit])
    main(args)
