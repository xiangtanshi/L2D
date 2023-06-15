import argparse
import os
import random
import shutil
import time
import warnings
from utils.loss import *
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from utils.dataset_class import *
from utils.resnet import *
import models
import utils.dsl_loss as loss_expect
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# from torch.utils.tensorboard import SummaryWriter

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-data', metavar='DIR', default='/data/dengx/Animal',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_with_table',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--log_path',
                    default='./result_records/log.txt', type=str, metavar='PATH',
                    help='path to save logs (default: none)')

parser.add_argument ('--num_f', type=int, default=1, help = 'number of fourier spaces')

# for random sampling
parser.add_argument ('--sample_rate', type=float, default=1.0, help = 'sample ratio of the features involved in balancing')

parser.add_argument ('--lrbl', type = float, default = 1.0, help = 'learning rate of balance')

# parser.add_argument ('--cfs', type = int, default = 512, help = 'the dim of each feature')
parser.add_argument ('--lambdap', type = float, default = 2.0, help = 'weight decay for weight1 ')
parser.add_argument ('--lambdapre', type = float, default = 1, help = 'weight for pre_weight1 ')

parser.add_argument ('--epochb', type = int, default = 20, help = 'number of epochs to balance')
parser.add_argument ('--epochp', type = int, default = 0, help = 'number of epochs to pretrain')

# for table
parser.add_argument ('--n_feature', type=int, default=128, help = 'number of pre-saved features')
parser.add_argument ('--feature_dim', type=int, default=512, help = 'the dim of each feature')

parser.add_argument ('--lrwarmup_epo', type=int, default=0, help = 'the dim of each feature')
parser.add_argument ('--lrwarmup_decay', type=int, default=0.1, help = 'the dim of each feature')

parser.add_argument ('--n_levels', type=int, default=1, help = 'number of global table levels')

# for expectation
parser.add_argument ('--lambda_decay_rate', type=float, default=1, help = 'ratio of epoch for lambda to decay')
parser.add_argument ('--lambda_decay_epoch', type=int, default=5, help = 'number of epoch for lambda to decay')
parser.add_argument ('--min_lambda_times', type=float, default=0.01, help = 'number of global table levels')

parser.add_argument ('--renew_ratio', type=float, default=0.1, help = 'number of global table levels')

# for jointly train
parser.add_argument ('--train_cnn_with_lossb', type=bool, default=False, help = 'whether train cnn with lossb')
parser.add_argument ('--cnn_lossb_lambda', type=float, default=0, help = 'lambda for lossb')

# for more moments
parser.add_argument ('--moments_lossb', type=float, default=1, help = 'number of moments')

# for first step
parser.add_argument ('--first_step_cons', type=float, default=1, help = 'constrain the weight at the first step')

# for pow
parser.add_argument ('--decay_pow', type=float, default=2, help = 'value of pow for weight decay')

# for second order moment weight
parser.add_argument ('--second_lambda', type=float, default=0.2, help = 'weight lambda for second order moment loss')
parser.add_argument ('--third_lambda', type=float, default=0.05, help = 'weight lambda for second order moment loss')

# for dat/.a aug
parser.add_argument ('--lower_scale', type=float, default=0.5, help = 'weight lambda for second order moment loss')

# for lr decay epochs
parser.add_argument ('--epochs_decay', type=list, default=[15, 20], help = 'weight lambda for second order moment loss')

best_acc1 = 0


def get_weight(cfeatures, pre_features, pre_weight1, args, global_epoch=0, iter=0):
    # TODO write a feature save features with size of batchsize, and weight sum them
    # TODO how to deal with the first several iteration?
    # TODO should add a weight between new feature and old feature when concat them to calculate the weight
    # all iteration for one epoch is 9000 / 256 = 36, so the weight for new ones should be 0.1 or 0.2

    softmax = nn.Softmax(0)
    weight = Variable(torch.ones(cfeatures.size()[0], 1).cuda())

    cfeaturec = Variable(torch.FloatTensor(cfeatures.size()).cuda())
    cfeaturec.data.copy_(cfeatures.data)

    pre_features=pre_features.cuda()
    pre_weight1=pre_weight1.cuda()

    all_feature = torch.cat([cfeaturec, pre_features.detach()], dim=0)
    weight.requires_grad = True
    optimizerbl = torch.optim.SGD([weight], lr=args.lrbl, momentum=0.9)   # TODO define the lrbl
    lossb0 = 0.0
    lossb1 = 0.0
    lossg0 = 0.0
    lossg1 = 0.0
    lambdap0 = 0.0
    if args.moments_lossb == 1:
        lossb_func = loss_expect

    else:
        print("ERROR: MOMENTS FOR LOSSB CAN NOT BIGGER THAN 2!")

    for epoch in range(args.epochb):
        adjust_learning_rate_bl(optimizerbl, epoch, args)
        all_weight = torch.cat((weight, pre_weight1.detach()), dim=0)
        optimizerbl.zero_grad()

        if args.moments_lossb > 1:
            lossb = lossb_func.lossb_expect(all_feature, softmax(all_weight), args.feature_dim, args.moments_lossb,
                                            args.second_lambda, args.third_lambda)
        else:
            lossb = loss_expect.lossb_expect(all_feature, softmax(all_weight), args.feature_dim, args.num_f, args.sample_rate)

        lossp = softmax(weight).pow(args.decay_pow).sum()
        lambdap = args.lambdap * max((args.lambda_decay_rate ** (global_epoch // args.lambda_decay_epoch)),
                                     args.min_lambda_times)
        lambdap0 = lambdap
        lossg = lossb / lambdap + lossp # TODO define the lambdap
        if global_epoch == 0:
            lossg = lossg * args.first_step_cons

        lossg.backward(retain_graph=True)
        if epoch == 0:
            lossb0 = lossb
            lossg0 = lossg
        if epoch == args.epochb - 1:
            lossb1 = lossb
            lossg1 = lossg
        optimizerbl.step()
    all_feature_tach = torch.cat([cfeatures, pre_features.detach()], dim=0)
    lossb_out = lossb_func.lossb_expect(all_feature_tach, softmax(all_weight), args.feature_dim, args.num_f, args.sample_rate)

    if global_epoch == 0 and iter < 10:
        pre_features = (pre_features * iter + cfeatures) / (iter + 1)
        pre_weight1 = (pre_weight1 * iter + weight) / (iter + 1)
    elif cfeatures.size()[0] < pre_features.size()[0]:
        pre_features[:cfeatures.size()[0]] = pre_features[:cfeatures.size()[0]] * (1 - args.renew_ratio) + cfeatures * args.renew_ratio
        pre_weight1[:cfeatures.size()[0]] = pre_weight1[:cfeatures.size()[0]] * (1 - args.renew_ratio) + weight * args.renew_ratio
    else:
        pre_features = pre_features * (1 - args.renew_ratio) + cfeatures * args.renew_ratio
        pre_weight1 = pre_weight1 * (1 - args.renew_ratio) + weight * args.renew_ratio

    lossb_diff = lossb0 - lossb1
    lossg_diff = lossg0 - lossg1
    softmax_weight = softmax(weight)
    weight_diff = torch.max(softmax_weight) - torch.min(softmax_weight)
    weight_var = torch.var(softmax_weight)
    dis = torch.sum(torch.abs(softmax_weight-softmax(torch.ones(weight.size()).cuda())))
    dis_ori = torch.sum(torch.abs(softmax(torch.ones(weight.size()).cuda() - softmax(torch.zeros(weight.size()).cuda()))))
    return softmax_weight, lossb0, lossb1, lossp, pre_features, pre_weight1, lossb_diff, lossg_diff, \
           weight_diff, weight_var, lambdap0, lossb_out, dis, dis_ori

def main():
    args = parser.parse_args()
    args.pretrained = True
    if not os.path.exists(os.path.dirname(args.log_path)):
        os.makedirs(os.path.dirname(args.log_path))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    model = ResNet(num_classes=10)
    model.load_state_dict(torch.load('./models/resnet/nico_animal.pkl'))


    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_train = nn.CrossEntropyLoss(reduce=False).cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=14,gamma=0.1)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(a_ani_train)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        a_ani_train, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        a_ani_valid,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        a_ani_test,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    log_dir = os.path.dirname(args.log_path)
    # print('tensorboard dir {}'.format(log_dir))
    tensor_writer =  None #  SummaryWriter(log_dir)

    if args.evaluate:
        # validate(val_loader, model, criterion, 0, args, tensor_writer)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, criterion_train, optimizer, epoch, args, tensor_writer)
        scheduler.step()
        val_acc = testing(val_loader,torch.device('cuda:0'),model)
        test_acc = testing(test_loader,torch.device('cuda:0'),model)
        print("valid acc:{},test acc:{}".format(val_acc,test_acc))
        if epoch >14:
            torch.save(model.state_dict(),'./models/inference/model_dsl_{}.pkl'.format(int(epoch)))

        # acc1 = validate(val_loader, model, criterion, epoch, args, tensor_writer)

        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)

        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank % ngpus_per_node == 0):
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best, args.log_path)


def train(train_loader, model, criterion, optimizer, epoch, args, tensor_writer=None):
    ''' TODO write a dict to save previous featrues  check vqvae,
        the size of each feature is 512, os we need a tensor of 1024 * 512
        replace the last one every time
        and a weight with size of 1024,
        replace the last one every time
        TODO init the tensors
    '''

    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')

    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    model.train()
    # end = time.time()
    for i, (images, target) in enumerate(train_loader):
        batchsizes = target.size(0)
        if batchsizes < args.batch_size:
            continue

        # data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        output, cfeatures = model(images)
        pre_features = model.pre_features
        pre_weight1 = model.pre_weight1

        if epoch >= args.epochp:
            weight1, lossb1, lossb2, lossp, pre_features, pre_weight1, diff_lossb, diff_lossg, diff_weight, var_weight,\
            lambdap0, lossb_out, dis, dis_ori = get_weight(cfeatures, pre_features, pre_weight1, args, epoch, i)

        else:
            weight1 = Variable(torch.ones(cfeatures.size()[0], 1).cuda())

        model.pre_features.data.copy_(pre_features)
        model.pre_weight1.data.copy_(pre_weight1)
        loss = criterion(output, target).view (1, -1).mm(weight1).view(1)
        if args.train_cnn_with_lossb:
            loss += args.cnn_lossb_lambda * lossb_out

        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # batch_time.update(time.time() - end)
        # end = time.time()

        # method_name = args.log_path.split('/')[-2]
        # if i % args.print_freq == 0:
        #     progress.display(i, method_name)
        #     progress.write_log(i, args.log_path)
        


    # tensor_writer.add_scalar('loss/train', losses.avg, epoch)
    # tensor_writer.add_scalar('ACC@1/train', top1.avg, epoch)
    # tensor_writer.add_scalar('ACC@5/train', top5.avg, epoch)
    # tensor_writer.add_scalar('lossb1', lossb1.sum(), epoch)
    # tensor_writer.add_scalar('lossb2', lossb2.sum(), epoch)
    # tensor_writer.add_scalar('lossb_dif', diff_lossb, epoch)
    # tensor_writer.add_scalar('lossg_dif', diff_lossg, epoch)
    # tensor_writer.add_scalar('weight_dif', diff_weight, epoch)
    # tensor_writer.add_scalar('weight_var', var_weight, epoch)
    # tensor_writer.add_scalar('weight_dis', dis, epoch)
    # tensor_writer.add_scalar('lambdap', lambdap0, epoch)


def validate(val_loader, model, criterion, epoch=0, test=True, args=None, tensor_writer=None):
    if test:
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')
    else:
        batch_time = AverageMeter('val Time', ':6.3f')
        losses = AverageMeter('val Loss', ':.4e')
        top1 = AverageMeter('Val Acc@1', ':6.2f')
        top5 = AverageMeter('Val Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Val: ')
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, paths) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output, cfeatures = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                method_name = args.log_path.split('/')[-2]
                progress.display(i, method_name)
                progress.write_log(i, args.log_path)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        with open(args.log_path, 'a') as f1:
            f1.writelines(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        # if test:
        #     tensor_writer.add_scalar('loss/test', loss.item(), epoch)
        #     tensor_writer.add_scalar('ACC@1/test', top1.avg, epoch)
        #     tensor_writer.add_scalar('ACC@5/test', top5.avg, epoch)
        # else:
        #     tensor_writer.add_scalar('loss/val', loss.item(), epoch)
        #     tensor_writer.add_scalar('ACC@1/val', top1.avg, epoch)
        #     tensor_writer.add_scalar('ACC@5/val', top5.avg, epoch)


    return top1.avg


def save_checkpoint(state, is_best, log_path, filename='checkpoint.pth.tar'):
    savename = os.path.join(os.path.dirname(log_path), filename)
    torch.save(state, savename)
    if is_best:
        shutil.copyfile(savename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, name):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('method name {}'.format(name))
        print('\t'.join(entries))

    def write_log(self, batch, log_path):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter)+'\n' for meter in self.meters]
        with open(log_path, 'a') as f1:
            f1.writelines(entries)


    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    # for lr decay
    lr = args.lr
    if epoch >= args.epochs_decay[0]:
        lr *= 0.1
    if epoch >= args.epochs_decay[1]:
        lr *= 0.1

    if epoch < args.lrwarmup_epo:
        lr = lr * args.lrwarmup_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_bl(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lrbl * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def testing(loader,device,model):
    model.eval()
    positive_prediction = 0
    total = 0
    for data in loader:
        images,labels = data
        images, labels = images.to(device), labels.to(device)
        logits,_ = model(images)
        _,pred = logits.max(dim=1)
        positive_prediction += torch.sum(pred == labels)
        total += labels.size(0)
    
    acc = float(positive_prediction)/total
    return acc

if __name__ == '__main__':
    main()