import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck
from torch.autograd import Variable
import numpy.random as npr
import random
import math

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        # change params: block and layers to swift from res18 to other models 
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc_ = nn.Linear(512 * block.expansion, num_classes)

        self.pecent = 1/3

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.register_buffer('pre_features', torch.zeros(128, 512))
        self.register_buffer('pre_weight1', torch.ones(128, 1))


    def _make_layer(self, block, planes, blocks, stride=1):
     
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x, gt=0, flag=False, epoch=0):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if flag:
            interval = 10
            if epoch % interval == 0:
                self.pecent = 3.0 / 10 + (epoch / interval) * 2.0 / 10

            self.eval()
            x_new = x.clone().detach()
            x_new = Variable(x_new.data, requires_grad=True)
            x_new_view = self.avgpool(x_new)
            x_new_view = x_new_view.view(x_new_view.size(0), -1)
            output = self.fc_(x_new_view)
            class_num = output.shape[1]
            index = gt
            num_rois = x_new.shape[0]
            num_channel = x_new.shape[1]
            H = x_new.shape[2]
            HW = x_new.shape[2] * x_new.shape[3]
            one_hot = torch.zeros((1), dtype=torch.float32).cuda()
            one_hot = Variable(one_hot, requires_grad=False)
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)
            one_hot = torch.sum(output * one_hot_sparse)
            self.zero_grad()
            one_hot.backward()
            grads_val = x_new.grad.clone().detach()
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            channel_mean = grad_channel_mean
            grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
            spatial_mean = torch.sum(x_new * grad_channel_mean, 1)
            spatial_mean = spatial_mean.view(num_rois, HW)
            self.zero_grad()

            choose_one = random.randint(0, 9)
            if choose_one <= 4:
                # ---------------------------- spatial -----------------------
                spatial_drop_num = math.ceil(HW * 1 / 3.0)
                th18_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]
                th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, 49)
                mask_all_cuda = torch.where(spatial_mean > th18_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                            torch.ones(spatial_mean.shape).cuda())
                mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)
            else:
                # -------------------------- channel ----------------------------
                vector_thresh_percent = math.ceil(num_channel * 1 / 3.2)
                vector_thresh_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]
                vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)
                vector = torch.where(channel_mean > vector_thresh_value,
                                     torch.zeros(channel_mean.shape).cuda(),
                                     torch.ones(channel_mean.shape).cuda())
                mask_all = vector.view(num_rois, num_channel, 1, 1)

            # ----------------------------------- batch ----------------------------------------
            cls_prob_before = F.softmax(output, dim=1)
            x_new_view_after = x_new * mask_all
            x_new_view_after = self.avgpool(x_new_view_after)
            x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
            x_new_view_after = self.fc_(x_new_view_after)
            cls_prob_after = F.softmax(x_new_view_after, dim=1)

            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
            change_vector = before_vector - after_vector - 0.0001
            change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
            th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][int(round(float(num_rois) * self.pecent))]
            drop_index_fg = change_vector.gt(th_fg_value).long()
            ignore_index_fg = 1 - drop_index_fg
            not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
            mask_all[not_01_ignore_index_fg.long(), :] = 1

            self.train()
            mask_all = Variable(mask_all, requires_grad=True)
            x = x * mask_all


        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        logits = self.fc_(features)

        return logits,features


def main():
    # downloading pretrained resnet to local env
    # resnet18 = models.resnet18(pretrained=True)    
    # resnet50 = models.resnet50(pretrained=True)

    # torch.save(resnet18.state_dict(), './resnet18_model.pth')
    # torch.save(resnet50.state_dict(), './resnet50_model.pth')
    # return 1


    # resnet18 = models.resnet18(weights=None)      # new version of loading architecture
    resnet50 = models.resnet50(weights=None)
    # resnet18.load_state_dict(torch.load('models/resnet18/resnet18_model.pth'))
    resnet50.load_state_dict(torch.load('models/resnet50/resnet50_model.pth'))

    # model_loc = ResNet(num_classes=17, block=BasicBlock, layers=[2, 2, 2, 2])    # resnet18  
    model_loc = ResNet(num_classes=17, block=Bottleneck, layers=[3, 4, 6, 3])    # resnet50

    # pretrained_dict = resnet18.state_dict()
    pretrained_dict = resnet50.state_dict()

    model_dict_loc = model_loc.state_dict()

    pretrained_dict_loc = {k:v for k,v in pretrained_dict.items() if k in model_dict_loc}
    model_dict_loc.update(pretrained_dict_loc)
    model_loc.load_state_dict(model_dict_loc)

    # torch.save(model_loc.state_dict(),'models/resnet18/nico.pkl')
    torch.save(model_loc.state_dict(),'models/resnet50/nico.pkl')

if __name__ == '__main__':
    main()
