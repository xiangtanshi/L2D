import os
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from cgn_class import *
import random
from tqdm import tqdm
import random
from collections import defaultdict
import argparse

unloader = transforms.ToPILImage()

def mkdir(path):

    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

class Datas:
    def __init__(self):
        return 

parser = argparse.ArgumentParser(description='parameter setting for generating counterfactual images where background remians unchanged and foreground texture is replaced by texture templates.')
parser.add_argument('--task',type=str, default='train')
parser.add_argument('--classes',type=int, default=17,help='number of classes')
parser.add_argument('--mode', type=str, default='val',choices=['val','test'], help='for siamese network training, organized as triplets')
parser.add_argument('--mode1', type=str, default='val',choices=['val','test'], help='for consensus testing, organized as sequence')

args =  parser.parse_args()

def data_for_training(args,data_obj):
    '''
    generate triad based on the train and test set. 
    for each raw image: generate 9 triplet with replaced foreground from various classes and original background unchanged,
    as we cannot ensure the foregound are separated successfully for every test image, we do not change the separated 'background' that may contain foreground.
    the images are stored in the order that they are generated
    '''

    if args.mode == 'val':
        steps = len(data_obj.valid_loader)   
        iters = iter(data_obj.valid_loader)   
        mask_iters = iter(data_obj.valid_mask_loader)   
    elif args.mode == 'test':
        steps = len(data_obj.test_loader)   
        iters = iter(data_obj.test_loader)   
        mask_iters = iter(data_obj.test_mask_loader)  
 
    foreground = defaultdict(list)
    for item in data_obj.fg_loader:
        image, label = item  
        foreground[int(label.item())].append(image)  # add textures to the corresponding list

    for step in tqdm(range(steps)):
        # process 1 image for each iteration
        image,label = next(iters)
        mask,_ = next(mask_iters)
        if args.mode == 'val':
            dir_path = './datas/counter/triad/triple_val1/{}'.format(step)  
        elif args.mode == 'test':
            dir_path = './datas/counter/triad/triple_con2/{}'.format(step)
        mkdir(dir_path)  
        
        other_fg = [i for i in range(args.classes) if i != label] 

        for i in range(args.classes-1):
            # create triplets for each class
            anchor = image
            p_fg = random.choice(foreground[int(label.item())])
            n_fg = random.choice(foreground[other_fg[i]])

            p_m = p_fg*mask
            n_m = n_fg*mask
            a_b = anchor*(1-mask)

            p = p_m + a_b
            n = n_m + a_b

            anchor = unloader(anchor[0])
            p = unloader(p[0])
            n = unloader(n[0])

            anchor.save(dir_path + '/{}.jpg'.format(3*i)) 
            p.save(dir_path + '/{}.jpg'.format(3*i+1))  
            n.save(dir_path + '/{}.jpg'.format(3*i+2))   
    

def data_for_testing(args,data_obj):
              
    if args.mode1 == 'test':        
        steps = len(data_obj.test_loader)   
        itr = iter(data_obj.test_loader)    
        mask_itr = iter(data_obj.test_mask_loader)   
    elif args.mode1 == 'val':
        steps = len(data_obj.valid_loader)   
        itr = iter(data_obj.valid_loader)    
        mask_itr = iter(data_obj.valid_mask_loader)

    foreground = defaultdict(list)
    for item in data_obj.fg_loader:
        image, label = item  
        foreground[int(label.item())].append(image)   

    for step in tqdm(range(steps)):

        image,_ = next(itr)
        mask,_ = next(mask_itr)
        if args.mode1 == 'val':
            dir_path = './datas/counter/triad/val1/{}'.format(step)  
        elif args.mode1 == 'test':
            dir_path = './datas/counter/triad/con2/{}'.format(step)
        mkdir(dir_path)     

        anchor = image
        anchor_im = unloader(anchor[0])
        anchor_im.save(dir_path + '/0.jpg') 

        idx = 0
        for j in range(3):
            for i in range(args.classes):
                idx += 1
                # texture_i = random.choice(foreground[i])
                texture_i = foreground[i][j]
                gen = texture_i * mask + image * (1-mask)
                gen_im = unloader(gen[0])
                gen_im.save(dir_path + '/{}.jpg'.format(idx))



def separate_background(data_obj):

    steps = len(data_obj.test_loader)
    iters = iter(data_obj.test_loader)
    mask_iters = iter(data_obj.test_mask_loader)

    bar = tqdm(total=steps)
    for step in range(steps):
        bar.update()
        image,label = next(iters)
        mask,_ = next(mask_iters)
        target = image*(1 - mask)
        target = unloader(target[0])
        target.save('./datas/counter/test_bg/{}.jpg'.format(step))
    bar.close()


if __name__ == '__main__':

    data_obj = Datas()

    # true image
    data_obj.valid_loader = DataLoader(dataset=val1, batch_size=1, shuffle=False, num_workers=1)   
    data_obj.test_loader = DataLoader(dataset=con2, batch_size=1, shuffle=False, num_workers=1)
    # mask   
    data_obj.valid_mask_loader = DataLoader(dataset=val1__shape, batch_size=1, shuffle=False, num_workers=1)   
    data_obj.test_mask_loader = DataLoader(dataset=con2__shape, batch_size=1, shuffle=False, num_workers=1)
    # foreground texture template
    data_obj.fg_loader = DataLoader(dataset=feature, batch_size=1, shuffle=False, num_workers=1)
        
    if args.task == 'train':
        data_for_training(args,data_obj)
    elif args.task == 'test':
        data_for_testing(args,data_obj)
