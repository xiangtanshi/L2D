'''
we generate counterfactual data here
these data are organized as triplets during training, the triplets are trained online with batch hard sample stragety
during testing, we need to utilize both the probability and the cosine distance, so they are organized differently
'''
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from cgn_class import datas,masks,ani_tx,vel_tx,background
import random
from tqdm import tqdm
import argparse

unloader = transforms.ToPILImage()

def mkdir(path):

    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

class Datas:
    def __init__(self):
        return 1

def get_args():

    parser = argparse.ArgumentParser(description='parameter setting for generating templates, mainly for loading the data.')
    parser.add_argument('task',default=0)
    parser.add_argument('strs',default='triple_train',choices=['triple_train','triple_test','test','valid'])
    parser.add_argument('strs1',default='animals',choices=['animals','vehicles'])
    parser.add_argument('classes',default=10,choices=[8,10],help='number of classes, 10 for animal and 8 for vehicle')
    parser.add_argument('num_template',default=10,choices=[8,10],help='number of texture templates, 10 for aniaml and 8 for vehicle')
    parser.add_argument('mode',default='train',choices=['train','test'], help='for training, organized as triplets')
    parser.add_argument('mode1',default='valid',choices=['valid','test'], help='for testing, organized as sequence')

    return parser.parse_args()

def data_for_training(args,data_obj):
    '''
    generate triad based on the train and test set. 
    for each raw image: generate 9 triplet with original background, 9 with different backgrounds
    they are stored in the order that they are generated
    '''

    if args.mode == 'train':
        steps = len(data_obj.train_loader)   
        iters = iter(data_obj.train_loader)   
        mask_iters = iter(data_obj.train_mask_loader)   
    elif args.mode == 'test':
        steps = len(data_obj.test_loader)   
        iters = iter(data_obj.test_loader)   
        mask_iters = iter(data_obj.test_mask_loader)  

    # item = (image,label)
    foreground = [item for item in data_obj.fg_loader]    
    background = [item for item in data_obj.bg_loader]   # 100 different kinds of backgrounds

    bar = tqdm(total=steps)
    for step in range(steps):
        bar.update()
        # process 1 image for each iteration
        image,label_1 = next(iters)
        mask,_ = next(mask_iters)
        mkdir('/data/dengx/counterfactual/' + args.strs1 + '/triad/' + args.strs + '/{}'.format(step))   
        
        shift = int(np.random.randint(0,args.num_template,1))

        if args.strs1 == 'animals':
            fgs = [0,1,2,3,4,5,6,7,8,9]
        else:
            fgs = [0,1,2,3,4,5,6,7]
        random.shuffle(fgs)
        other_fgs = [args.num_template*x+shift for x in range(args.classes) if x != label_1]
        bgs = np.random.randint(0,100,args.classes-1)

        for i in range(args.classes-1):
            # original background
            anchor = image
            p_fg = foreground[fgs[i]+args.num_template*label_1][0]
            n_fg = foreground[other_fgs[i]][0]

            p_m = p_fg*mask
            n_m = n_fg*mask
            a_b = anchor*(1-mask)

            p = p_m + a_b
            n = n_m + a_b

            anchor = unloader(anchor[0])
            p = unloader(p[0])
            n = unloader(n[0])

            anchor.save('/data/dengx/counterfactual/' + args.strs1 + '/triad/' + args.strs + '/{}/{}.jpg'.format(step,3*i)) 
            p.save('/data/dengx/counterfactual/' + args.strs1 + '/triad/' + args.strs + '/{}/{}.jpg'.format(step,3*i+1))  
            n.save('/data/dengx/counterfactual/' + args.strs1 + '/triad/' + args.strs + '/{}/{}.jpg'.format(step,3*i+2))  

            for j in range(1):
                # other background
                o_b = background[bgs[i]][0]*(1-mask)
                anchor_j = image*mask + o_b
                p_j = p_m + o_b
                n_j = n_m + o_b

                anchor_j = unloader(anchor_j[0])
                p_j = unloader(p_j[0])
                n_j = unloader(n_j[0])

                anchor_j.save('/data/dengx/counterfactual/' + args.strs1 + '/triad/' + args.strs + '/{}/{}.jpg'.format(step,3*i+3*(args.classes-1)*(j+1))) 
                p_j.save('/data/dengx/counterfactual/' + args.strs1 + '/triad/' + args.strs + '/{}/{}.jpg'.format(step,3*i+1+3*(args.classes-1)*(j+1))) 
                n_j.save('/data/dengx/counterfactual/' + args.strs1 + '/triad/' + args.strs + '/{}/{}.jpg'.format(step,3*i+2+3*(args.classes-1)*(j+1))) 
    
    bar.close()


def data_for_testing(args,data_obj):
              
    if args.mode1 == 'test':        
        steps = len(data_obj.test_loader)   
        itr = iter(data_obj.test_loader)    
        mask_itr = iter(data_obj.test_mask_loader)   
    elif args.mode1 == 'valid':
        steps = len(data_obj.valid_loader)   
        itr = iter(data_obj.valid_loader)    
        mask_itr = iter(data_obj.valid_mask_loader)

    foreground = [item for item in data_obj.fg_loader]    
    background = [item for item in data_obj.bg_loader]
    bar = tqdm(total=steps)

    for step in range(steps):

        bar.update()
        image,_ = next(itr)
        mask,_ = next(mask_itr)
        mkdir('/data/dengx/counterfactual/' + args.strs1 + '/triad/' + args.strs + '/{}'.format(step))   

        bgs = np.random.randint(0,100,3)     

        anchor = image
        anchor_im = unloader(anchor[0])
        anchor_im.save('/data/dengx/counterfactual/' + args.strs1 + '/triad/' + args.strs + '/{}/{}.jpg'.format(step,args.classes)) 

        for i in range(3):
            anchor = image*mask + background[bgs[i]][0]*(1-mask)
            anchor_im = unloader(anchor[0])
            anchor_im.save('/data/dengx/counterfactual/' + args.strs1 + '/triad/' + args.strs + '/{}/{}.jpg'.format(step,(args.classes+1)*(i+2)-1))


        for i in range(args.classes):

            shifts = random.sample(range(1,args.num_template),4)
            gen = foreground[i*args.num_template+shifts[0]-1][0]*mask + image*(1-mask)
            gen_im = unloader(gen[0])
            gen_im.save('/data/dengx/counterfactual/' + args.strs1 + '/triad/' + args.strs + '/{}/{}.jpg'.format(step,i))

            for j in range(3):
                gen = foreground[i*args.num_template+shifts[j+1]-1][0]*mask + background[bgs[j]][0]*(1-mask)
                gen_im = unloader(gen[0])
                gen_im.save('/data/dengx/counterfactual/' + args.strs1 + '/triad/' + args.strs + '/{}/{}.jpg'.format(step,i+(args.classes+1)*(j+1)))

    bar.close()


def separate_foreground(args,data_obj):

    steps = len(data_obj.test_loader)
    iters = iter(data_obj.test_loader)
    mask_iters = iter(data_obj.test_mask_loader)

    bar = tqdm(total=steps)
    for step in range(steps):
        bar.update()
        image,label_1 = next(iters)
        mask,_ = next(mask_iters)
        target = image*mask
        target = unloader(target[0])
        target.save('/data/dengx/counterfactual/' + args.strs1 + '/pure_fg/{}.jpg'.format(step))
    bar.close()


if __name__ == '__main__':

    args = get_args()
    data_obj = Datas()

    if args.strs1 == 'animals':
        # true image
        data_obj.train_loader = DataLoader(dataset=datas[0], batch_size=1, shuffle=False, num_workers=1)    
        data_obj.valid_loader = DataLoader(dataset=datas[1], batch_size=1, shuffle=False, num_workers=1)   
        data_obj.test_loader = DataLoader(dataset=datas[2], batch_size=1, shuffle=False, num_workers=1)
        # mask
        data_obj.train_mask_loader = DataLoader(dataset=masks[0], batch_size=1, shuffle=False, num_workers=1)    
        data_obj.valid_mask_loader = DataLoader(dataset=masks[1], batch_size=1, shuffle=False, num_workers=1)   
        data_obj.test_mask_loader = DataLoader(dataset=masks[2], batch_size=1, shuffle=False, num_workers=1)

        data_obj.fg_loader = DataLoader(dataset=ani_tx, batch_size=1, shuffle=False, num_workers=1)

    else:
        # true image
        data_obj.train_loader = DataLoader(dataset=datas[3], batch_size=1, shuffle=False, num_workers=1)    
        data_obj.valid_loader = DataLoader(dataset=datas[4], batch_size=1, shuffle=False, num_workers=1)   
        data_obj.test_loader = DataLoader(dataset=datas[5], batch_size=1, shuffle=False, num_workers=1)
        # mask
        data_obj.train_mask_loader = DataLoader(dataset=masks[3], batch_size=1, shuffle=False, num_workers=1)    
        data_obj.valid_mask_loader = DataLoader(dataset=masks[4], batch_size=1, shuffle=False, num_workers=1)   
        data_obj.test_mask_loader = DataLoader(dataset=masks[5], batch_size=1, shuffle=False, num_workers=1)

        data_obj.fg_loader = DataLoader(dataset=vel_tx, batch_size=1, shuffle=False, num_workers=1)

        
    data_obj.bg_loader = DataLoader(dataset=background, batch_size=1, shuffle=True, num_workers=1)
    
    if args.task == 0:
        data_for_training(args,data_obj)
    else:
        data_for_testing(args,data_obj)

    # python Cgn/generate_tripletset.py --task 0 --strs1 animals --classes 10 --num_template 10 --strs triple_train --mode train
    # python Cgn/generate_tripletset.py --task 0 --strs1 animals --classes 10 --num_template 10 --strs triple_test --mode test
    # python Cgn/generate_tripletset.py --task 0 --strs1 vehicles --classes 8 --num_template 8 --strs triple_train --mode train
    # python Cgn/generate_tripletset.py --task 0 --strs1 vehicles --classes 8 --num_template 8 --strs triple_test --mode test

    # python Cgn/generate_tripleset.py --task 1 --strs1 animals --classes 10 --num_template 10 --mode1 test --strs test
    # python Cgn/generate_tripleset.py --task 1 --strs1 animals --classes 10 --num_template 10 --mode1 valid --strs valid
    # python Cgn/generate_tripleset.py --task 1 --strs1 vehicles --classes 8 --num_template 8 --mode1 test --strs test
    # python Cgn/generate_tripleset.py --task 1 --strs1 vehicles --classes 8 --num_template 8 --mode1 valid --strs valid



