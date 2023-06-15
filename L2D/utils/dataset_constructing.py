import os
from tqdm import tqdm
import sys

dictionary_of_class_A = {'bear':0, 'bird':1, 'cat':2, 'cow':3, 'dog':4, 'elephant':5, 'horse':6, 'monkey':7, 'rat':8, 'sheep':9}
dictionary_of_class_B = {'airplane':0, 'bicycle':1, 'boat':2, 'bus':3, 'car':4, 'helicopter':5, 'motorcycle':6, 'train':7}


list_of_class_A = ['bear', 'bird', 'cat', 'cow', 'dog', 'elephant', 'horse', 'monkey', 'rat', 'sheep']
list_of_contexts_A = [
    ['black', 'brown', 'eating grass', 'in forest', 'in water', 'lying', 'on ground', 'on snow', 'on tree', 'white'],
    ['eating', 'flying', 'in cage', 'in hand', 'in water', 'on branch', 'on grass', 'on ground', 'on shoulder','standing'],
    ['at home', 'eating', 'in cage', 'in river', 'in street', 'in water', 'on grass', 'on snow', 'on tree', 'walking'],
    ['aside people', 'at home', 'eating', 'in forest', 'in river', 'lying', 'on grass','on snow', 'spotted', 'standing'],
    ['at home', 'eating', 'in cage', 'in street', 'in water', 'lying', 'on beach', 'on grass', 'on snow', 'running'],
    ['eating', 'in circus', 'in forest', 'in river', 'in street', 'in zoo', 'lying', 'on grass', 'on snow', 'standing'],
    ['aside people', 'at home', 'in forest', 'in river', 'in street', 'lying', 'on beach', 'on grass', 'on snow', 'running'],
    ['climbing', 'eating', 'in cage', 'in forest', 'in water', 'on beach', 'on grass', 'on snow', 'sitting', 'walking'],
    ['at home', 'eating', 'in cage', 'in forest', 'in hole', 'in water', 'lying', 'on grass', 'on snow', 'running'],
    ['aside people', 'at sunset', 'eating', 'in forest', 'in water', 'lying', 'on grass', 'on road', 'on snow', 'walking']
]

list_of_class_V = ['airplane', 'bicycle', 'boat', 'bus', 'car', 'helicopter', 'motorcycle', 'train']
list_of_contexts_V = [
    ['around cloud', 'aside mountain', 'at airport', 'at night', 'in city', 'in sunrise', 'on beach', 'on grass', 'taking off', 'with pilot'],
    ['in garage', 'in street', 'in sunset', 'on beach', 'on grass', 'on road', 'on snow', 'shared', 'velodrome','with people'],
    ['at wharf', 'cross bridge', 'in city', 'in river', 'in sunset', 'on beach', 'sailboat', 'with people', 'wooden', 'yacht'],
    ['aside traffic light', 'aside tree', 'at station', 'at yard', 'double decker', 'in city', 'on bridge', 'on snow', 'with people'],
    ['at park', 'in city', 'in sunset', 'on beach', 'on booth', 'on bridge', 'on road', 'on snow', 'on track', 'with people'],
    ['aside mountain', 'at heliport', 'in city', 'in forest', 'in sunset', 'on beach', 'on grass', 'on sea', 'on snow', 'with people'],
    ['in city', 'in garage', 'in street', 'in sunset', 'on beach', 'on grass', 'on road', 'on snow', 'on track', 'with people'],
    ['aside mountain', 'at station', 'cross tunnel', 'in forest', 'in sunset', 'on beach', 'on bridge', 'on snow', 'subway']
]

suffix = ['.jpg','.jpeg']


def main(argv):

    if argv[1] == 'Animal':
        train = open('./Dataset/Ani_train.txt', 'w')
        valid = open('./Dataset/Ani_valid.txt', 'w')
        test = open('./Dataset/Ani_test.txt', 'w')
        N = 10
        list_of_class = list_of_class_A
        list_of_context = list_of_contexts_A
    elif argv[1] == 'Vehicle':
        train = open('./Dataset/Vel_train.txt', 'w')
        valid = open('./Dataset/Vel_valid.txt', 'w')
        test = open('./Dataset/Vel_test.txt', 'w')
        N = 8
        list_of_class = list_of_class_V
        list_of_context = list_of_contexts_V
    else:
        raise ValueError('wrong dataset name.')

    bar = tqdm(total = N)
    for a in range(N):  
        dir = '/data/dengx/' + argv[1] + '/' + list_of_class[a] + '/'  #address of each class.
        label = a
        context_num = len(list_of_context[a])
        
        for b in range(context_num):
         
            # os.listdir outputs a list
            dir_sub = dir + list_of_context[a][b] + '/'
            files = os.listdir(dir_sub)
            i = 1
            num = len(files)
            for file in files:
                if b < 5:
                    # 5 contexts appear in train/val set
                    if i < num*0.8:

                        fileType = os.path.splitext(file)
                        if fileType[1] not in suffix:       #remove all the wrong images
                            continue
                        name = str(dir_sub) + file + '$' + str(int(label)) + '\n'
                        train.write(name)
                        i = i + 1

                    else:

                        fileType = os.path.splitext(file)
                        if fileType[1] not in suffix:
                            continue
                        name = str(dir_sub) + file + '$' + str(int(label)) + '\n'
                        valid.write(name)
                        i = i + 1
                    
                else:
                    # the rest 5 contexts appear in test set
                    if i < num * 0.4:
                        fileType = os.path.splitext(file)
                        if fileType[1] not in suffix:
                            continue
                        name = str(dir_sub) + file + '$' + str(int(label)) + '\n'
                        test.write(name)
                        i = i + 1
                
        bar.update()
    bar.close()
    valid.close()
    test.close()
    train.close()

if __name__ == '__main__':
    main(sys.argv)

    # run python utils/dataset_constructing.py Animal to implement it

