'''
This file contains the code that select the training data classes and OOD test data.
'''
import os
from tqdm import tqdm
import sys
import random

# the subset of classes are chosen so that each class has at leat one ambiguous class that the classfier will struggle to determine, for
# reasons like similar shape, similar contexture or similar background environments that they apper, especially in th OOD scenarios. 
list_of_class = ['football','hot air balloon','airplane','bird','bear','monkey','bicycle','motorcycle','bus','car','cat','dog','cow','horse','elephant','seal','ship']
N = 17

context_train = ['autumn','dim','grass']
context_test = ['outdoor','rock','water']

suffix = ['.jpg','.jpeg']


def main():

    if os.path.exists('../datas/NICO_paths/train.txt'):
        raise ValueError('There already exists the train/val/test splitting!')
    con1 = open('./datas/NICO_paths/context1.txt', 'w')
    val1 = open('./datas/NICO_paths/val1.txt', 'w')
    con2 = open('./datas/NICO_paths/context2.txt', 'w')
    val2 = open('./datas/NICO_paths/val2.txt', 'w')


    bar = tqdm(total = N)
    for a in range(N):  
        dir = './datas/NICO_DG/'  #address of each class.
        label = a
        
        for b in range(3):
         
            # os.listdir outputs a list
            dir_sub = dir + context_train[b] + '/' + list_of_class[a] + '/'
            files = os.listdir(dir_sub)
            for file in files:
                fileType = os.path.splitext(file)
                if fileType[1] not in suffix:       #remove all the wrong images
                    continue
                name = dir_sub + file + '$' + str(int(label)) + '\n'
                number = random.random()
                if number < 0.9:
                    con1.write(name)
                else:
                    val1.write(name)

        for b in range(3):
         
            # os.listdir outputs a list
            dir_sub = dir + context_test[b] + '/' + list_of_class[a] + '/'
            files = os.listdir(dir_sub)
            for file in files:
                fileType = os.path.splitext(file)
                if fileType[1] not in suffix:       #remove all the wrong images
                    continue
                name = dir_sub + file + '$' + str(int(label)) + '\n'
                number = random.random()
                if number < 0.9:
                    con2.write(name)
                else:
                    val2.write(name)
 
        bar.update()
    bar.close()
    con1.close()
    val1.close()
    con2.close()
    val2.close()

if __name__ == '__main__':
    main()

    # run python utils/dataset_constructing.py to implement it

