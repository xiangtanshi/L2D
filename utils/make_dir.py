'''
initialize the directory for the whole project 
'''

import os

def mkdir(path):

    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def main():
    mkdir('./datas/counter')

    mkdir('./datas/counter/' + 'foreground')
    mkdir('./datas/counter/' + 'mask')
    mkdir('./datas/counter/' + 'triad')

    
    mkdir('./datas/counter/' + 'mask/con1')
    mkdir('./datas/counter/' + 'mask/val1')
    mkdir('./datas/counter/' + 'mask/con2')
    mkdir('./datas/counter/' + 'mask/val2')

    mkdir('./datas/counter/' + 'triad/triple_train')
    mkdir('./datas/counter/' + 'triad/triple_test')
    mkdir('./datas/counter/' + 'triad/con1')
    mkdir('./datas/counter/' + 'triad/val1')
    mkdir('./datas/counter/' + 'triad/con2')
    mkdir('./datas/counter/' + 'triad/val2')


if __name__ == '__main__':
    main()
