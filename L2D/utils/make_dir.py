'''
initialize the directory for the whole project 
'''

import os

def mkdir(path):

    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def main():
    mkdir('/data/dengx/counterfactual')

    mkdir('/data/dengx/counterfactual/' + 'animals')
    mkdir('/data/dengx/counterfactual/' + 'vehicles')

    mkdir('/data/dengx/counterfactual/' + 'animals/foreground')
    mkdir('/data/dengx/counterfactual/' + 'animals/mask')
    mkdir('/data/dengx/counterfactual/' + 'animals/triad')
    mkdir('/data/dengx/counterfactual/' + 'animals/logit')
    mkdir('/data/dengx/counterfactual/' + 'animals/pure_fg')

    mkdir('/data/dengx/counterfactual/' + 'vehicles/foreground')
    mkdir('/data/dengx/counterfactual/' + 'vehicles/mask')
    mkdir('/data/dengx/counterfactual/' + 'vehicles/triad')
    mkdir('/data/dengx/counterfactual/' + 'vehicles/logit')
    mkdir('/data/dengx/counterfactual/' + 'vehicles/pure_fg')
    
    mkdir('/data/dengx/counterfactual/' + 'animals/mask/train')
    mkdir('/data/dengx/counterfactual/' + 'animals/mask/val')
    mkdir('/data/dengx/counterfactual/' + 'animals/mask/test')
    mkdir('/data/dengx/counterfactual/' + 'animals/triad/triple_train')
    mkdir('/data/dengx/counterfactual/' + 'animals/triad/triple_test')
    mkdir('/data/dengx/counterfactual/' + 'animals/triad/test')
    mkdir('/data/dengx/counterfactual/' + 'animals/triad/valid')
    mkdir('/data/dengx/counterfactual/' + 'animals/logit/test')
    mkdir('/data/dengx/counterfactual/' + 'animals/logit/valid')

    mkdir('/data/dengx/counterfactual/' + 'vehicles/mask/train')
    mkdir('/data/dengx/counterfactual/' + 'vehicles/mask/val')
    mkdir('/data/dengx/counterfactual/' + 'vehicles/mask/test')
    mkdir('/data/dengx/counterfactual/' + 'vehicles/triad/triple_train')
    mkdir('/data/dengx/counterfactual/' + 'vehicles/triad/triple_test')
    mkdir('/data/dengx/counterfactual/' + 'vehicles/triad/test')
    mkdir('/data/dengx/counterfactual/' + 'vehicles/triad/valid')
    mkdir('/data/dengx/counterfactual/' + 'vehicles/logit/test')
    mkdir('/data/dengx/counterfactual/' + 'vehicles/logit/valid')


if __name__ == '__main__':
    main()
