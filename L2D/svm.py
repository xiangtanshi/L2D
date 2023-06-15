import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as ms
from utils.logit_dataset import *
from torch.utils.data import DataLoader


def main():

    # load the tensor to one dimensional numpy array
    valid_loader = DataLoader(dataset=a_valid_logit, batch_size=1, shuffle=False, num_workers=1)    
    test_loader = DataLoader(dataset=a_test_logit, batch_size=1, shuffle=False, num_workers=1)  


    num_valid = len(a_valid_logit)
    num_test = len(a_test_logit)
    length = 20
    Valid_data = np.zeros((num_valid,length))
    Valid_class = np.zeros(num_valid)       #binary
    Test_data = np.zeros((num_test,length))
    Test_class = np.zeros(num_test)         #binary

    for index,(logit,label) in enumerate(valid_loader):
        Valid_data[index] = logit[0,0,0]
        Valid_class[index] = label

    for index,(logit,label) in enumerate(test_loader):
        Test_data[index] = logit[0,0,0]
        Test_class[index] = label
    
    # grid = ms.GridSearchCV(SVC(decision_function_shape='ovo'),param_grid={'C': [0.1,0.5,1,5,10,100,1000], 'gamma':[10,1,0.1,0.01]},cv=3)
    # grid.fit(Valid_data,Valid_class)
    # print('valid: best params are %s with a score of %0.3f' % (grid.best_params_,grid.best_score_))
    # grid.fit(Test_data,Test_class)
    # print('test: best params are %s with a score of %0.3f' % (grid.best_params_,grid.best_score_))

    
    # judging
    c=1
    g=0.1
    clf = SVC(C=c,gamma=g)
    clf.fit(Valid_data,Valid_class)  

    # test
    positive = 0.0
    total = num_test
    for i in range(num_test):
        pred = clf.predict(Test_data[i:i+1,:])
        if pred == Test_class[i]:
            positive += 1
    acc = positive/total

    print(acc)    
    
if __name__ == '__main__':
    main()

