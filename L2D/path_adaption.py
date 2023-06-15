
def set_bg_path():
    # only run 1 time

    file = open('Cgn/cgn_data/counterfactual_data/background.txt','w')
    paths = []
    for i in  range(100):
        paths.append('/data/dengx/counterfactual/background/{}.jpg${}\n'.format(i,0))
    file.writelines(paths)
    file.close()


def set_fg_path():

    file1 = open('Cgn/cgn_data/counterfactual_data/foreground/ani_ft.txt','w')
    file2 = open('Cgn/cgn_data/counterfactual_data/foreground/ani_tx.txt','w')
    file3 = open('Cgn/cgn_data/counterfactual_data/foreground/vel_ft.txt','w')
    file4 = open('Cgn/cgn_data/counterfactual_data/foreground/vel_tx.txt','w')
    paths1 = []
    paths2 = []
    paths3 = []
    paths4 = []

    for i in range(10):
        for j in range(10):
            paths1.append('/data/dengx/counterfactual/animals/foreground/feature/{}/{}.jpg${}\n'.format(i,j,i))
            paths2.append('/data/dengx/counterfactual/animals/foreground/texture/{}/{}.jpg${}\n'.format(i,j,i))
    file1.writelines(paths1)
    file2.writelines(paths2)

    for i in range(8):
        for j in range(8):
            paths3.append('/data/dengx/counterfactual/vehicles/foreground/feature/{}/{}.jpg${}\n'.format(i,j,i))
            paths4.append('/data/dengx/counterfactual/vehicles/foreground/texture/{}/{}.jpg${}\n'.format(i,j,i))
    file3.writelines(paths3)
    file4.writelines(paths4)
    file1.close()
    file2.close()
    file3.close()
    file4.close()
    

def set_shape_path_ani():
    '''
    generate path file for the mask set of train, valid and test mask for Animal
    '''
    file1 = ['ani/train_mask.txt','ani/valid_mask.txt','ani/test_mask.txt']          
    file2 = ['Dataset/ood/Ani_train.txt','Dataset/ood/Ani_valid.txt','Dataset/ood/Ani_test.txt']  
    strings = ['/data/dengx/counterfactual/animals/mask/train/','/data/dengx/counterfactual/animals/mask/valid/', 
                '/data/dengx/counterfactual/animals/mask/test/'] 
   
    for i in range(3):
        f1 = open('Cgn/cgn_data/counterfactual_data/mask/' + file1[i],'w')
        f2 = open(file2[i],'r')
        contents = f2.readlines()
        mask_cotents = [strings[i]+ '{}.jpg'.format(idx) + strs[-3:] for idx,strs in enumerate(contents)]
        f1.writelines(mask_cotents)
        f1.close()
        f2.close()

def set_shape_path_vel():
    '''
    generate path file for the mask set of train, valid and test mask for Vehicle
    '''
    file1 = ['vel/train_mask.txt','vel/valid_mask.txt','vel/test_mask.txt']          
    file2 = ['Dataset/ood/Vel_train.txt','Dataset/ood/Vel_valid.txt','Dataset/ood/Vel_test.txt']  
    strings = ['/data/dengx/counterfactual/vehicles/mask/train/','/data/dengx/counterfactual/vehicles/mask/valid/', 
                '/data/dengx/counterfactual/vehicles/mask/test/'] 
   
    for i in range(3):
        f1 = open('Cgn/cgn_data/counterfactual_data/mask/' + file1[i],'w')
        f2 = open(file2[i],'r')
        contents = f2.readlines()
        mask_cotents = [strings[i]+ '{}.jpg'.format(idx) + strs[-3:] for idx,strs in enumerate(contents)]
        f1.writelines(mask_cotents)
        f1.close()
        f2.close()

def set_counter_path_ani():
    # generate paths for counterfactual data that is used to calculate consensus in inference.py
    file = open('Cgn/cgn_data/counterfactual_data/retroset/ani/test_set.txt','w')         
    file1 = open('Dataset/ood/Ani_test.txt','r')    
    file_ = open('Cgn/cgn_data/counterfactual_data/retroset/ani/valid_set.txt','w')        
    file1_ = open('Dataset/ood/Ani_valid.txt','r')    

    contents = file1.readlines()
    contents_ = file1_.readlines()
    adapt_contents = []
    adapt_contents_ = []

    classes = 10
    for index,item in enumerate(contents):
        for j in range(4*(classes+1)):
            adapt_contents.append('/data/dengx/counterfactual/animals/triad/test/{}/{}.jpg'.format(index,j)+item[-3:])  # change 'animals' to 'vehicles'
            
    for index,item in enumerate(contents_):
        for j in range(4*(classes+1)):
            adapt_contents_.append('/data/dengx/counterfactual/animals/triad/valid/{}/{}.jpg'.format(index,j)+item[-3:])  # change 'animals' to 'vehicles'

    file.writelines(adapt_contents)
    file_.writelines(adapt_contents_)
    file.close()
    file1.close()
    file_.close()
    file1_.close()

def set_counter_path_vel():
    # generate paths for counterfactual data that is used to calculate consensus in inference.py
    file = open('Cgn/cgn_data/counterfactual_data/retroset/vel/test_set.txt','w')          
    file1 = open('Dataset/ood/Vel_test.txt','r')       
    file_ = open('Cgn/cgn_data/counterfactual_data/retroset/vel/valid_set.txt','w')         
    file1_ = open('Dataset/ood/Vel_valid.txt','r')    

    contents = file1.readlines()
    contents_ = file1_.readlines()
    adapt_contents = []
    adapt_contents_ = []

    classes = 10
    for index,item in enumerate(contents):
        for j in range(4*(classes+1)):
            adapt_contents.append('/data/dengx/counterfactual/vehicles/triad/test/{}/{}.jpg'.format(index,j)+item[-3:]) 
            
    for index,item in enumerate(contents_):
        for j in range(4*(classes+1)):
            adapt_contents_.append('/data/dengx/counterfactual/vehicles/triad/valid/{}/{}.jpg'.format(index,j)+item[-3:])  

    file.writelines(adapt_contents)
    file_.writelines(adapt_contents_)
    file.close()
    file1.close()
    file_.close()
    file1_.close()

def set_train_aug_path_ani():
    # counterfactual data augmentation
    file1 = open('Dataset/ood/Ani_train.txt','r')      
    file2 = open('Dataset/ood/Ani_train1.txt','w')     

    contents = file1.readlines()
    aug_contents = []
    for idx,strs in enumerate(contents):
        aug_contents.append(strs)
        aug_data = '/data/dengx/counterfactual/animals/triad/triple_train/{}/27.jpg'.format(idx) + strs[-3:]        # change 'animals' to 'vehicles'
        aug_contents.append(aug_data)
    file2.writelines(aug_contents)
    file1.close()
    file2.close()

def set_train_aug_path_vel():
    # counterfactual data augmentation
    file1 = open('Dataset/ood/Vel_train.txt','r')       
    file2 = open('Dataset/ood/Vel_train1.txt','w')      

    contents = file1.readlines()
    aug_contents = []
    for idx,strs in enumerate(contents):
        aug_contents.append(strs)
        aug_data = '/data/dengx/counterfactual/animals/triad/triple_train/{}/27.jpg'.format(idx) + strs[-3:]        # change 'animals' to 'vehicles'
        aug_contents.append(aug_data)
    file2.writelines(aug_contents)
    file1.close()
    file2.close()

if __name__ == '__main__':

    set_bg_path()
    set_fg_path()

    set_shape_path_ani()
    set_shape_path_vel()
    set_counter_path_ani()
    set_counter_path_vel()
    set_train_aug_path_ani()
    set_train_aug_path_vel()
