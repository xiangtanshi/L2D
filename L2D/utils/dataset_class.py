'''
All datasets are subclasses of torch.utils.data.Dataset i.e,
 they have __getitem__ and __len__ methods implemented.
 Hence, they can all be passed to a torch.utils.data.DataLoader which can load multiple samples parallelly using torch.multiprocessing workers.
We construct our dataset class here.
'''

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import sys



class MyDataset(Dataset):

    def __init__(self, data_txt, transform=None):
        super().__init__()
        fh = open(data_txt, 'r')
        imgs = []
        # data_txt is the file that contains the path and label of images
        for line in fh:
            line = line.rstrip('\n')
            words = line.split('$')     
            imgs.append((words[0], int(words[1])))
        #words[0] is the image path，words[1] is the image label
        self.imgs = imgs
        self.transform = transform


    def __getitem__(self, index):
        path, label = self.imgs[index]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        except:
            print(path)


    def __len__(self):
        #return the size of the dataset
        return len(self.imgs)



#data augmentation for training set
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224,224),(0.8,1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.4),
    # transforms.RandomGrayscale(0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
]
)

#data preprocessing for validation and test set
test_transform = transforms.Compose([
    # transforms.Resize((256,256)),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])


# a_ani_training = MyDataset(data_txt='Dataset/ood/Ani_train.txt', transform=test_transform)
# a_ani_test_pure = MyDataset(data_txt='Cgn/cgn_data/counterfactual_data/mask/ani/test_pure.txt', transform=test_transform)

a_ani_train = MyDataset(data_txt='./Dataset/ood/Ani_train.txt', transform=train_transform)
a_ani_valid = MyDataset(data_txt='./Dataset/ood/Ani_valid.txt', transform=test_transform)
a_ani_test = MyDataset(data_txt='./Dataset/ood/Ani_test.txt', transform=test_transform)

a_vel_train = MyDataset(data_txt='./Dataset/ood/Vel_train.txt', transform=train_transform)
a_vel_valid = MyDataset(data_txt='./Dataset/ood/Vel_valid.txt', transform=test_transform)
a_vel_test = MyDataset(data_txt='./Dataset/ood/Vel_test.txt', transform=test_transform)

# with counterfactual data augmentation
a_ani_train1 = MyDataset(data_txt='./Dataset/ood/Ani_train1.txt', transform=train_transform)
a_vel_train1 = MyDataset(data_txt='./Dataset/ood/Vel_train1.txt', transform=train_transform)

def main(argv):
    # there are about 10 images in NICO that is broken and can not read with PIL, remove them from our datasets.
    file_name = argv[1]
    # remove the damaged image that can not read
    file = open('./Dataset/ood/'+file_name,'r')

    index = 1
    for item in file:
        item = item.rstrip('\n')
        path = item.split('$')[0]
        try:
            img = Image.open(path)
        except:
            print('corrupt img', index)
        index = index + 1
    
    file.close()


if __name__ == '__main__':
    main(sys.argv)

    #run python utils/dataset_class.py Ani_train.txt to implement it