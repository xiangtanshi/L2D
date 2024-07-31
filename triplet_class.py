from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from utils.dataset_class import MyDataset



def select_triplet(out_id = 0, in_id = 0, types='val1'):
    '''
    choose a triad from triplet_dir according to its outer and inner index and paths:triple_train or triple_test
    '''
    if in_id > 15:
        raise ValueError('only create 16 triplets for each sample, the index should be less than 16.')
    path = './datas/counter/triad/triple_{}'.format(types) 
    anchor_path = path + '/{}/{}.jpg'.format(out_id,in_id*3)
    positive_path = path + '/{}/{}.jpg'.format(out_id,in_id*3+1)
    negative_path = path + '/{}/{}.jpg'.format(out_id,in_id*3+2)

    return anchor_path,positive_path,negative_path

def select_counter(idx = 0, types='val1'):
    '''
    read the counterfactual images for each class for the given image, 0-> original image, 1:17,18:34,35:51-> counterfactual samples belong to class 0-16
    '''
    path = './datas/counter/triad/{}/{}/'.format(types,idx) 
    counter_path_lists = [path + '{}.jpg'.format(ids) for ids in range(52)]
    return counter_path_lists

class Tripletset(Dataset):

    def __init__(self, types='val1', transform=None):
       
        super().__init__()
        if types == 'val1':
            self.triplets = [select_triplet(o,i,types) for o in range(1475) for i in range(16)]
        elif types == 'con2':
            self.triplets = [select_triplet(o,i,types) for o in range(14564) for i in range(16)]
        self.transform = transform


    def __getitem__(self, index):
        # return one triplet a time
        triplet = self.triplets[index]
        try:
            anchor = self.transform(Image.open(triplet[0]).convert('RGB'))
            positive = self.transform(Image.open(triplet[1]).convert('RGB'))
            negative = self.transform(Image.open(triplet[2]).convert('RGB'))
 
            return anchor,positive,negative
        except:
            raise ValueError('wrong triplet')


    def __len__(self):
        #return the size of the dataset
        return len(self.triplets)

class Counter_set(Dataset):

    def __init__(self, types='val1', transform=None):
       
        super().__init__()
        if types == 'val1':
            self.counters = [select_counter(o,types) for o in range(1475)]
        elif types == 'con2':
            self.counters = [select_counter(o,types) for o in range(14564)]
        self.transform = transform


    def __getitem__(self, index):
        # return one triplet a time
        counter = self.counters[index]
        # try:
        image_list = []
        for path in counter:
            image_list.append(self.transform(Image.open(path).convert('RGB')))

        return image_list
        # except:
        #     raise ValueError('Problems occur when reading the counterfactual samples for the image indexed as {}'.format(index))


    def __len__(self):
        #return the size of the dataset
        return len(self.counters)

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

# triplets for training and evaluation to check if the training is effective.
siamese_train = Tripletset(types='val1',transform=test_transform)
siamese_test = Tripletset(types='con2',transform=test_transform)

# counterfactual images used to calculate the counterfactual consensus
consensus_val = Counter_set(types='val1',transform=test_transform)
consensus_test = Counter_set(types='con2',transform=test_transform)


