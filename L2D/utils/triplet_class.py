from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from utils.dataset_class import MyDataset



def select_triplet(out_id = 0, in_id = 0,paths = None, datatype=None):
    '''
    choose a triad from triplet_dir according to its outer and inner index and paths:triple_train or triple_test
    '''

    path = '/data/dengx/counterfactual/' + datatype + '/triad/' + paths
    anchor_path = path + '/{}/{}.jpg'.format(out_id,in_id*3)
    positive_path = path + '/{}/{}.jpg'.format(out_id,in_id*3+1)
    negative_path = path + '/{}/{}.jpg'.format(out_id,in_id*3+2)

    return anchor_path,positive_path,negative_path


class Tripletset(Dataset):

    def __init__(self, types='triple_train', data='animals', outer=1, inner=1, transform=None):
       
        super().__init__()
        
        self.triplets = [select_triplet(o,i,types,data) for o in range(outer) for i in range(inner)]
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


triplet_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

a_train_triplets = Tripletset(types='triple_train',data='animals',outer=5318,inner=9*2,transform=triplet_transform)
a_test_triplets = Tripletset(types='triple_test',data='animals',outer= 2524,inner=9*2,transform=triplet_transform)

v_train_triplets = Tripletset(types='triple_train',data='vehicles',outer=4332,inner=7*2,transform=triplet_transform)
v_test_triplets = Tripletset(types='triple_test',data='vehicles',outer=2073,inner=7*2,transform=triplet_transform)

a_valid_counter = MyDataset(data_txt='./Cgn/cgn_data/counterfactual_data/retroset/ani/valid_set.txt',transform=triplet_transform)
a_test_counter = MyDataset(data_txt='./Cgn/cgn_data/counterfactual_data/retroset/ani/test_set.txt',transform=triplet_transform)

v_valid_counter = MyDataset(data_txt='./Cgn/cgn_data/counterfactual_data/retroset/vel/valid_set.txt',transform=triplet_transform)
v_test_counter = MyDataset(data_txt='./Cgn/cgn_data/counterfactual_data/retroset/vel/test_set.txt',transform=triplet_transform)
