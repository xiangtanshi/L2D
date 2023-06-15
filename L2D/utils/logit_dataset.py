'''
construct a new dataset whose elements are logits of the output of backbone that is well trained
'''
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class LogitSet(Dataset):

    def __init__(self, data_txt, transform=None):
        super().__init__()
        fh = open(data_txt, 'r')
        logits = []
        # data_txt is the file that contains the path and label of feature tensors
        for line in fh:
            line = line.rstrip('\n')
            words = line.split('$')
            logits.append((words[0], int(words[1])))
        # words[0] is tensor path, words[1] is tensor label
        self.logits = logits
        self.transform = transform


    def __getitem__(self, index):
        path, label = self.logits[index]
        try:
            feature = np.load(path)
            # feature = feature.transpose()
            if self.transform is not None:
                feature = self.transform(feature)
            return feature, label
        except:
            print(path)


    def __len__(self):
        #return the size of the dataset
        return len(self.logits)

transform = transforms.Compose([
    transforms.ToTensor()
]
)

a_valid_logit = LogitSet(data_txt='./Cgn/cgn_data/counterfactual_data/logit/ani/valid_logit.txt',transform=transform)
a_test_logit = LogitSet(data_txt='./Cgn/cgn_data/counterfactual_data/logit/ani/test_logit.txt',transform=transform)

v_valid_logit = LogitSet(data_txt='./Cgn/cgn_data/counterfactual_data/logit/vel/valid_logit.txt',transform=transform)
v_test_logit = LogitSet(data_txt='./Cgn/cgn_data/counterfactual_data/logit/vel/test_logit.txt',transform=transform)

