from torchvision import transforms
from utils.dataset_class import MyDataset


# input of u2net is of size (batchsize, 3, 256, 256), we need to define a specific dataloader for it
cgn_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

cgn_transform_norm = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])   #for shape template
])

Trans_t_ani = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

Trans_t_vel = transforms.Compose([
    transforms.Resize((64,128)),
    transforms.ToTensor(),
])

simp_transform = transforms.Compose([
    transforms.ToTensor()
])

#data
a_train__data = MyDataset(data_txt='./Dataset/ood/Ani_train.txt', transform=cgn_transform)
a_valid__data = MyDataset(data_txt='./Dataset/ood/Ani_valid.txt', transform=cgn_transform)
a_test__data = MyDataset(data_txt='./Dataset/ood/Ani_test.txt', transform=cgn_transform)

v_train__data = MyDataset(data_txt='./Dataset/ood/Vel_train.txt', transform=cgn_transform)
v_valid__data = MyDataset(data_txt='./Dataset/ood/Vel_valid.txt', transform=cgn_transform)
v_test__data = MyDataset(data_txt='./Dataset/ood/Vel_test.txt', transform=cgn_transform)
# no normalization, raw image
datas = [a_train__data,a_valid__data,a_test__data,v_train__data,v_valid__data,v_test__data]

a_train__data1 = MyDataset(data_txt='./Dataset/ood/Ani_train.txt', transform=cgn_transform_norm)
a_valid__data1 = MyDataset(data_txt='./Dataset/ood/Ani_valid.txt', transform=cgn_transform_norm)
a_test__data1 = MyDataset(data_txt='./Dataset/ood/Ani_test.txt', transform=cgn_transform_norm)

v_train__data1 = MyDataset(data_txt='./Dataset/ood/Vel_train.txt', transform=cgn_transform_norm)
v_valid__data1 = MyDataset(data_txt='./Dataset/ood/Vel_valid.txt', transform=cgn_transform_norm)
v_test__data1 = MyDataset(data_txt='./Dataset/ood/Vel_test.txt', transform=cgn_transform_norm)
# with normalization, which helps to effeciently extract shape from the image
datas1 = [a_train__data1,a_valid__data1,a_test__data1,v_train__data1,v_valid__data1,v_test__data1]

# mask
v_train__shape = MyDataset(data_txt='./Cgn/cgn_data/counterfactual_data/mask/vel/train_mask.txt', transform=simp_transform)
v_valid__shape = MyDataset(data_txt='./Cgn/cgn_data/counterfactual_data/mask/vel/valid_mask.txt', transform=simp_transform)
v_test__shape = MyDataset(data_txt='./Cgn/cgn_data/counterfactual_data/mask/vel/test_mask.txt', transform=simp_transform)

a_train__shape = MyDataset(data_txt='./Cgn/cgn_data/counterfactual_data/mask/ani/train_mask.txt', transform=simp_transform)
a_valid__shape = MyDataset(data_txt='./Cgn/cgn_data/counterfactual_data/mask/ani/valid_mask.txt', transform=simp_transform)
a_test__shape = MyDataset(data_txt='./Cgn/cgn_data/counterfactual_data/mask/ani/test_mask.txt', transform=simp_transform)

masks = [a_train__shape,a_valid__shape,a_test__shape,v_train__shape,v_valid__shape,v_test__shape]

background = MyDataset(data_txt='./Cgn/cgn_data/counterfactual_data/background.txt', transform=simp_transform)

# foreground
vel_ft = MyDataset(data_txt='./Cgn/cgn_data/counterfactual_data/foreground/vel_ft.txt', transform=Trans_t_vel)
vel_tx = MyDataset(data_txt='./Cgn/cgn_data/counterfactual_data/foreground/vel_tx.txt', transform=simp_transform)
ani_ft = MyDataset(data_txt='./Cgn/cgn_data/counterfactual_data/foreground/ani_ft.txt', transform=Trans_t_ani)
ani_tx = MyDataset(data_txt='./Cgn/cgn_data/counterfactual_data/foreground/ani_tx.txt', transform=simp_transform)
