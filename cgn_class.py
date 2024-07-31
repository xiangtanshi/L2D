from torchvision import transforms
from utils.dataset_class import MyDataset


cgn_transform_norm = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])   #for shape template
])

cgn_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

trans_resize = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

trans_resize_1 = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])

simp_transform = transforms.Compose([
    transforms.ToTensor()
])

# for mask extraction, the normalization are necessary for accurate and meaningful foreground object shape recognition
con1__data = MyDataset(data_txt='./datas/NICO_paths/context1.txt', transform=cgn_transform_norm)
val1__data = MyDataset(data_txt='./datas/NICO_paths/val1.txt', transform=cgn_transform_norm)
con2__data = MyDataset(data_txt='./datas/NICO_paths/context2.txt', transform=cgn_transform_norm)
val2__data = MyDataset(data_txt='./datas/NICO_paths/val2.txt', transform=cgn_transform_norm)

# for counterfactual texture replacement, normalization is not used here
con1 = MyDataset(data_txt='./datas/NICO_paths/context1.txt', transform=cgn_transform)
val1 = MyDataset(data_txt='./datas/NICO_paths/val1.txt', transform=cgn_transform)
con2 = MyDataset(data_txt='./datas/NICO_paths/context2.txt', transform=cgn_transform)
val2 = MyDataset(data_txt='./datas/NICO_paths/val2.txt', transform=cgn_transform)


# stored mask files, shape are already of shape 256x256, resize is not needed
# con1__shape = MyDataset(data_txt='./datas/counter/mask/con1-mask.txt', transform=simp_transform)
val1__shape = MyDataset(data_txt='./datas/counter/mask/val1-mask.txt', transform=simp_transform)
con2__shape = MyDataset(data_txt='./datas/counter/mask/con2-mask.txt', transform=simp_transform)
# val2__shape = MyDataset(data_txt='./datas/counter/mask/val2-mask.txt', transform=simp_transform)


# foreground
feature = MyDataset(data_txt='./datas/counter/foreground/feature.txt', transform=simp_transform)
feature_8 = MyDataset(data_txt='./datas/counter/foreground/feature-8.txt', transform=simp_transform)
texture = MyDataset(data_txt='./datas/counter/foreground/texture.txt', transform=trans_resize)
texture_1 = MyDataset(data_txt='./datas/counter/foreground/texture.txt', transform=trans_resize_1)
