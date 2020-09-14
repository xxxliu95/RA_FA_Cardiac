from PIL import Image
import torchfile
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torch
import torch.nn as nn
import os
import torchvision.utils as vutils
import torchvision.transforms.functional as F
import numpy as np
import torch.nn.init as init
import torch.utils.data as data
import torch
import random
import xlrd


def get_un_data_loaders(batch_size):
    random.seed(14)
    train_size = 200


    train_loader, train_data = get_data_loader_folder('../labeled_data_nn', '../unlabeled_data_nn', batch_size, True, train_size)

    return train_loader, train_data

def get_data_loader_folder(input_folder1, input_folder2, batch_size, train, new_size=None, num_data=None, num_workers=4, mode=True):
    # transform_list = [transforms.ToTensor(),
    #                   transforms.Normalize([0.5],
    #                                        [0.5])]
    transform_list = [transforms.ToTensor()]
    # transform_list = [transforms.RandomCrop((new_size, new_size))] + transform_list
    transform_list = [transforms.Resize((new_size, new_size))] + transform_list
    # transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)

    if num_data:
        dataset = ImageFolder(input_folder1, input_folder2, transform=transform, num_data=num_data)
    else:
        dataset = ImageFolder(input_folder1, input_folder2, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader, dataset

# def default_loader(path):
#     return Image.open(path)

def default_loader(path):
    return np.load(path)['arr_0']

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)

    return images

class ImageFolder(data.Dataset):
    def __init__(self, root1, root2, transform=None, num_data=None, return_paths=False,
                 loader=default_loader):

        print(root1)
        print(root2)

        temp_imgs1 = sorted(make_dataset(root1)) # make_dataset(root): a list
        temp_imgs2 = sorted(make_dataset(root2)) # make_dataset(root): a list


        imgs = []

        # add something here to index, such that can split the data
        # index = random.sample(range(len(temp_img)), len(temp_mask))

        for i in range(len(temp_imgs1)):
            imgs.append(temp_imgs1[i])

        for i in range(len(temp_imgs2)):
            imgs.append(temp_imgs2[i])

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root1 + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.new_size = 224
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path_img = self.imgs[index]
        img = self.loader(path_img)

        img -= img.min()
        img /= img.max()
        img = img.astype('float32')

        img_tensor = F.to_tensor(img)

        img_size = img_tensor.size()
        left_size = 0
        top_size = 0
        right_size = 0
        bot_size = 0
        if img_size[-2] < self.new_size:
            top_size = (self.new_size - img_size[-2]) // 2
            bot_size = (self.new_size - img_size[-2]) - top_size
        if img_size[-1] < self.new_size:
            left_size = (self.new_size - img_size[-1]) // 2
            right_size = (self.new_size - img_size[-1]) - left_size

        transform_list = [transforms.Normalize([0.5], [0.5])]
        transform_list = [transforms.ToTensor()] + transform_list
        transform_list = [transforms.CenterCrop((self.new_size, self.new_size))] + transform_list
        transform_list = [transforms.Pad((left_size, top_size, right_size, bot_size))] + transform_list
        transform_list = [transforms.ToPILImage()] + transform_list
        transform = transforms.Compose(transform_list)

        img = transform(img)

        return img # pytorch: N,C,H,W

    def __len__(self):
        return len(self.imgs)

