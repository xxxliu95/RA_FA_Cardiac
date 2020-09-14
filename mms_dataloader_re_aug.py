from PIL import Image
import torchfile
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torchvision.transforms.functional as F
import torch
import torch.nn as nn
import os
import torchvision.utils as vutils
import numpy as np
import torch.nn.init as init
import torch.utils.data as data
import torch
import random
import xlrd
import numpy.random

######################################################################################################
#Load excel information:
# cell_value(1,0) -> cell_value(175,0)
ex_file = 'M&Ms_Dataset_Information.xlsx'
wb = xlrd.open_workbook(ex_file)
sheet = wb.sheet_by_index(0)
# sheet.cell_value(r, c)

scan_re_A = np.load('../scan_re_A.npz')['arr_0'] #75x1
scan_re_B = np.load('../scan_re_B.npz')['arr_0'] #75x1
scan_re_C = np.load('../scan_re_C.npz')['arr_0'] #25x1

scan_re = []
for re in scan_re_A:
    if re in scan_re:
        pass
    else:
        scan_re.append(re)
for re in scan_re_B:
    if re in scan_re:
        pass
    else:
        scan_re.append(re)
for re in scan_re_C:
    if re in scan_re:
        pass
    else:
        scan_re.append(re)
scan_re = sorted(scan_re)
scan_re_np = np.array(scan_re)

num_pat = 0
vendor_A = []
vendor_B = []
for i in range(1, 176):
    if sheet.cell_value(i, 1)=='A':
        vendor_A.append(num_pat)
    elif sheet.cell_value(i, 1)=='B':
        vendor_B.append(num_pat)
    else:
        continue
    num_pat += 1


def get_all_data_loaders(batch_size, train_num_data=None):
    random.seed(14)
    train_size = 224

    train_loader, train_data = get_data_loader_folder('../labeled_mask_data_nn', batch_size, True, '../mask_3_nn', train_size, train_num_data)

    return train_loader, train_data

def get_data_loader_folder(input_folder, batch_size, train, labels_root, new_size=None, num_data=None, num_workers=4):
    if num_data:
        dataset = ImageFolder(input_folder, labels_root, new_size, num_data=num_data)
    else:
        dataset = ImageFolder(input_folder, labels_root, new_size)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader, dataset

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
    def __init__(self, root, labels_root, new_size, num_data=None, return_paths=False, loader=default_loader):
        temp_imgs = sorted(make_dataset(root))  # make_dataset(root): a list
        temp_masks = sorted(make_dataset(labels_root))  # make_dataset(root): a list
        temp_re = []

        for j in range(len(temp_imgs)):
            for i in range(len(vendor_A)):
                if temp_imgs[j][24:27] == ('00' + str(vendor_A[i])) or temp_imgs[j][24:27] == (
                        '0' + str(vendor_A[i])) or temp_imgs[j][24:27] == str(vendor_A[i]):
                    temp_re.append(scan_re_A[i])
            for i in range(len(vendor_B)):
                if temp_imgs[j][24:27] == ('00' + str(vendor_B[i])) or temp_imgs[j][24:27] == (
                        '0' + str(vendor_B[i])) or temp_imgs[j][24:27] == str(vendor_B[i]):
                    temp_re.append(scan_re_B[i])

        imgs = []

        if num_data:
            itr = num_data
        else:
            itr = len(temp_imgs)
        for i in range(itr):
            imgs.append((temp_imgs[i], temp_masks[i]))
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.new_size = new_size
        self.imgs = imgs
        self.re = temp_re
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path_img = self.imgs[index][0]
        path_mask = self.imgs[index][1]
        img_re = self.re[index]
        img = self.loader(path_img)  # numpy, HxW, numpy.Float64
        mask = Image.open(path_mask)  # numpy, HxWx3
        img -= img.min()
        img /= img.max()
        img = img.astype('float32')

        img_tensor = F.to_tensor(img)
        img_size = img_tensor.size()
        rand_re = numpy.random.choice(scan_re_np)
        resize_order = img_re / rand_re
        resize_size_h = int(img_size[-2] * resize_order)
        resize_size_w = int(img_size[-1] * resize_order)

        left_size = 0
        top_size = 0
        right_size = 0
        bot_size = 0
        if resize_size_h < self.new_size:
            top_size = (self.new_size - resize_size_h) // 2
            bot_size = (self.new_size - resize_size_h) - top_size
        if resize_size_w < self.new_size:
            left_size = (self.new_size - resize_size_w) // 2
            right_size = (self.new_size - resize_size_w) - left_size

        transform_list = [transforms.Normalize([0.5], [0.5])]
        transform_list = [transforms.ToTensor()] + transform_list
        transform_list = [transforms.CenterCrop((self.new_size, self.new_size))] + transform_list
        transform_list = [transforms.Pad((left_size, top_size, right_size, bot_size))] + transform_list
        transform_list = [transforms.Resize((resize_size_h, resize_size_w))] + transform_list
        transform_list = [transforms.ToPILImage()] + transform_list
        transform = transforms.Compose(transform_list)

        transform_mask_list = [transforms.ToTensor()]
        transform_mask_list = [transforms.CenterCrop((self.new_size, self.new_size))] + transform_mask_list
        transform_mask_list = [transforms.Pad((left_size, top_size, right_size, bot_size))] + transform_mask_list
        transform_mask_list = [transforms.Resize((resize_size_h, resize_size_w),
                                                 interpolation=Image.NEAREST)] + transform_mask_list
        transform_mask = transforms.Compose(transform_mask_list)

        img = transform(img)
        mask = transform_mask(mask)  # C,H,W
        mask_bg = (mask.sum(0) == 0).type_as(mask)  # H,W
        mask_bg = mask_bg.reshape((1, mask_bg.size(0), mask_bg.size(1)))
        mask = torch.cat((mask, mask_bg), dim=0)

        return img, mask  # pytorch: C,H,W

    def __len__(self):
        return len(self.imgs)
