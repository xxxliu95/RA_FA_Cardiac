import torch
import sys
import os
import nibabel as nib
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image

def walk_path(dir):
    dir = dir+'mnms'
    paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dirs, _ in sorted(os.walk(dir)):
        for name in dirs:
            paths.append(os.path.join(root, name))
            print(paths)
    return paths

def load_path(input_data_directory):
    return walk_path(input_data_directory)

def load_phase(path):
    ED_path = os.path.join(path, path[-6:-1]+path[-1]+'_sa_ED.nii.gz')
    ES_path =os.path.join(path, path[-6:-1]+path[-1]+'_sa_ES.nii.gz')

    img_ED = nib.load(ED_path)
    img_np_ED = img_ED.get_fdata()

    img_ES = nib.load(ES_path)
    img_np_ES = img_ES.get_fdata()

    return img_np_ED, img_np_ES

def pre_transform(img_in):
    img = np.array(img_in)
    img -= img.min()
    img /= img.max()
    img = img.astype('float32')

    new_size = 224
    img_size = img.shape

    left_size = 0
    top_size = 0
    right_size = 0
    bot_size = 0
    if img_size[-2] < new_size:
        top_size = (new_size - img_size[-2]) // 2
        bot_size = (new_size - img_size[-2]) - top_size
    if img_size[-1] < new_size:
        left_size = (new_size - img_size[-1]) // 2
        right_size = (new_size - img_size[-1]) - left_size

    transform_list = [transforms.Normalize([0.5], [0.5])]
    transform_list = [transforms.ToTensor()] + transform_list
    transform_list = [transforms.CenterCrop((new_size, new_size))] + transform_list
    transform_list = [transforms.Pad((left_size, top_size, right_size, bot_size))] + transform_list
    transform_list = [transforms.ToPILImage()] + transform_list
    transform = transforms.Compose(transform_list)
    img = transform(np.array(img))
    return img

def predict_img(net,
                full_img,
                out_threshold=0.5):
    net.eval()

    img = full_img.unsqueeze(0)
    img = img

    with torch.no_grad():
        output = net(img)

        probs = output.squeeze(0)

        full_mask = probs.squeeze()

    return full_mask > out_threshold


def post_transform(img, mask):
    img = np.array(img)
    img_size = img.shape

    new_size = 224
    height = new_size
    weight = new_size

    # H
    if img_size[-2] < new_size:
        height = img_size[-2]
    # W
    if img_size[-1] < new_size:
        weight = img_size[-1]

    left_size = 0
    top_size = 0
    right_size = 0
    bot_size = 0
    if height < img_size[-2]:
        top_size = (img_size[-2] - height) // 2
        bot_size = (img_size[-2] - height) - top_size
    if weight < img_size[-1]:
        left_size = (img_size[-1] - weight) // 2
        right_size = (img_size[-1]- weight) - left_size


    ### Here: do the anti-transformation for the mask.
    ### if img size is smaller -> crop to that size
    ### if img size is larger -> zero padding

    transform_list = [transforms.CenterCrop((height, weight))]
    transform_list = transform_list + [transforms.Pad((left_size, top_size, right_size, bot_size))]
    transform_list = transform_list + [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)

    mask = np.array(mask.cpu().numpy())
    mask = mask[0, :, :] * 1 + mask[1, :, :] * 2 + mask[2, :, :] * 3
    mask = mask.astype('float32')

    mask = Image.fromarray(mask, 'F')

    mask = transform(mask)
    ### tensor to np array.
    mask = mask.cpu().numpy()
    mask = np.transpose(mask, (1,2,0))

    return mask # one channel numpy array

def safe_mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def save_phase(ED_np, ES_np, output_data_directory, path):

    safe_mkdir(output_data_directory)

    ED_img_path = os.path.join(path, path[-6:-1]+path[-1]+'_sa_ED.nii.gz')
    ES_img_path =os.path.join(path, path[-6:-1]+path[-1]+'_sa_ES.nii.gz')

    img_ED = nib.load(ED_img_path)
    img_ES = nib.load(ES_img_path)

    ED_out_path = os.path.join(output_data_directory, path[-6:-1]+path[-1]+'_sa_ED.nii.gz')
    ES_out_path = os.path.join(output_data_directory, path[-6:-1]+path[-1]+'_sa_ES.nii.gz')
    mask_ED = nib.Nifti1Image(ED_np, img_ED.affine, img_ED.header)
    mask_ES = nib.Nifti1Image(ES_np, img_ES.affine, img_ES.header)

    nib.save(mask_ED, ED_out_path)
    nib.save(mask_ES, ES_out_path)