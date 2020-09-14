import torch
import sys
import os
import configparser
import argparse
import nibabel as nib
import numpy as np
from unet_model import UNet
from utils import load_path, walk_path, load_phase, pre_transform, predict_img, post_transform, save_phase

def get_args():
    parser = argparse.ArgumentParser(description='Test the mnms dataset.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-inp', '--input_data_directory', type=str, default='', help='input_data_directory', dest='inp')
    parser.add_argument('-out', '--output_data_directory', type=str, default='', help='output_data_directory', dest='out')
    parser.add_argument('-g', '--gpu', dest='gpu', type=int, default=0, help='GPU')

    return parser.parse_args()


def submit_mnms(model_path, input_data_directory, output_data_directory, device):

    data_paths = load_path(input_data_directory)

    net = UNet(n_channels=1, n_classes=4, bilinear=True)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)

    for path in data_paths:
        ED_np, ES_np = load_phase(path) # HxWxF
        ED_masks = []
        ES_masks = []
        for i in range(ED_np.shape[2]):
            img_np = ED_np[:,:,i]
            img_tensor = pre_transform(img_np)
            img_tensor = img_tensor.to(device)

            mask = predict_img(net, img_tensor)

            mask = post_transform(img_np, mask[0:3,:,:])
            ED_masks.append(mask)

        for i in range(ES_np.shape[2]):
            img_np = ES_np[:, :, i]
            img_tensor = pre_transform(img_np)
            img_tensor = img_tensor.to(device)

            mask = predict_img(net, img_tensor)

            mask = post_transform(img_np, mask[0:3,:,:])
            ES_masks.append(mask)

        ED_masks = np.concatenate(ED_masks, axis=2)
        ES_masks = np.concatenate(ES_masks, axis=2)
        save_phase(ED_masks, ES_masks, output_data_directory, path)


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    submit_mnms(model_path='checkpoints/CP_epoch50.pth', input_data_directory=args.inp, output_data_directory=args.out, device=device)