import datetime
import argparse
from eval import eval_net
from tqdm import tqdm
import logging
from focal_loss import FocalLoss
from torch.utils.data import DataLoader, random_split

from mms_dataloader_re_aug import get_all_data_loaders
import models
import supervision
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from torch.utils.tensorboard import SummaryWriter

dir_checkpoint = 'checkpoints/'

def get_args():
    usage_text = (
        "SNet Pytorch Implementation"
        "Usage:  python train.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    #training details
    parser.add_argument('-e','--epochs', type= int, default=10, help='Number of epochs')
    parser.add_argument('-bs','--batch_size', type= int, default=1, help='Number of inputs per batch')
    parser.add_argument('-n','--name', type=str, default='default_name', help='The name of this train/test. Used when storing information.')
    parser.add_argument('-mn','--model_name', type=str, default='sdnet', help='Name of the model architecture to be used for training/testing.')
    parser.add_argument('-lr','--learning_rate', type=float, default='0.0001', help='The learning rate for model training')
    parser.add_argument('-wi','--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--save_path', type=str, default='checkpoints', help= 'Path to save model checkpoints')
    parser.add_argument('--decoder_type', type=str, default='film', help='Choose decoder type between FiLM and SPADE')
    #hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    parser.add_argument('--num_workers' ,type= int, default = 0, help='Number of workers to use for dataload')

    return parser.parse_args()

def train_net(device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True):

    #Model selection and initialization
    model_params = {
        'width': 224,
        'height': 224,
        'ndf': 64,
        'norm': "batchnorm",
        'upsample': "nearest",
        'num_classes': 3,
        'decoder_type': args.decoder_type,
        'anatomy_out_channels': 8,
        'z_length': 8,
        'num_mask_channels': 8,

    }
    model = models.get_model(args.model_name, model_params)
    models.initialize_weights(model, args.weight_init)
    model.to(device)

    train_loader, train_data = get_all_data_loaders(batch_size)


    n_val = int(len(train_data) * val_percent)
    n_train = len(train_data)
    train, val = random_split(train_data, [n_train-n_val, n_val])
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=True)

    l1_distance = nn.L1Loss().to(device)
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Reversed:  {reversed}
    ''')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # need to use a more useful lr_scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    focal = FocalLoss()

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for imgs, true_masks in train_loader:

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                true_masks = true_masks.to(device=device, dtype=mask_type)

                reco, z_out, mu_tilde, a_out, masks_pred, mu, logvar = model(imgs, true_masks, 'training')

                dice_loss_lv = supervision.dice_loss(masks_pred[:, 0, :, :], true_masks[:, 0, :, :])
                dice_loss_myo = supervision.dice_loss(masks_pred[:, 1, :, :], true_masks[:, 1, :, :])
                dice_loss_rv = supervision.dice_loss(masks_pred[:, 2, :, :], true_masks[:, 2, :, :])
                dice_loss_bg = supervision.dice_loss(masks_pred[:, 3, :, :], true_masks[:, 3, :, :])
                loss_dice = dice_loss_lv + dice_loss_myo + dice_loss_rv + dice_loss_bg
                loss_focal = focal(masks_pred[:, 0:3, :, :], true_masks[:, 0:3, :, :])
                regression_loss = l1_distance(mu_tilde, z_out)
                reco_loss = l1_distance(reco, imgs)

                loss = loss_focal+loss_dice+regression_loss+reco_loss
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Loss/loss_dice', loss_dice.item(), global_step)
                writer.add_scalar('Loss/dice_loss_lv', dice_loss_lv.item(), global_step)
                writer.add_scalar('Loss/dice_loss_myo', dice_loss_myo.item(), global_step)
                writer.add_scalar('Loss/dice_loss_rv', dice_loss_rv.item(), global_step)
                writer.add_scalar('Loss/dice_loss_bg', dice_loss_bg.item(), global_step)
                writer.add_scalar('Loss/loss_focal', loss_focal.item(), global_step)
                writer.add_scalar('Loss/loss_regression_loss', regression_loss.item(), global_step)
                writer.add_scalar('Loss/loss_reco', reco_loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1


        val_score, val_lv, val_myo, val_rv = eval_net(model, val_loader, device)
        scheduler.step(val_score)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        logging.info('Validation Dice Coeff: {}'.format(val_score))
        logging.info('Validation LV Dice Coeff: {}'.format(val_lv))
        logging.info('Validation MYO Dice Coeff: {}'.format(val_myo))
        logging.info('Validation RV Dice Coeff: {}'.format(val_rv))

        writer.add_scalar('Dice/val', val_score, epoch)
        writer.add_scalar('Dice/val_lv', val_lv, epoch)
        writer.add_scalar('Dice/val_myo', val_myo, epoch)
        writer.add_scalar('Dice/val_rv', val_rv, epoch)

        writer.add_images('images/val', imgs, epoch)
        writer.add_images('masks/val_true', true_masks[:,0:3,:,:], epoch)
        writer.add_images('masks/val_pred', masks_pred[:,0:3,:,:] > 0.5, epoch)

        if save_cp and (epoch + 1) > (4*(epochs // 5)):
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(model.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')
    writer.close()




if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    train_net(epochs=args.epochs,
              batch_size=args.batch_size,
              lr=args.learning_rate,
              device=device,
              val_percent=0.1)

