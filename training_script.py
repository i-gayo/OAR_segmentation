import numpy as np
import torch
from utils import * 
from unet_network import * 
from torch.utils.data import Dataset, DataLoader, RandomSampler 

if __name__ == '__main__':

    # TODO - fix dataloader to take in n num of images (30) at a time rather than the whole dataset
    FOLDER_NAME = '../Data_folder'
    train_loader = DataLoader(Image_dataloader(FOLDER_NAME, mode = 'train'), batch_size = 4, shuffle = True)
    val_loader = DataLoader(Image_dataloader(FOLDER_NAME, mode = 'val'), batch_size = 1)

    # Initialising the model 
    model = UNet_3D(1, 1)

    # Initialising the training procedure 
    experiment_name = 'baseline'
    train_loss, val_loss, train_iou, val_iou = train(model, train_loader, val_loader, \
        num_epochs = 10, use_cuda = True, save_folder = experiment_name)

    print('Chicken')
    
