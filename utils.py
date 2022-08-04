import torch
import numpy as np 
import pandas as pd 
from matplotlib import pyplot 
import os 
from torch.utils.data import Dataset, DataLoader, RandomSampler 
import SimpleITK as sitk
from unet_network import UNet_3D

###### DATALOADER ######
class ImageReader:

    def __call__(self, file_path, require_sitk_img = False):
        image_vol = sitk.ReadImage(file_path)
        image_vol = sitk.GetArrayFromImage(image_vol)
        image_size = np.shape(image_vol)

        if require_sitk_img: 
            sitk_image_vol = sitk.ReadImage(file_path, sitk.sitkUInt8)
            return image_vol, sitk_image_vol

        else:        
            return image_vol

class Image_dataloader(Dataset):

    def __init__(self, folder_name, mode = 'train', use_all = False):
        
        self.folder_name = folder_name
        self.mode = mode

        # Obtain list of patient names with multiple lesions 
        df_dataset = pd.read_csv('./patient_data_multiple_lesions.csv')
        self.all_file_names = df_dataset['patient_name'].tolist()
        self.num_lesions = df_dataset[' num_lesions'].tolist()

        # Train with all patients 
        if use_all:

            size_dataset = len(self.all_file_names)

            train_len = int(size_dataset * 0.7) 
            test_len = int(size_dataset * 0.2) 
            val_len = size_dataset - (train_len + test_len)

            # both test and val have simila rnumber of lesions (mean = 2.4 lesions)
            self.train_names = self.all_file_names[0:train_len]
            self.val_names = self.all_file_names[train_len:train_len + val_len]
            self.test_names = self.all_file_names[train_len + val_len:]

        # Only train with 100 patients, validate with 15 and validate with 30 : all ahve mean num lesions of 2.6
        else:

            size_dataset = 50

            train_len = int(size_dataset * 0.7) 
            test_len = int(size_dataset * 0.2) 
            val_len = size_dataset - (train_len + test_len)

            self.train_names = self.all_file_names[0:105]
            self.val_names = self.all_file_names[105:120]
            self.test_names = self.all_file_names[120:150]

        self.dataset_len = {'train' : train_len, 'test': test_len, 'val' : val_len}

        # Folder names
        self.lesion_folder = os.path.join(folder_name, 'lesion')
        self.mri_folder = os.path.join(folder_name, 't2w')
        self.rectum_folder = os.path.join(folder_name, 'rectum_mask')

    def _get_patient_list(self, folder_name):
        """
        A function that lists all the patient names
        """
        all_file_names = [f for f in os.listdir(folder_name) if not f.startswith('.')]

        return all_file_names

    def _normalise(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """

        max_img = np.max(img)
        min_img = np.min(img)

        #Normalise values between 0 to 1
        normalised_img =  ((img - min_img)/(max_img - min_img)) 

        return normalised_img.astype(np.float32)

    def __len__(self):
        return self.dataset_len[self.mode]
 
    def __getitem__(self, idx):

        if self.mode == 'train':
            #idx_ = idx
            patient_name = self.train_names[idx]

        elif self.mode == 'val':
            #idx_ = idx + self.dataset_len['train']
            patient_name = self.val_names[idx]

        elif self.mode == 'test':
            #idx_ = idx + self.dataset_len['train'] + self.dataset_len['val']
            patient_name = self.test_names[idx]

        # Read prostate mask, lesion mask, prostate mask separately using ImageReader    
        #patient_name = self.all_file_names[idx_]
        read_img = ImageReader()
        
        mri_vol = np.transpose(self._normalise(read_img(os.path.join(self.mri_folder, patient_name))), [1, 2, 0])
        lesion_mask = np.transpose((read_img(os.path.join(self.lesion_folder, patient_name))), [1, 2, 0])
        rectum_mask = np.transpose(self._normalise(read_img(os.path.join(self.rectum_folder, patient_name))), [1, 2, 0])
        
        # Get rectum positions
        #rectum_pos = self._get_rectum_pos(patient_name) 
        rectum_pos = 0 
        sitk_img_path = os.path.join(self.lesion_folder, patient_name)

        return mri_vol, rectum_mask, lesion_mask, sitk_img_path , rectum_pos, patient_name

###### LOSS FUNCTIONS, DICE SCORE METRICS ######

def dice_loss(gt_mask, pred_mask):
    '''
    y_pred, y_true -> [N, C=1, D, H, W]
    '''
    numerator = torch.sum(gt_mask*pred_mask, dim=(2,3,4)) * 2
    denominator = torch.sum(gt_mask, dim=(2,3,4)) + torch.sum(pred_mask, dim=(2,3,4)) + 1e-6
    
    return torch.mean(1. - (numerator / denominator))

def dice_score(gt_mask, pred_mask):
    """
    Dice score metric 
    """
    numerator = torch.sum(gt_mask*pred_mask, dim=(2,3,4)) * 2
    denominator = torch.sum(gt_mask, dim=(2,3,4)) + torch.sum(pred_mask, dim=(2,3,4)) + 1e-6

    dice = torch.mean((numerator / denominator))
    
###### Training scripts ######
def train(prostate_dataloader, num_epochs = 10):
    
    """
    A function that performs the training and validation loop 

    :params:
    :prostate_dataloader: torch dataloader that can load images and masks in
    :num_epochs: (int) Number of epochs to train for 
    """
    
    model = UNet_3D(1, 1)
    loss_fn = dice_loss()
    optimiser_fn = torch.optim.Adam(model.parameters(), lr=1e-4)


