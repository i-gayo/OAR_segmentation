import torch
import numpy as np 
import pandas as pd 
from matplotlib import pyplot 
import os 
from torch.utils.data import Dataset, DataLoader, RandomSampler 
import SimpleITK as sitk
from unet_network import UNet_3D
from torch.utils.tensorboard import SummaryWriter
import h5py 

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

        return mri_vol, rectum_mask, patient_name

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
    
def iou_score(gt_mask, pred_mask):
    """
    IOU metric
    """
    numerator = torch.sum(gt_mask*pred_mask, dim=(2,3,4)) 
    combined_masks = torch.logical_or(gt_mask, pred_mask)
    denominator = torch.sum(combined_masks, dim= (2,3,4)) + 1e-6

    #combined_masks = gt_mask + pred_mask
    #combined_masks[combined_masks != 0] = 1:
    #denominator = torch.sum(combined_masks, (dim = 2,3,4)) + 1e-6
    #denominator = torch.sum(gt_mask, dim=(2,3,4)) + torch.sum(pred_mask, dim=(2,3,4)) + 1e-6

    iou = torch.mean((numerator / denominator))

    return iou 

###### Training scripts ######
def validate(val_dataloader, model, use_cuda = True, save_path = 'model_1'):

    # Set to evaluation mode 
    model.eval()
    iou_vals_eval = [] 
    loss_vals_eval = [] 

    for idx, (image, label, patient_name) in enumerate(val_dataloader):
        

        if use_cuda:
            image, label = image.cuda(), label.cuda()

        with torch.no_grad():
            output = model(image)
            loss = dice_loss(label, output) 
            iou = iou_score(label, output)                
            
            loss_vals_eval.append(loss)
            iou_vals_eval.append(iou)
    
    mean_iou = torch.mean(iou_vals_eval)
    mean_loss = torch.mean(loss_vals_eval)

    # Save image, labels and outputs into h5py files
    img_name = patient_name + '_rectum.nii.gz'
    img_path = os.path.join(save_path, img_name)
    sitk.WriteImage(image, img_path)


    return mean_loss, mean_iou
    
def train(train_dataloader, val_dataloader, num_epochs = 10, use_cuda = False, save_folder = 'model_1'):
    
    """
    A function that performs the training and validation loop 

    :params:
    :prostate_dataloader: torch dataloader that can load images and masks in
    :num_epochs: (int) Number of epochs to train for 
    """
    
    model = UNet_3D(1, 1)
    writer = SummaryWriter() 
    os.mkdir(save_folder, exist_ok = True) 

    if use_cuda:
        model.cuda()

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
    step = 0 
    freq_print = 8
    freq_eval = 4
    all_loss_train = np.zeros(num_epochs,1)
    all_iou_train = np.zeros(num_epochs, 1)
    all_loss_val = [] 
    all_iou_val = []
    best_loss = np.inf 
    best_iou = 0 

    for epoch_no in range(num_epochs):
        
        iou_vals = []
        loss_vals = [] 

        # Initialise training loop
        for idx, (images, labels) in enumerate(train_dataloader):

            # Move to GPU 
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()

            # Training steps 
            optimiser.zero_grad()
            pred_masks = model(images)  # Obtain predicted segmentation masks 
            loss = dice_loss(labels, pred_masks) # Compute the masks 
            loss.backward() # Backward propagation of gradients with respect to loss 
            optimiser.step() 

            # Compute metrics for each mini batch and append to statistics for each epoch
            iou = iou_score(labels, pred_masks)
            iou_vals.append(iou_vals.item())
            loss_vals.append(loss.item())

            # Print loss every nth minibatch and dice score 
            if idx % freq_print == 0: 
                print(f'Epoch {epoch_no} minibatch {idx} : loss : {loss.item():05f}, IOU score : {iou.item():05f}')
            
            
        # Obtain mean dice loss and IOU over this epoch, save to tensorboard 
        iou_epoch = torch.mean(torch.tensor(iou_vals))
        loss_epoch = torch.mean(torch.tensor(loss_vals))
        print(f'Epoch : {epoch_no} Average loss : {loss_epoch:5f} average IOU {iou_epoch:5f}')

        # Save for all_loss_train
        all_loss_train[loss_epoch] = loss_epoch
        all_iou_train[iou_epoch] = iou_epoch 
        
        #Tensorboard saving 
        writer.add_scalar('Loss/train', loss_epoch, epoch_no)
        writer.add_scalar('IOU/train', iou_epoch, epoch_no)
    
        # Save newest model 
        train_model_path = os.path.join(save_folder, 'train_model.pth')
        torch.save(model.state_dict(), train_model_path)

        # Validate every nth epoch and save every nth mini batch 
        if epoch_no % freq_eval == 0: 

            # Set to evaluation mode 
            mean_loss, mean_iou = validate(val_dataloader, model, use_cuda = True)
            print(f'Validation loss: {epoch_no} Average loss : {mean_loss:5f} average IOU {mean_iou:5f}')
            all_loss_val.append(mean_loss)
            all_iou_val.append(mean_iou)

            #Tensorboard saving
            writer.add_scalar('Loss/val', mean_loss, epoch_no)
            writer.add_scalar('IOU/val', mean_iou, epoch_no)

            if mean_loss < best_loss: 
                
                # Save best model as best validation model 
                val_model_path = os.path.join(save_folder, 'best_val_model.pth')
                torch.save(model.state_dict(), val_model_path)
        
                # Use as new best loss
                best_loss = mean_loss 
            
        # Save images





            
















