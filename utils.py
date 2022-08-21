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
        names_path = os.path.join(folder_name, 'patient_names.csv')
        df_dataset = pd.read_csv(names_path)
        self.all_file_names = df_dataset['patient_name'].tolist()
       

        # Train with all patients 
        size_dataset = len(self.all_file_names)

        train_len = 16
        test_len = 0
        val_len = 4
        self.dataset_len = {'train': train_len, 'test' :test_len, 'val' : val_len}

        # both test and val have simila rnumber of lesions (mean = 2.4 lesions)
        self.train_names = self.all_file_names[0:train_len]
        self.val_names = self.all_file_names[train_len:]
        #self.test_names = self.all_file_names[train_len + val_len:]

        # Folder names
        self.prostate_folder = os.path.join(folder_name, 'prostate_mask')
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
        prostate_mask = np.transpose((read_img(os.path.join(self.prostate_folder, patient_name))), [1, 2, 0])
        rectum_name = patient_name.split('.')[0] + '.nrrd'
        #print(rectum_name)
        rectum_mask = np.transpose(self._normalise(read_img(os.path.join(self.rectum_folder, rectum_name))), [1, 2, 0])
        
        # Return as tensor 
        mri_vol = torch.from_numpy(mri_vol).unsqueeze(0)
        prostate_mask = torch.from_numpy(prostate_mask).unsqueeze(0) 
        rectum_mask =    torch.from_numpy(rectum_mask).unsqueeze(0) 

        # Pad tensors from (200,200,96) -> (224, 224, 96) for unet encoder/decoder compatibility 
        mri_vol = torch.nn.functional.pad(mri_vol[:, ::2, ::2, ::2], (0, 0, 6, 6,6, 6))
        prostate_mask = torch.nn.functional.pad(prostate_mask[:, ::2, ::2, ::2], (0, 0, 6, 6, 6, 6))
        rectum_mask = torch.nn.functional.pad(rectum_mask[:, ::2, ::2, ::2], (0, 0, 6, 6, 6, 6))

        return mri_vol, rectum_mask, patient_name

###### LOSS FUNCTIONS, DICE SCORE METRICS ######

def dice_loss(gt_mask, pred_mask):
    '''
    y_pred, y_true -> [N, C=1, D, H, W]
    '''
    # Cropped back to original dimensions 
    gt_cropped = gt_mask[:,:,12:-12, 12:-12, :]
    pred_cropped = pred_mask[:,:,12:-12, 12:-12, :]

    numerator = torch.sum(gt_cropped*pred_cropped, dim=(2,3,4)) * 2
    denominator = torch.sum(gt_cropped, dim=(2,3,4)) + torch.sum(pred_cropped, dim=(2,3,4)) + 1e-6

    dice_loss = torch.mean(1. - (numerator / denominator))
    
    return dice_loss

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

    # CRopped to compute score on original non-padded image 
    gt_cropped = gt_mask[:,:,12:-12, 12:-12, :]
    pred_cropped = pred_mask[:,:,12:-12, 12:-12, :]

    numerator = torch.sum(gt_cropped*pred_cropped, dim=(2,3,4)) 
    combined_masks = torch.logical_or(gt_cropped, pred_cropped)
    denominator = torch.sum(combined_masks, dim= (2,3,4)) + 1e-6

    #combined_masks = gt_mask + pred_mask
    #combined_masks[combined_masks != 0] = 1:
    #denominator = torch.sum(combined_masks, (dim = 2,3,4)) + 1e-6
    #denominator = torch.sum(gt_mask, dim=(2,3,4)) + torch.sum(pred_mask, dim=(2,3,4)) + 1e-6

    iou = torch.mean((numerator / denominator))

    return iou 

###### Training scripts ######
def validate(val_dataloader, model, use_cuda = True, save_path = 'model_1', save_images = False):

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
            
            loss_vals_eval.append(loss.item())
            iou_vals_eval.append(iou.item())

        if save_images:
            # Save image, labels and outputs into h5py files
            img_name = patient_name[0].split(".")[0] + '_rectum_PRED.nrrd'
            img_path = os.path.join(save_path, img_name)
            sitk.WriteImage(sitk.GetImageFromArray(image.cpu()), img_path)
    
    mean_iou = torch.mean(torch.FloatTensor(iou_vals_eval))
    mean_loss = torch.mean(torch.FloatTensor(loss_vals_eval))



    return mean_loss, mean_iou
    
def train(model, train_dataloader, val_dataloader, num_epochs = 10, use_cuda = False, save_folder = 'model_1'):
    
    """
    A function that performs the training and validation loop 

    :params:
    :prostate_dataloader: torch dataloader that can load images and masks in
    :num_epochs: (int) Number of epochs to train for 
    """

    writer = SummaryWriter() 
    current_dir = os.getcwd()
    train_folder = os.path.join(current_dir, save_folder)
    os.makedirs(train_folder, exist_ok = True) 
    print(f'train folder path : {train_folder}')
    writer = SummaryWriter(os.path.join(train_folder, 'runs')) 

    if use_cuda:
        model.cuda()

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-05)
    step = 0 
    freq_print = 4
    freq_eval = 4
    all_loss_train = np.zeros((num_epochs,1))
    all_iou_train = np.zeros((num_epochs, 1))
    all_loss_val = [] 
    all_iou_val = []
    best_loss = np.inf 
    best_iou = 0 
    
    csv_train_path = os.path.join(train_folder, 'train_loss.csv')
    csv_val_path = os.path.join(train_folder, 'val_loss.csv')

    with open(csv_train_path, 'w') as fp:
        fp.write('''epoch_num, loss, iou''')
        fp.write('\n')

    with open(csv_val_path, 'w') as fp:
        fp.write('''epoch_num, loss, iou''')
        fp.write('\n')

    for epoch_no in range(num_epochs):
        
        iou_vals = []
        loss_vals = [] 

        # Initialise training loop
        for idx, (images, labels, _) in enumerate(train_dataloader):

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
            iou_vals.append(iou.item())
            loss_vals.append(loss.item())

            # Print loss every nth minibatch and dice score 
            #if idx % freq_print == 0: 
            #    print(f'Epoch {epoch_no} minibatch {idx} : loss : {loss.item():05f}, IOU score : {iou.item():05f}')
            
        # Obtain mean dice loss and IOU over this epoch, save to tensorboard 
        iou_epoch = torch.mean(torch.tensor(iou_vals))
        loss_epoch = torch.mean(torch.tensor(loss_vals))
        print(f'\n Epoch : {epoch_no} Average loss : {loss_epoch:5f} average IOU {iou_epoch:5f}')

        with open(csv_train_path, 'a') as fp: 
            loss_points = np.stack([epoch_no, loss_epoch, iou_epoch]).reshape(1,-1)
            np.savetxt(fp, loss_points, '%s', delimiter =",")

        # Save for all_loss_train
        all_loss_train[epoch_no] = loss_epoch
        all_iou_train[epoch_no] = iou_epoch 
        
        #Tensorboard saving 
        writer.add_scalar('Loss/train', loss_epoch, epoch_no)
        writer.add_scalar('IOU/train', iou_epoch, epoch_no)
    
        # Save newest model 
        train_model_path = os.path.join(train_folder, 'train_model.pth')
        torch.save(model.state_dict(), train_model_path)

        # Validate every nth epoch and save every nth mini batch 
        if epoch_no % freq_eval == 0: 

            # Set to evaluation mode 
            if epoch_no % 100 == 0: 
                save_img = True
            else:
                save_img = False 
                
            mean_loss, mean_iou = validate(val_dataloader, model, use_cuda = use_cuda, save_path = train_folder, save_images = save_img)
            print(f'Validation loss for epoch {epoch_no} Average loss : {mean_loss:5f} average IOU {mean_iou:5f}')
            all_loss_val.append(mean_loss)
            all_iou_val.append(mean_iou)
            
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, mean_loss, mean_iou]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

            #Tensorboard saving
            writer.add_scalar('Loss/val', mean_loss, epoch_no)
            writer.add_scalar('IOU/val', mean_iou, epoch_no)

            if mean_loss < best_loss: 
                
                # Save best model as best validation model 
                val_model_path = os.path.join(train_folder, 'best_val_model.pth')
                torch.save(model.state_dict(), val_model_path)
        
                # Use as new best loss
                best_loss = mean_loss 
        
        elif epoch_no == 0: 
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, 1, 0]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")
        else:
            with open(csv_val_path, 'a') as fp: 
                loss_points = np.stack([epoch_no, all_loss_val[-1], all_iou_val[-1]]).reshape(1,-1)
                np.savetxt(fp, loss_points, '%s', delimiter =",")

        #print('Chicken') 
    return all_loss_train, all_loss_val, all_iou_train, all_iou_val 






            
















