import torch 
import numpy as np 
import os 
import torch.nn as nn

### 3D U-net implementation for segmenting organs at risk 

class UNet_3D(nn.Module):

    def __init__(self, input_channel = 1, output_channel = 1, num_features = 32, num_layers = 4):
        super(UNet_3D, self).__init__()

        self.num_features = num_features 

        # Identify each layers in the UNet
        self.encoder_1 = UNet_3D._build_conv_block(input_channel, num_features)
        self.maxpool_1 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder_2 = UNet_3D._build_conv_block(num_features, num_features*2)
        self.maxpool_2 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder_3 = UNet_3D._build_conv_block(num_features*2, num_features*4)
        self.maxpool_3 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder_4 = UNet_3D._build_conv_block(num_features*4, num_features*8)
        self.maxpool_4 = nn.MaxPool3d(kernel_size = 2, stride = 2)

        self.bottle_neck = UNet_3D._build_conv_block(num_features *8, num_features * 16)

        self.upconv_4 = nn.ConvTranspose3d(num_features * 16, num_features * 8, kernel_size = 2, stride = 2)
        self.decoder_4 = UNet_3D._build_conv_block((num_features*8)*2, num_features *8)
        self.upconv_3 = nn.ConvTranspose3d(num_features * 8, num_features * 4, kernel_size = 2, stride = 2)
        self.decoder_3 = UNet_3D._build_conv_block((num_features*4)*2, num_features *4)
        self.upconv_2 = nn.ConvTranspose3d(num_features * 4, num_features * 2, kernel_size = 2, stride = 2)
        self.decoder_2 = UNet_3D._build_conv_block((num_features*2)*2, num_features *2)
        self.upconv_1 = nn.ConvTranspose3d(num_features*2 , num_features, kernel_size = 2, stride = 2)
        self.decoder_1 = UNet_3D._build_conv_block(num_features*2, num_features)

        self.final_conv = torch.nn.Conv3d(num_features, output_channel, kernel_size = 1)

    def forward(self, x):
        enc1 = self.encoder_1(x)
        enc2 = self.encoder_2(self.maxpool_1(enc1))
        enc3 = self.encoder_3(self.maxpool_2(enc2))
        enc4 = self.encoder_4(self.maxpool_3(enc3))

        bottleneck = self.bottle_neck(self.maxpool_4(enc4))

        dec4 = self.upconv_4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder_4(dec4)

        dec3 = self.upconv_3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder_3(dec3)

        dec2 = self.upconv_2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder_2(dec2)

        dec1 = self.upconv_1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder_1(dec1)

        return torch.sigmoid(self.final_conv(dec1))

    @staticmethod
    def _build_conv_block(input_channel, num_features):
        
        conv_block = nn.Sequential(
            nn.Conv3d(in_channels = input_channel, out_channels = num_features, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm3d(num_features = num_features),
            nn.ReLU(inplace= True),
            nn.Conv3d(in_channels = num_features, out_channels = num_features, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm3d(num_features = num_features),
            nn.ReLU(inplace=True))

        return conv_block 

    def _centre_crop(encoder_layer, decoder_layer):
        """
        A function that centre crops encoder layer to match that of the encoder layer 
        """
        pass 