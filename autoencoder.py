# -*- coding: utf-8 -*-
"""
This script will run through the directory of training images, load
the image pairs, and then batch them before feading them into a pytorch based
autoencoder using MSE reconstruction loss for the superresolution.

Created on Sun Jul 26 11:17:09 2020

@author: mullenj
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

im_dir = '/media/me-user/Backup Plus/training_data_2'
n_epochs = 0
n_latents = 512
batch_size = 32
learning_rate = 1e-4
log_interval = 100

random_seed = 1
torch.manual_seed(random_seed)

class npDataset(Dataset):
    """
    npDataset will take in the list of numpy files and create a torch dataset
    out of it.
    """
    def __init__(self, data_list, batch_size):
        self.array = data_list
        self.batch_size = batch_size
        
    def __len__(self): return int((len(self.array)/self.batch_size))
    
    def __getitem__(self, i):
        """
        getitem will first select the batch of files before loading the files
        and splitting them into the goes and viirs components, the input and target
        of the network
        """
        files = self.array[i*self.batch_size:i*self.batch_size + self.batch_size]
        x = []
        y = []
        for file in files:
            file_path = os.path.join(im_dir, file)
            sample = np.load(file_path)
            x.append(sample[:,:,1])
            y.append(sample[:,:,0])
        x, y = np.array(x)/255., np.array(y)/255.
        x, y = np.expand_dims(x, 1), np.expand_dims(y, 1)
        return torch.Tensor(x), torch.Tensor(y)
    
class Encoder(nn.Module):
    """
    Autoencoder trained to separate class distributions in the latent space
    This is the encoder portion
    """
    def __init__(self, n_latents, in_features):
        super(Encoder, self).__init__()
        self.in_features = in_features
        
        self.conv1 = nn.Conv2d(in_features, 64, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3,3), padding=1)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3,3), padding=1)
        
        # self.fc1 = nn.Linear(self.in_features, 4096)
        # self.fc2 = nn.Linear(4096, 1024)
        # self.fc3 = nn.Linear(1024, n_latents) 
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2,2))
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=(2,2))
        
        # x = F.pad(x, (1, 1, 1, 1))
        x = F.relu(self.conv5(x))
        return x

class Decoder(nn.Module):
    """
    Autoencoder trained to separate class distributions in the latent space
    This is the decoder portion
    """
    def __init__(self, n_latents, in_features):
        super(Decoder, self).__init__()
        self.in_features = in_features
        
        self.tconv1 = nn.ConvTranspose2d(in_features, 128, kernel_size=(3,3), stride = 2, padding = 1)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=(3,3), padding = 1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3,3), padding = 1)        
        
        self.tconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), padding = 1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,3), padding = 1)     

        
        self.conv5 = nn.Conv2d(64, 1, kernel_size=(3, 3), padding = 1)
        
        # self.fc1 = nn.Linear(n_latents, 1024)
        # self.fc2 = nn.Linear(1024, 4096)
        # self.fc3 = nn.Linear(4096, self.in_features)     
        
    def forward(self, x):
        x = F.relu(self.conv1(self.tconv1(x, output_size = (10, 128, 64, 64))))
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.conv3(self.tconv2(x, output_size = (10, 64, 128, 128))))
        x = F.relu(self.conv4(x))
        
        x = F.relu(self.conv5(x))
        return x

def train(train_loader, test_loader, encoder, decoder, opt):
    
    for epoch in range(n_epochs+1):
        encoder.train()
        decoder.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            x, y = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0)
            opt.zero_grad()
            encoder_output = encoder(x)
            decoder_output = decoder(encoder_output)
            loss = F.mse_loss(decoder_output, y)
            loss.backward()
            opt.step()
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader.dataset)} ({loss.item()})]')
        encoder.eval()
        decoder.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                x, y = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0)
                encoder_output = encoder(x)
                decoder_output = decoder(encoder_output)
                test_loss += F.mse_loss(decoder_output, y)
        test_loss /= len(test_loader.dataset)        
        print(f'Test Reconstruction Loss: ({loss.item()})]')
            
            

def main():
    # Get List of downloaded files and set up data loader
    file_list = os.listdir(im_dir)
    print(f'{len(file_list)} data samples found')
    train_files, test_files = train_test_split(file_list, test_size = 0.2, random_state = 42)
    train_loader = DataLoader(npDataset(train_files, batch_size))
    test_loader = DataLoader(npDataset(test_files, batch_size))
    
    # Set up the encoder, decoder. and optimizer
    encoder = Encoder(n_latents, 1)
    decoder = Decoder(n_latents, 256)
    encoder.cuda()
    decoder.cuda()
    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = learning_rate)
    
    # Train and save the model components
    train(train_loader, test_loader, encoder, decoder, opt)
    torch.save(encoder.state_dict(), 'SuperRes_Encoder.pth')
    torch.save(decoder.state_dict(), 'SuperRes_Decoder.pth')
    torch.save(opt.state_dict(), 'SuperRes_Opt.pth')
    

if __name__ == "__main__":
    main()
