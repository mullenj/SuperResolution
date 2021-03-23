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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time, math
import matplotlib.pyplot as plt

im_dir = '/media/me-user/Backup Plus/training_data_2'
model_path = 'checkpoint/model_epoch_3.pth'
batch_size = 1

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



def test(test_loader, model):
    avg_psnr_predicted = 0.0
    avg_psnr_control = 0.0
    avg_elapsed_time = 0.0
    count = 0.0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            count += 1
            print("Processing ", batch_idx)
            psnr_bicubic = PSNR(x, y)
            avg_psnr_control += psnr_bicubic
            x, y = torch.squeeze(x, dim=0), torch.squeeze(y, dim=0)
            x = x.cuda()
            start_time = time.time()
            output = model(x)
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time
    
            output = output.cpu()
            x = x.cpu()
            
            psnr_predicted = PSNR(output, y)
            avg_psnr_predicted += psnr_predicted
            
            x = np.squeeze(x)
            output = np.squeeze(output)
            y = np.squeeze(y)
            
            if count%10000 ==0:
                fig, axs = plt.subplots(3, 1, constrained_layout=True)
                axs[0].imshow(x)
                axs[0].set_title('GOES Imagery')
                axs[1].imshow(output)
                axs[1].set_title('Network Prediction')
                axs[2].imshow(y)
                axs[2].set_title('VIIRS-I Imagery')
                fig.savefig(f'ims/{batch_idx}_together.png')
                plt.close()

    print("PSNR_predicted=", avg_psnr_predicted/count)
    print("PSNR_control=", avg_psnr_control/count)
    print("It takes average {}s for processing".format(avg_elapsed_time/count))
        
def PSNR(pred, gt, shave_border=0):
    imdff = pred - gt
    imdff = imdff.flatten()
    rmse = math.sqrt(np.mean(np.array(imdff ** 2)))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
            
            

def main():
    # Get List of downloaded files and set up data loader
    file_list = os.listdir(im_dir)
    print(f'{len(file_list)} data samples found')
    train_files, test_files = train_test_split(file_list, test_size = 0.2, random_state = 42)
    test_loader = DataLoader(npDataset(test_files, batch_size))
    
    # Set up the encoder, decoder. and optimizer
    model = torch.load(model_path, map_location=lambda storage, loc: storage)["model"]
    model.cuda()
    
    # test the model components
    test(test_loader, model)
    

if __name__ == "__main__":
    main()
