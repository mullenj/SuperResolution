"""
This script will run through the directory of preprocessed png images, load
the image pairs, and then run a sliding window over them to generate the 
chipped training data,

Created on Sun Jul 26 11:17:09 2020

@author: mullenj
"""
import os
from PIL import Image
import numpy as np
from datetime import datetime, timedelta

pp_dir = '/media/me-user/Backup Plus/processed_data'
out_dir = '/media/me-user/Backup Plus/training_data_2'
v_reader = 'viirs_l1b'
g_reader = 'abi_l1b'

def main():
    # Get List of downloaded files
    file_list = os.listdir(pp_dir)
    print('Roughly {} data samples found'.format(len(file_list)/2))
    for file in file_list:
        # Only run loop when you have a viirs file
        if not file.startswith('VNP03'):
            continue
        # Get the band file
        if os.path.exists(os.path.join(pp_dir, file[:-3] + '.png')):
            continue
        viirs_path = os.path.join(pp_dir, file)
        # Get comparable GOES file
        v_time = datetime.strptime(file[10:22], '%Y%j.%H%M')
        g_subs = 'C07_G17_s{}'.format(file[10:17])
        g_files = [i for i in file_list if g_subs in i]
        goes_file = None
        for g_file in g_files:
            g_time = datetime.strptime(g_file[27:38], '%Y%j%H%M')
            tm_delta = v_time - g_time
            if abs(tm_delta.total_seconds()) < 4*60:
                goes_file = g_file
        if not goes_file:
            print('No Goes File for {}'.format(file))
            continue      
        goes_path = os.path.join(pp_dir, goes_file)
        # Load PNGs and Output Windowed Numpy Arrays
        print(file)
        vf = Image.open(viirs_path)
        gf = Image.open(goes_path)
        vf = np.array(vf)[:,:,0]
        gf = np.array(gf)[:,:,0]
        if vf.shape != gf.shape:
            print("Failure {}".format(file))
            continue
        stack = np.stack((vf, gf), axis = 2)
        for x, y, window in sliding_window(stack, 128, (128,128)):
            if window.shape != (128,128,2):
                continue
            g_win = window[:,:,1]
            if (np.count_nonzero(g_win)/g_win.size < 0.985):
                continue
            else:
                np.save(os.path.join(out_dir, 'comb.' + v_time.strftime('%Y%j.%H%M')
                                     + '.' + str(x) + '.' + str(y) + '.npy'), window)
            

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0], :])

if __name__ == "__main__":
    main()