"""
This script will preprocess the downloaded VIIRS data and GOES data. The 
script will create a satpy scene of the VIIRS and GOES images, resample the
GOES image based off of the VIIRS area, then save as a PNG. Once all PNGs are 
saved, they will be chipped and saved as (np arrays or new pngs).

Created on Fri Jul 17 12:07:37 2020

@author: mullenj
"""
import os
from satpy import Scene
from PIL import Image
import numpy as np
from datetime import datetime, timedelta

data_dir = r'D:\raw_data'
pp_dir = r'D:\processed_data'
v_reader = 'viirs_l1b'
g_reader = 'abi_l1b'

def main():
    failures = open(r'D:\failures.txt', 'w')
    # Get List of downloaded files
    file_list = os.listdir(data_dir)
    print('Roughly {} data samples found'.format(len(file_list)/3))
    for file in file_list:
        # Only run loop when you have a geo file
        if not file.startswith('VNP03'):
            continue
        # Get the band file
        if os.path.exists(os.path.join(pp_dir, file[:-3] + '.png')):
            continue
        geo_path = os.path.join(data_dir, file)
        subs = file[:4] + '2' + file[5:24]
        band_path = [i for i in file_list if subs in i]
        if len(band_path) != 1:
            print('failure {} {}'.format(file, band_path))
            break
        band_path = os.path.join(data_dir, band_path[0])
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
            failures.write("No Match found for {}\n".format(file))
            continue      
        # Load SatPy Scenes
        viirs_files = [band_path, geo_path]
        goes_files = [os.path.join(data_dir, goes_file)]
        viirs_scene = Scene(reader = v_reader, filenames = viirs_files)
        goes_scene = Scene(reader = g_reader, filenames = goes_files)
        viirs_scene.load(['I04'])
        goes_scene.load(['C07'])
        
        # Resample and Save PNGs
        print(file)
        rs = viirs_scene.resample(viirs_scene['I04'].attrs['area'], resampler = 'nearest')
        rs.save_dataset('I04', os.path.join(pp_dir, file[:-3] + '.png'))
        rs_g = goes_scene.resample(viirs_scene['I04'].attrs['area'], resampler = 'nearest')
        rs_g.save_dataset('C07', os.path.join(pp_dir, goes_file[:-3] + '.png'))     
    failures.close


if __name__ == "__main__":
    main()