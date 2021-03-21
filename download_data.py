"""
This script will download VIIRS data through NASA VIIRS SIPS and goes data
through AWS S3. The script will select VIIRS data that is over the western 
portion of the United States and GOES 17 data from a similar timestamp.

Created on Sun Jul 12 11:36:33 2020

@author: mullenj
"""

import os
import sys
import subprocess
import requests
import csv
from datetime import datetime, timedelta

data_dir = r'D:\raw_data'

def main():
    # Get VIIRS SIPS Query csv file
    results = requests.get("https://sips.ssec.wisc.edu/api/v1/products/search.csv" +
        "?products=VNP02IMG|VNP03IMG&satellite=snpp" +
        "&start=2019-01-01T00:00:00Z&end=2020-07-01T00:00:00Z" +
        "&loc=39,-118,1000")
    if results.status_code != 200:
        print("Failure in VIIRS API Request")
        sys.exit()
    content = results.content.decode('utf-8')
    cr = csv.reader(content.splitlines(), delimiter=',')
    file_list = list(cr)
    file_list = file_list[7:]
    print("Found {} VIIRS-I files to download".format(len(file_list)))
    failures = open('failures.txt', 'w')
    for file in file_list:
        # Download file if necessary and save in external storage
        fname = file[0]
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            print("Already Downloaded {}".format(fname))
            continue
        url = file[2]
        r = requests.get(url)
        open(path, 'wb').write(r.content)
        print("Downloaded {}".format(fname))
        if file[1] == 'VNP03IMG':
            # Download comperable GOES file for every set of viirs files
            v_time = datetime.strptime(file[3], '%Y-%m-%dT%H:%M:%S.%fZ')
            goes_path = 'publicAWS:noaa-goes17/ABI-L1b-RadC/{}/{:03d}/{:02d}'.format(
                v_time.year, v_time.timetuple().tm_yday, v_time.hour)
            goes_files = subprocess.check_output('./rclone.exe ls {}'.format(goes_path)).decode().split('\n')
            match = False
            for gfile in goes_files[:-1]:
                gf = gfile.split()[1]
                splits = gf.split('_')
                band = int(splits[1][-1])
                if band != 7:
                    continue
                g_time = datetime.strptime(splits[3], 's%Y%j%H%M%S%f')
                tm_delta = v_time - g_time
                if abs(tm_delta.total_seconds()) < 4*60:
                    subprocess.run('./rclone.exe copy {}/{} {}'.format(goes_path, gf, data_dir))
                    print("Downloaded {}".format(gf))
                    match = True
                    break
            if not match:
                print("No Match found for {}".format(fname))
                failures.write("No Match found for {}".format(fname))
    failures.close

if __name__ == "__main__":
    main()