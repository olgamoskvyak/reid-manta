"""Copy files to a separate folder as per csv

This script allows the user to select and copy the files from one folder
to another as specified in the first column of csv file.
"""

import os, argparse
import numpy as np
from imageio import imread, imsave

argparser = argparse.ArgumentParser(
    description='Copy files from one folder to another as per csv file.')

argparser.add_argument('-s', '--src', required=True, help='Source folder with images')
argparser.add_argument('-t','--trg', required=True, help='Target folder to copy images to. The folder will be created if  not exists')
argparser.add_argument('-f', '--file', required=True, help='Csv file with a list of files in the first column to copy (file with header)')

def _main_(args):
    labels_file = args.file #'data/whales_not_trained_2test.csv'
    src_folder = args.src #os.path.join('data', 'whales_train')
    trg_folder = args.trg #os.path.join('data', 'whales_test_images')

    labels = np.genfromtxt(labels_file, dtype=str, skip_header=1, delimiter=',', usecols = (0))
    print('Read {} labels from {}'.format(labels.shape, labels_file))



    if not os.path.exists(trg_folder): os.makedirs(trg_folder)

    for file in labels:
        img = imread(os.path.join(src_folder, file))
        imsave(os.path.join(trg_folder, file), img)
    
if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
