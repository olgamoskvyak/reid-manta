'''
===============================================================================
Tool to preprocess images before training/using it with CNN for re-identification.

This tool allows a user to process one image or images in a directory (no nested directories).

USAGE:
    with required parameters only (optional are defaults):
    python prepare_db.py -i <image_path/image_dir>
    
    with all optional parametres:
    python prepare_db.py -i <image_path/image_dir> -d <yes/no> -c <path_config>
                         -o <path_output_folder>   -l <path_to_csv>
                         
FUNCTIONALITY:
    1) Interactive drawing tool to highligh the region of interest in the image.
    Use option -d no to disable.
    2) After drawing the script crops each image by highlighed region and resizes it to 
    the size specified in config file.
    3) If csv file is specified then files and corresponding labels are read from 
    the file. Otherwise, it is assumed that the name of the folder is the label.
    4) If files are supplied in one folder with labels in csv then 
    the script rearranges the files in folders based on the label.
    
    
PARAMETERS:
    -i     path to image or a folder with images to process (no nested). Required argument.
    -d     yes/no, activate or not mask drawing tool. Default: yes. 
           Use it to localize a pattern of interest.
    -c     path to configuration file with default parameters. Default: configs/manta.json  
           Preconfigured files: configs/manta.json for manta rays, configs/whale.json for whales.
    -o     path to save processed files. Default is in config.prod.output
    -l     path to a csv files with a list of files and corresponding labels. Default is in config.prod.lfile.
    
README:
    If drawing tool is activated, draw a line around a pattern of interest.
    To change a mask simply start drawing a new line.
    Key 's' - to save the drawing
    Key 'q' - to process next image without saving current one
    Key esc - to exit the drawing tool

    Each image is cropped by mask, resized to the size specified in config file and saved to
    an output directory.
    

===============================================================================
'''

import numpy as np
import os, argparse, json

from utils.preprocessing import crop_im_by_mask, resize_imgs, convert_to_fmt
from utils.drawer import MaskDrawer
from utils.utils import str2bool

argparser = argparse.ArgumentParser(description='Prepare database from the data')

argparser.add_argument('-i', '--impath', required = True, help='Path to image/folder with images to process (no nested)')
argparser.add_argument('-d','--draw', type=str2bool, help='Activate mask drawing tool for each image: True/False. Default: True', default=True)
argparser.add_argument('-c','--conf', help='Path to configuration file with default parameters', default='configs/manta.json')
argparser.add_argument('-o', '--output', help='Path to output files. Default is in config.prod.output')
argparser.add_argument('-l','--lfile', help='List of files to process in csv file (w/header:file,label). Filenames only, not the full path. Default is in config.prod.lfile.')
argparser.add_argument('-x','--idx', type=int, default=0, help='Index (number) of file to resume the process. Zero based. Useful for large folders')

    
if __name__ == "__main__":
    print(__doc__)
    #Get arguments
    args = argparser.parse_args()
    impath = args.impath.strip(os.sep)
    if not os.path.exists(impath):
        raise ValueError('Image file/folder does not exist. Check input.')
    
    config_path = args.conf
    if not os.path.exists(config_path):
        raise Exception('Config file does not exist. Check input.')
    
    #Get flag to activate/deactivate drawing tool
    draw = args.draw     
        
    #Read config file
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
        
    if args.output is None:
        output_dir = config['prod']['output'].strip(os.sep)
    else:
        output_dir = args.output.strip(os.sep)
        
    if args.lfile is None:
        lfile = config['prod']['lfile']
    else:
        lfile = args.lfile
    
    #Reading list of image filenames
    if os.path.isdir(impath):
        if os.path.exists(lfile):
            print('Reading only files specified in a file {}'.format(lfile))
            #Read files and labels from csv
            fileslabels = np.genfromtxt(lfile, dtype=str, skip_header=1, delimiter=',')
            #Append parent directory to each filename to make a path
            files = np.array([os.path.join(impath,file) for file in fileslabels[:,0]])
            #Remove extension from filename (all files will be converted to .png in later steps)
            fileslabels = np.array([[os.path.splitext(row[0])[0], row[1]] for row in fileslabels])
            
            #Create dictionary filename_noext: label
            file2lab = dict(fileslabels)
        else:
            files = [os.path.join(impath, f) for f in os.listdir(impath) if os.path.isfile(os.path.join(impath, f))]
            print('Found {} files in source {}. Subdirectories are ignored'.format(len(files), impath))
    else:
        if not os.path.exists(impath):
            raise ValueError('Image file does not exist. Check input')
        print('Reading image {}'.format(impath))
        files = [impath]
    
    #Add subdirectory to the output folder
    (_, folder) = os.path.split(impath)
    output_dir = os.path.join(output_dir, folder)     
    print('Output files will be stored in {}'.format(output_dir))
    if not os.path.exists(output_dir): os.makedirs(output_dir)
        
    size = (config['model']['input_width'], config['model']['input_height'])
        
    #Count processed images
    proc_count = 0
    
    #Sort order of the files
    files.sort()
   
    #Draw mask on each file and save
    for i in range(args.idx, len(files)):
        file = files[i]
        print('Processing file {} {}'.format(i, file))
        if draw:
            #Get filename for mask
            (_, imname) = os.path.split(file)
            maskpath = os.path.join(output_dir, imname)
            md = MaskDrawer(file, maskpath)
            respond = md.run()
            if respond == 'exit':
                print('Exiting the programm')
                quit()
            if respond == 'next':
                print('Skipped file {} {}'.format(i, file))
                #do not include this image in the dataset
                continue
            if respond == 'save' and md.done:
                proc_count += 1
                square = (config['model']['input_width']==config['model']['input_height'])
                croppedpath = crop_im_by_mask(file, maskpath, output_dir, padding = 0, square=square)
            if respond == 'save' and not md.done:
                proc_count += 1
                croppedpath = file
        else:
            proc_count += 1
            croppedpath = file
        #Resize to the size and convert to png format
        resizedpath = resize_imgs(croppedpath, output_dir, size)
        resizedpath = convert_to_fmt(resizedpath, imformat = 'png')
        print('Processed {} images'.format(proc_count))
    
    print('Total processed {} images'.format(proc_count))
    
    #If files are supplied in one folder with labels in csv, rearrange processed it to one folder per class
    if os.path.exists(lfile):
        print('Rearranging files in directories as per labels...')
        out_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        for file in out_files:
            #Get filename only with no extention
            filename = os.path.basename(file)
            filename_noext = os.path.splitext(filename)[0]
            #Get label and create folder with the same name
            folder = os.path.join(output_dir, file2lab[filename_noext])
            if not os.path.exists(folder): os.makedirs(folder)
            #Move image to it's class folder
            os.rename(file, os.path.join(folder, filename))            
