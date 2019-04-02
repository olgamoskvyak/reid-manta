import os
import cv2
import numpy as np
from imageio import imread, imsave
from skimage import img_as_float
from sklearn.model_selection import KFold
from glob import glob
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


def convert_to_fmt(src, imformat = 'png', logstep = 1000):
    """Convert image/images to specified format. Changes are made in place.
    Note: OpenCV is used instead of PIL as it handles better different formats including .tiff
    Input:
    src: string, path to image or a directory with images
    informat: string, target format for images
    logstep: integer, number of steps to display progress
    """
    print('Converting source', src)
    if os.path.isdir(src):
        imfiles = [os.path.join(src, f) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
        print('Found {} files in source {}. Subdirectories are ignored'.format(len(imfiles), src))
    else:
        imfiles = [src]
    
    #Remove leading dot if it is in the format
    imformat.replace('.', '')
    
    #imfiles - list of full path to images
    new_files = []
    for i, file in enumerate(imfiles):
        try:
            file_noext, ext = os.path.splitext(file)
            #ext has leading dot
            #If format is the same, do not do anything
            if ext[1:] != imformat:
                img = cv2.imread(file, 1)
                new_filename = file_noext + '.' + imformat
                cv2.imwrite(new_filename, img)
                new_files.append(new_filename)
                #Remove original file
                os.remove(file)
            else:
                new_files.append(file)
        except Exception as e:
            print(file, e)
            
        if i%logstep==0 and i>0:
            print('Converted {} images'.format(i))
    print('Converted {} images.'.format(len(imfiles)))
    if os.path.isdir(src):
        return new_files
    else:
        return new_files[0]
    
        
def crop_im_by_mask(impath, maskpath, cropped_dir, padding = 0, square=True):
    """Crop an image by masks. Cropped image is a square size if possible.
    Input:
    impath: string, path to image
    masks_dir: string, path to mask
    cropped_dir: string, directory to save cropped image
    padding: float[0,1], how much padding to add around mask, proportional to the width and height.
    
    """
    mask = imread(maskpath)
    image = imread(impath)
    (x,y,w,h) = get_bound_box(maskpath)

    #Find 'new' coordinates.
    if square:
        #Find longer side
        side = max(w, h)
        diff = abs(w - h)
        w_n = side
        h_n = side

        if (w >= h):
            y_n = max(0, y - round(diff/2))
            x_n = x

        if (w < h):
            x_n = max(0, x - round(diff/2))
            y_n = y
    else:
        (x_n, y_n, w_n, h_n) = (x,y,w,h)

    #Add padding around bounding box:
    (x_p, y_p, w_p, h_p) = (x_n-round(h_n*padding),
                            y_n-round(w_n*padding),
                            round(w_n+w_n*2*padding), round(h_n+h_n*2*padding))    
    x_min = max(0, x_p)
    y_min = max(0, y_p)
    x_max = min(x_p+w_p, mask.shape[1])
    y_max = min(y_p+h_p, mask.shape[0])

    cropped = image[y_min:y_max, x_min:x_max]

    #copy and crop manta image
    imdir, imfile = os.path.split(impath)
    croppedpath = os.path.join(cropped_dir, imfile)
    imsave(croppedpath, cropped)
    print('Cropped by mask and saved as {}'.format(croppedpath))
    
    return croppedpath
    
        
def resize_imgs(src, dest, size, del_src=False):
    """Crop images from source folder to a specific size and save to a dest folder.
    ---
    Input:
    src: string, path to image or a directory with images
    dest: string, target folder for resized images
    size: 2D tuple, target size of images
    del_src: bool, delete or not source image. Default: False
    """
    if os.path.isdir(src):
        imfiles = [os.path.join(src, f) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
        print('Found {} files in source {}. Subdirectories are ignored'.format(len(imfiles), src))
    else:
        imfiles = [src]
        
    if not os.path.exists(dest): os.mkdir(dest)
    
    resized_files = []
    for i, file in enumerate(imfiles):
        try:
            #Read image and resize
            img = imread(file)
            res = cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)

            resized_file = os.path.join(dest, os.path.basename(file))
            imsave(resized_file, res)
            resized_files.append(resized_file)
            if del_src:
                os.remove(file)
        except Exception as e:
            print(file, e)
    #Return a list of new resized files or 1 file 
    if os.path.isdir(src):
        return resized_files
    else:
        return resized_files[0]
        

def get_bound_box(filename):
    """Find bounding box for masked image.
    --------------------------------------
    Input:
    filename - string, path to masked image
    
    Return:
    (x,y,w,h) - tuple, bounding box
    """
    img = imread(filename)

    #Threshold mask
    ret,thresholded = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    contours = cv2.findContours(thresholded, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
    # Choose largest contour
    best = 0
    maxsize = 0
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > maxsize :
            maxsize = cv2.contourArea(cnt)
            best = count

        count = count + 1
    x,y,w,h = cv2.boundingRect(contours[best])
    return (x,y,w,h)
        
        
def crop_im_by_box(filename, bbox, verbose=False):
    """Crop and save image by bounding box.
    ---------------------------------
    Input:
    filename - string, path to image
    bbox - list or tuple, (x,y,w,h), (x, y) - top left corner
    verbose - True or False, flag to display messages
    Return:
    message
    """
    try:
        image = imread(filename)
        #print('image shape: {}'.format(image.shape))
        (x,y,w,h) = bbox
        #print('bbox: {}'.format(bbox))
        cropped = image[y:y+h, x:x+w]
        imsave(filename, cropped)
        if verbose: print(filename + ' cropped')
    except Exception as e:
        print(e)
        print('Not found {}'.format(filename))
    
def read_dataset(img_dir, data_type='uint8', return_filenames=False, original_labels=False):
    """Read a dataset from a directory where each class is in a subdirectory:
    img_dir: string, path to image directory
    data_type: string, data type of images, expects float32 or uint8
    return_filenames: boolean, if True, an array with filenames is returned
    original_labels: boolean, if True, an original labels returned, if False, integer labels are returned
    
    Return:
    X - ndarray of images
    y - 1D integer labels
    class_dict - dictionary of labels and subdir names
    filenames - optional, array of filenames (relative path)
    """
    print('Reading files from {}'.format(img_dir))
    class_dict = {}
    y = []
    label_count = 0
    filenames = glob(img_dir + '/*/*')
    n = len(filenames)
    print('Found %d files' % n)

    for i, file in enumerate(filenames):
        if data_type == 'float32':
            img = img_as_float(imread(file))
        elif data_type == 'uint8':
            img = imread(file)
        else:
            raise ValueError('Incorrect data type')

        if i == 0:
            imsize = img.shape
            X = np.zeros((n, imsize[0], imsize[1], imsize[2]), dtype = data_type)
            #y = np.zeros(n, dtype = 'uint32')

        #get class name from subfolder
        (head, tail) = os.path.split(file)
        (subhead, subtail) = os.path.split(head)
        subfolder = subtail
        if not subfolder in class_dict:
            class_dict[subfolder] = label_count
            label_count += 1

        X[i] = img[:, :, :3]
        if original_labels:
            y.append(subfolder)
        else:
            y.append(class_dict[subfolder])

        if i % 1000 == 0 and i > 0:
            print("%d images read" % (i))

    y = np.array(y)
    print('Read %d files from %d classes' % (n, label_count))
    print('X shape: ' + str(X.shape))
    print('Labels shape: ' + str(y.shape))

    #Add reversed keys to dictionary
    class_dict.update({v:k for k,v in class_dict.items()})

    if return_filenames:
        return X, y, class_dict, filenames
    else:
        return X, y, class_dict

def split_classes(dataset, labels, test_size=0.15, seed=None, return_mask=False, split_num=-1):
    '''Split dataset and labels into train and validation without class overlap
    
    Input:
    -----------
    dataset: 4D array of images
    labels: ndarray of labels
    test_size: float from 0 to 1, portion of test set
    seed: integer, seed to initialise random generator
    Returns:
    ---------
    dataset_t, labels_t, dataset_v, labels_v
    '''
    dataset_len = dataset.shape[0]
    print('Splitting dataset of size: {}'.format(dataset_len))
    unique_lbls = list(np.unique(labels))
    
    if seed is None:
        seed = 0
  
    if split_num == -1:
        lbls_train, lbls_valid = train_test_split(unique_lbls, test_size=test_size, random_state=seed)
    
    else:
        print('K-Fold split. Loading data for the split: {}'.format(split_num))
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_gen = kf.split(unique_lbls)
        train_index, test_index = next(split_gen)
        for step in range(split_num):
            train_index, test_index = next(split_gen)

        print('test index:', test_index)
        lbls_train = np.array(unique_lbls)[train_index]
        
    mask_train = np.full((dataset_len,), False, dtype=bool)

    for i, image in enumerate(dataset):
        if labels[i] in lbls_train:
            mask_train[i] = True
                
    dataset_t = dataset[mask_train]
    dataset_v = dataset[~mask_train]
    labels_t = labels[mask_train]
    labels_v = labels[~mask_train]
    
    print('Shape of train set : {}, shape of valid set: {},\ntrain labels: {}, valid labels: {}'.format(
    dataset_t.shape, dataset_v.shape, labels_t.shape, labels_v.shape))
    
    if return_mask:
        return dataset_t, labels_t, dataset_v, labels_v, mask_train
    else:
        return dataset_t, labels_t, dataset_v, labels_v

        
    
def analyse_dataset(imgs, lbls, name=None):
    """Analyse labelled dataset

    # Arguments:
    imgs: ndarray, a set of images
    lbls: ndarray, labels for a set of images
    """
    if name is not None:
        print('Dataset: {}'.format(name))

    unique_lbl, counts = np.unique(lbls, return_counts=True)
    min_samples = min(counts)
    max_samples = max(counts)
    avr_samples = np.mean(counts)
    std_dev = np.std(counts)

    imgs_dict = dict()
    imgs_dict['name'] = name
    imgs_dict['n_samples'] = imgs.shape[0]
    imgs_dict['samples_shape'] = imgs.shape[1:]
    imgs_dict['n_unique_labels'] = len(counts)
    imgs_dict['unique_labels'] = unique_lbl
    imgs_dict['min_samples'] = min_samples
    imgs_dict['max_samples'] = max_samples
    imgs_dict['average_samples'] = round(avr_samples, 0)
    imgs_dict['std_dev'] = round(std_dev, 2)
    for k,v in imgs_dict.items():
        print('{}: {}'.format(k,v))
    return imgs_dict
        