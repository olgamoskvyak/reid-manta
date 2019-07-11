import matplotlib
matplotlib.use('Agg')

from glob import glob
import os, csv, random
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


def make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).
    Copied from https://github.com/keras-team/keras/blob/master/keras/engine/training_utils.py
    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.
    # Returns
        A list of tuples of array indices.
    """
    num_batches = (size + batch_size - 1) // batch_size  # round up
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(num_batches)]

def export_emb(emb, info=None, folder='', prefix='prefix', info_header=None):
    """Export embeddings and extra information (labels, filenames) to csv file
    Input:
    emb: 2D float array (num_emb, emb_size):  embeddings
    info: list of string 1D arrays of size (num_emb,): extra information to each embedding: label, filename, class_name
    folder: string, folder to save files
    prefix: string to add to each filename
    info_headers: list of strings, lisf of headers for info list
    """
    if folder !='' and not os.path.exists(folder): os.makedirs(folder)
    
    #Save embeddings as csv
    emb_header = ['emb_'+str(i) for i in range(emb.shape[1])]
    emb_header = ','.join(map(str, emb_header))
    filename_emb = os.path.join(folder, prefix+'_emb.csv')
    np.savetxt(filename_emb, emb, fmt='%s', delimiter=',', header=emb_header, comments='')
    print('Embeddings and are saved to file: {}'.format(filename_emb))
    
    if info is not None:
        #Save labels and filenames to a csv file
        if info_header is None:
            info_header = ['info_'+str(i) for i in range(len(info))]
        info_header = ','.join(map(str, info_header[:len(info)]))
        info_to_file = np.stack(info, axis=-1)

        filename_info = os.path.join(folder, prefix+'_lbl.csv')
        np.savetxt(filename_info, info_to_file, fmt='%s', delimiter=',', header=info_header, comments='')
        print('Info is saved to file: {}'.format(filename_info))    


def plot_some(imgs, k=5, random_seed = None, same_order=False, labels=None):
    """Displays k random images from list
    Input:
    imgs: list of images
    k: integer, number of images to display
    random_seed: integer, number to initialise random generation
    same_order: boolean, if True, displays k first images
    """
    if len(imgs)<k:
        k=len(imgs)
    
    #Show some images
    fig, ax = plt.subplots(ncols=k, figsize=(12, 12*k))
    if same_order:
        idx=range(k)
    else:
        if random_seed is not None:
            random.seed(random_seed)
        idx = random.sample(range(len(imgs)), k)
    
    if(len(imgs[0].shape) > 2):
        if imgs[0].shape[-1] > 1:
            for i in range(k):
                ax[i].imshow(imgs[idx[i]])
        else:
            for i in range(k):
                ax[i].imshow(np.squeeze(imgs[idx[i]]), cmap='gray')
    else:
        for i in range(k):
            ax[i].imshow(imgs[idx[i]], cmap='gray')            
    if labels is not None:
        for i in range(k):
            ax[i].set_title(labels[idx[i]])
            
    for i in range(k):
        ax[i].axis('off')

    plt.tight_layout()
    
def plot_pairs(arr1, arr2, labels, class1=[], class2=[], offset=0):
    """Prints four pairs of images in two rows.
    arr1 - 4D array of images
    arr2 - 4D array of images
    labels - 1D array where 0 - positive pair, 1 - negative pair
    class1 - 1D array of classes for first array
    class2 - 1D array of classes for second array
    offset - starting index to display images from array
    """
    fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(16, 8))

    for i in range(4):
        ax[0,i].imshow(arr1[i+offset])
        ax[1,i].imshow(arr2[i+offset])
        title = 'Positive' if (labels[i+offset] == 0)  else 'Negative'
        if len(class1)>0:
            title += " / " + str(class1[i])
        ax[0,i].set_title(title)
        
        if len(class2)>0:
            title2=class2[i]
            ax[1,i].set_title(title2)

    plt.tight_layout()
    
def print_nested(val, nesting = -5):
    """Print nested json. Copied from https://www.quora.com/How-do-I-nicely-print-a-nested-dictionary-in-Python"""
    if type(val) == dict:
        print('')
        nesting += 5
        for k in val:
            print(nesting * ' ', end='')
            print(k, end=':')
            print_nested(val[k],nesting)
    else:
        print(val)

def create_subfolders(src, dest):
    """Copy folder structure from src to dest
    src - source directory with 1 level of subfolders
    dest - destination directory
    """
    g = glob(src + '/*')
    n = len(g)
    print('Found %d subfolders' % n)
    
    if not os.path.exists(dest): os.makedirs(dest)
    count_created = 0
    for i in range(n):
        (head, tail) = os.path.split(g[i])
        if not os.path.exists(os.path.join(dest, tail)): 
            os.makedirs(os.path.join(dest, tail))
            count_created += 1
    print('Created %d subfolders' % count_created)
    
def read_dir(dir):
    """Read dataset in folder where each class is in separate folder
        """
    g = glob(dir+"/*/*")
    print('Found {} files'.format(len(g)))
    return g

def plot_model_loss_csv(file, from_epoch=0, showFig = True, saveFig = False, figName = 'plot.png'):
    model_history= genfromtxt(file, delimiter=',')
    fig, axs = plt.subplots(1,1,figsize=(6,4))
    # summarize history for loss
    axs.plot(range(1,len(model_history[from_epoch:, 1])+1), model_history[from_epoch:, 1])
    axs.plot(range(1,len(model_history[from_epoch:, 2])+1),model_history[from_epoch:, 2])
    axs.set_title('Model Loss')
    axs.set_ylabel('Loss')
    axs.set_xlabel('Epoch')
    axs.set_xticks(np.arange(1,len(model_history[from_epoch:, 1])+1),len(model_history[from_epoch:, 1])/10)
    axs.legend(['train', 'val'], loc='best')
    if showFig:
        plt.show()
    if saveFig:
        fig.savefig(figName)
    plt.close(fig)
        
def plot_model_loss_acc_csv(file, from_epoch=0, showFig = True, saveFig = False, figName = 'plot.png'):
    model_history= genfromtxt(file, delimiter=',')
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history[from_epoch:, 1])+1), model_history[from_epoch:, 1])
    axs[0].plot(range(1,len(model_history[from_epoch:, 3])+1),model_history[from_epoch:, 3])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history[from_epoch:, 1])+1),len(model_history[from_epoch:, 1])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history[from_epoch:, 2])+1),model_history[from_epoch:, 2])
    axs[1].plot(range(1,len(model_history[from_epoch:, 4])+1),model_history[from_epoch:, 4])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history[from_epoch:, 2])+1),len(model_history[from_epoch:, 2])/10)
    axs[1].legend(['train', 'val'], loc='best')
    if showFig:
        plt.show()
    if saveFig:
        fig.savefig(figName)
    plt.close(fig)  
        
    
def save_res_csv(results, filename):
    """Save dictionary with results to a csv file
    Input:
    results: dictionary, keys will be headers for the csv file, values - rows
    filename: string, name for csv file (eg. results.csv)
    """
    exp_header = [k for k, v in results.items()]
    exp_data = [v for k, v in results.items()]

    #Log iteration results. If file does not exist yet, create file with header
    if not os.path.isfile(filename):
        with open(filename, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(exp_header)
            print('File {} is created'.format(filename))
    #TODO add check if file exists, that header row is the same as header from data

    with open(filename, "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(exp_data)
            
def sort2(x,y):
    """Sort one array based on another array
    Input:
    x - 1D array, array to sort
    y - 1D array, elements in x are sorted based on y
    Returns:
    sorted numpy arrays
    """
    return np.array([l for _,l in sorted(zip(y,x))]), np.array([l for l,_ in sorted(zip(y,x))])

def rem_dupl(seq, seq2 = None):
    """Remove duplicates from a sequence and keep the order of elements. Do it in unison with a sequence 2."""
    seen = set()
    seen_add = seen.add
    if seq2 is None:
        return [x for x in seq if not (x in seen or seen_add(x))]
    else:
        a = [x for x in seq if not (x in seen or seen_add(x))]
        seen = set()
        seen_add = seen.add
        b = [seq2[i] for i, x in enumerate(seq) if not (x in seen or seen_add(x))]
        return a, b
    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def rgb2gray(rgb, data_type='uint8'):
    """Convert from RBG to gray-scale image.
    rbg: 4d or 3d ndarray of RGB image/images
    data_type: string, desired data type of output
    """
    gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    gray = np.stack((gray, gray, gray), -1).astype(data_type)
    return gray
