'''
===============================================================================
Interactive Tool to Re-Identify an Animal by natural markings

This tool allows a user to find a matching animal individuals based on database.

USAGE:
    python predict.py -i <path_to_image> -c <config_file.json> -o <path_to_output>
    
README FIRST:
    Draw a line around object of interest. Mask is saved automatically.
    To change saved mask simply draw a new line. Mask will be updated.
    
PREREQUISITES:
    Embeddings for the database images are precomputed and saved at
    the location specified in config.prod.embeddings

FUNCTIONALITY:
    1) Query image is preprocessed:
        a) the user is asked to draw a line around the pattern of interest (optional)
        b) query image is cropped by a provided line
        c) query image is resized to the size specified in the config
    2) Embeddings for the query image are computed.
    3) Embeddings for the database images are read from csv files.
    4) Matching individuals are returned.

===============================================================================
'''
import os, argparse, json
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from glob import glob

from utils.drawer import MaskDrawer

from model.triplet import TripletLoss
from utils.utils import export_emb, print_nested, str2bool
from utils.preprocessing import crop_im_by_mask, resize_imgs, convert_to_fmt
from evaluation.evaluate_accuracy import predict_k_neigh

argparser = argparse.ArgumentParser(
    description='Predict on image. Find matching individual from a database.')

argparser.add_argument('-i', '--impath', required = True, help='Path to image to process')
argparser.add_argument('-d', '--display', type=str2bool, help='Display images and predictions on the screen: True of False. Default: True.', default=True)
argparser.add_argument('-c','--conf', help='Path to configuration file', default='configs/config-whale.json')
argparser.add_argument('-o', '--output', help='Path to output files. Default is in config: prod.temp')
argparser.add_argument('-g', '--gtruth', help='Ground truth labels for testing. Csv: filename, label (w/headers)')

    
if __name__ == "__main__":
     # print documentation
    print(__doc__)
    
    #Get command line parameters
    args = argparser.parse_args()
    impath = args.impath
    config_path = args.conf
    display = args.display
    
    #Check input parameters
    if not os.path.isfile(impath):
        raise ValueError('Image file does not exist or not a file. Check input. Current input: ', impath)
        
    if not os.path.isfile(config_path):
        raise Exception('Config file does not exist or not a file. Check input. Current input: ', config_path)
    
    #Read config file
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    print('Config parameters:')
    print_nested(config, nesting = -2)  
    
    if args.output is None:
        output_dir = config['prod']['temp']
    else:
        output_dir = args.output
        
    if not isinstance(display, bool):
        raise ValueError('Display mode should be True or False. Check input. Current input: ', display)
        
    if not args.gtruth:
        gtruth_exists = False
    else:
        gtruth_exists = True
        if not os.path.exists(args.gtruth):
            raise('Ground truth file is not found in ', args.gtruth)
        else:
            gtruth = np.genfromtxt(args.gtruth, delimiter=',', dtype=str, skip_header=1)
        
        
    print('Output files will be stored in {}'.format(output_dir))
    if not os.path.exists(output_dir): os.makedirs(output_dir)
            
    #Get location of precomputed embeddings
    saved_emb = config['prod']['embeddings']
    
    #Set directories for processed images
    mask_directory = os.path.join(output_dir, 'masks')
    cropped_dir = os.path.join(output_dir, 'cropped')
    resized_dir = os.path.join(output_dir, 'resized')
    predict_dir = os.path.join(output_dir, 'predictions')
          
    if not os.path.exists(mask_directory): os.makedirs(mask_directory)
    if not os.path.exists(cropped_dir): os.makedirs(cropped_dir)
    if not os.path.exists(resized_dir): os.makedirs(resized_dir)
    if not os.path.exists(predict_dir): os.makedirs(predict_dir)
        
    size = (config['model']['input_width'], config['model']['input_height']) 

    #Set path to save processed images
    (imdir, imfile) = os.path.split(impath)
    maskpath = os.path.join(mask_directory, imfile)
    
    if display:
        #Ask user to draw a mask around the image
        md = MaskDrawer(impath, maskpath)
        respond = md.run()
        if respond in ('exit', 'next'):
            print('Exiting the program...')
            quit()
        if respond == 'save' and md.done:
            #Crop image by mask
            square = (config['model']['input_width']==config['model']['input_height'])
            croppedpath = crop_im_by_mask(impath, maskpath, cropped_dir, padding = 0, square=square)
        if respond == 'save' and not md.done:
            croppedpath = impath
    else:
        croppedpath = impath
        
    #Resize image to fit network input
    resizedpath = resize_imgs(croppedpath, resized_dir, size)
    resizedpath = convert_to_fmt(resizedpath, imformat = 'png')
        
    #Prediction step
    print('Loading model...')
    INPUT_SHAPE = (config['model']['input_height'], config['model']['input_width'], 3)
    model_args =     dict(backend         = config['model']['backend'],
                        frontend          = config['model']['frontend'],
                        input_shape       = INPUT_SHAPE,
                        embedding_size    = config['model']['embedding_size'],
                        connect_layer     = config['model']['connect_layer'],
                        train_from_layer  = config['model']['train_from_layer'],
                        loss_func         = config['model']['loss'],
                        weights           = None)
    
    mymodel = TripletLoss(**model_args)

    exp_folder = os.path.join(config['train']['exp_dir'], config['train']['exp_id'])
    saved_weights = os.path.join(exp_folder, 'best_weights.h5')
    
    if os.path.exists(saved_weights):
        print("Loading saved weights in ", saved_weights)
        mymodel.load_weights(saved_weights)
    else:
        raise("ERROR! No pre-trained weights are found in ", saved_weights)
    
    #compute embedding for the image
    image = imread(resizedpath)
    embedding = mymodel.preproc_predict(np.expand_dims(image, 0))
    
    print('Exporting computed embedding to csv files...')
    folder=os.path.join(output_dir, 'query-embeddings')
    emb_prefix = os.path.splitext(os.path.basename(impath))[0]
    export_emb(embedding, info=[[impath], [impath], [impath]], folder=folder, prefix=emb_prefix, info_header=['label,filename,name'])
    
    db_embs = []
    db_lbls = []
    db_info = []
    
    #Read embeddings and info from a list of provided csv files
    emb_list = glob(saved_emb+'/*')
    
    for i, file in enumerate(emb_list):
        if '_emb.csv' in file:
            print('Reading embeddings from file {}'.format(file))
            read_embs = np.genfromtxt(file, delimiter=',', skip_header=1)
            if len(read_embs.shape)==1:
                read_embs = np.expand_dims(read_embs, 0)
            db_embs.append(read_embs)
            
            lab_file = file[:-(len('_emb.csv'))] + '_lbl.csv'
            print('Reading corresponding labels from file {}'.format(lab_file))
            read_info = np.genfromtxt(lab_file, dtype=str, delimiter=',', skip_header=1)
            if len(read_info.shape)==1:
                read_info = np.expand_dims(read_info, 0)
            db_lbls.append(read_info[:,0])
            db_info.append(read_info[:,1:])

    #Concatenate together
    db_embs = np.concatenate(db_embs, axis=0)
    db_lbls = np.concatenate(db_lbls, axis=0)
    db_info = np.concatenate(db_info, axis=0)
    #print(db_embs.shape, db_lbls.shape, db_info.shape)
    
    #Fit nearest neighbours classifier
    neigh_lbl_un, neigh_ind_un, neigh_dist_un = predict_k_neigh(db_embs, db_lbls, embedding, k=10)
    
    #print(neigh_lbl_un, neigh_ind_un, neigh_dist_un)
    neigh_lbl_un = neigh_lbl_un[0]
    neigh_ind_un = neigh_ind_un[0]
    neigh_dist_un = neigh_dist_un[0]
    
    pred_imgs = [imread(db_info[idx, 0]) for idx in neigh_ind_un]
    pred_lbl = [db_info[idx, 1] for idx in neigh_ind_un]  
    pred_files = [db_info[idx, 0] for idx in neigh_ind_un]
    
    #Show some images
    n_preds = len(neigh_lbl_un)
    
    fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20, 10))
    
    if gtruth_exists:
        true_lab = dict(gtruth)[imfile]
        if true_lab in pred_lbl:
            match_num = np.where(np.array(pred_lbl)==true_lab)[0][0] + 1
            fig.suptitle('Correct match is in top-{}.\nGround truth info is provided in {}'.format(match_num, args.gtruth), fontsize=16)
        else:
            fig.suptitle('No match in top-10 predictions.\nGround truth is provided in {}'.format(args.gtruth), fontsize=16)     
    else:
        fig.suptitle('Top-10 predictions for the image {}.\nNo ground truth info is provided.'.format(imfile), fontsize=16)
    
    for i in range(6):
        for j in range(2):
            ax[j,i].axis('off')

    #Query
    ax[0,0].imshow(image)
    ax[0,0].set_title('Query image \n'+imfile[:20]+'\n'+imfile[20:])
    ax[0,0].text(.5, -0.1, 'Distance from query:', ha='center',transform=ax[0,0].transAxes)

    #Predictions
    for i in range(5):
        for j in range(2):
            if i+j*5 < len(pred_imgs):
                if(len(pred_imgs[i].shape) > 2):
                    ax[j,i+1].imshow(pred_imgs[i+j*5])
                else:
                    ax[j,i+1].imshow(pred_imgs[i+j*5], cmap='gray')
                file = os.path.split(pred_files[i+j*5])[-1]
                label = pred_lbl[i+j*5]
                ax[j,i+1].set_title('Prediction '+str(i+j*5+1)+':\n'+label+'\n'+file[-25:])
                ax[j,i+1].text(.5, -0.2, round(neigh_dist_un[i+j*5],4), ha='center',transform=ax[j,i+1].transAxes)
    
    prediction_file = os.path.join(predict_dir, 'preds_for_'+imfile)
    plt.savefig(prediction_file)
    print('Prediction is saved to: {}'.format(prediction_file))
    if display:
        plt.show()

    print('Program is finished')
    
    