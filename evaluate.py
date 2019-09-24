import argparse, os, sys, json
import matplotlib
matplotlib.use('Agg')

import numpy as np
from datetime import datetime

from model.triplet import TripletLoss
from model.siamese import Siamese
from model.triplet_pose_model import TripletLossPoseInv
from utils.utils import print_nested, save_res_csv, export_emb
from utils.preprocessing import read_dataset, analyse_dataset, split_classes
from evaluation.evaluate_pairs import evaluate_pairs
from evaluation.evaluate_accuracy import evaluate_1_vs_all

argparser = argparse.ArgumentParser(description='Evaluate model on any dataset')

argparser.add_argument('-c','--conf', help='path to configuration file', default='config.json')
argparser.add_argument('-s','--split_num', help='split number for K-fold', type=int, default=-1)
argparser.add_argument('-m','--mode', help='evaluation mode: all, pairs or life', default='all')


def _main_(args):
    
    ###############################
    #  Open config with parameters
    ###############################
    
    config_path = args.conf
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    split_num = args.split_num
    mode = args.mode
        
    #Define folder for experiment
    exp_folder = os.path.join(config['train']['exp_dir'], config['train']['exp_id'])
    if split_num >= 0 :
        exp_folder = exp_folder + '-split-' + str(split_num)
    
    #Redirect output as per config
    FULL_PRINT_LOG = os.path.join(exp_folder,'full_print_output.log')
    if config['general']['stdout-file']:
        sys.stdout = open(FULL_PRINT_LOG, 'a+')
        
    print('Date: {} / EVALUATION of experiment id: {}'.format(datetime.now(), config['train']['exp_id']))
    print('Config parameters:')
    print_nested(config, nesting = -2)
    print('='*40) 
    
    ###############################
    #   Get test set and labels
    ###############################

    #Get test set if exists, otherwise split train set
    if os.path.exists(config['evaluate']['test_set']):
        print('Loading test set from {}'.format(config['evaluate']['test_set']))
        test_imgs, test_names, _, files_test = read_dataset(config['evaluate']['test_set'], 
                                                original_labels=True, return_filenames=True)
        train_imgs, train_names, _, files_train= read_dataset(config['data']['train_image_folder'], 
                                                  original_labels=True, return_filenames=True)
        overlap = all(np.isin(train_names, test_names))
        print('Overlap between train and test set in individual names: ', overlap)
        
        unique_names = np.unique(np.concatenate((test_names, train_names)))
        name2lab = dict(enumerate(unique_names))
        name2lab.update({v:k for k,v in name2lab.items()})
        train_labels = np.array([name2lab[name] for name in train_names])
        test_labels = np.array([name2lab[name] for name in test_names])
        files_train = np.array(files_train)
        files_test = np.array(files_test)        
        
    else:
        print('Loading validation split from {}'.format(config['data']['train_image_folder']))
        imgs, labels, lbl2names, filenames = read_dataset(config['data']['train_image_folder'], return_filenames=True) 
        train_imgs, train_labels, test_imgs, test_labels, mask_train = split_classes(imgs, labels, 
                                                                   seed=config['data']['split_seed'],
                                                                   split_num=split_num, return_mask=True)
        #Get filenames and names
        files_train = np.array(filenames)[mask_train]
        files_test = np.array(filenames)[~mask_train]
        
        names = np.array([lbl2names[lab] for lab in labels])
        train_names = np.array(names)[mask_train]
        test_names = np.array(names)[~mask_train]        
        
                
    analyse_dataset(train_imgs, train_labels, 'train')
    analyse_dataset(test_imgs, test_labels, 'test')
    
    ##############################
    #   Load the model 
    ##############################
    
    INPUT_SHAPE = (config['model']['input_height'], config['model']['input_width'], 3)
    model_args = dict(backend           = config['model']['backend'],
                            frontend          = config['model']['frontend'],
                            input_shape       = INPUT_SHAPE,
                            embedding_size    = config['model']['embedding_size'],
                            connect_layer     = config['model']['connect_layer'],
                            train_from_layer  = config['model']['train_from_layer'],
                            loss_func         = config['model']['loss'],
                            weights           = None,
                            show_summary      = False)
    
    if config['model']['type'] == 'TripletLoss':
        mymodel = TripletLoss(**model_args)
    elif config['model']['type'] == 'Siamese':
        mymodel = Siamese(**model_args)
    elif config['model']['type'] == 'TripletPose':
        model_args['n_poses'] = config['model']['n_poses']
        model_args['bs'] = config['train']['cl_per_batch'] * config['train']['sampl_per_class']
        mymodel = TripletLossPoseInv(**model_args)
    else:
        raise Exception('{} model type is not supported'.format(config['model']['type']))    
    
    SAVED_WEIGHTS = os.path.join(exp_folder, 'best_weights.h5')
    
    if os.path.exists(SAVED_WEIGHTS):
        print("Loading saved weights in ", SAVED_WEIGHTS)
        mymodel.load_weights(SAVED_WEIGHTS)
    else:
        print("ERROR! No pre-trained weights are found")
    
    ############################################
    # Evaluation 
    ############################################ 
    #default values
    val, far, auc = (0,0,0)
    acc = {k: 0 for k in config['evaluate']['accuracy_at_k']}
    
    if mode in ('all', 'pairs'):
        print('========Calculating VAL/FAR on all possible pairs:=======')
        PLOT_FILE = os.path.join(exp_folder, 'roc_curve.png') 

        val, far, auc = evaluate_pairs(images          = test_imgs, 
                                       labels          = test_labels, 
                                       model           = mymodel, 
                                       far_target      = config['evaluate']['far_target'],
                                       plot_file       = PLOT_FILE,
                                       sample_size     = None)
        
    if mode in ('all', '1vsall'):
        print('=====Calculating accuracy 1 class vs all other classes:====')
        #Compute embeddings
        train_preds = mymodel.preproc_predict(train_imgs, config['train']['batch_size'])
        test_preds = mymodel.preproc_predict(test_imgs, config['train']['batch_size'])
        #Evaluate accuracy over a multiple runs. Report average results.
        
        acc, stdev = evaluate_1_vs_all(train_preds, train_labels, test_preds, test_labels,
                                                 n_eval_runs=config['evaluate']['n_eval_epochs'], 
                                                 move_to_db = config['evaluate']['move_to_dataset'],
                                                 k_list = config['evaluate']['accuracy_at_k'])
        
        
    if mode in ('all', 'compute'):
        print('=====Computing and saving embeddings as csv:=========')
        #Preprocess images and get embeddings
        print('Computing embeddings...')
        train_preds = mymodel.preproc_predict(train_imgs, config['train']['batch_size'])
        test_preds = mymodel.preproc_predict(test_imgs, config['train']['batch_size'])
        
        #Export embeddings                                   
        folder = os.path.join(exp_folder, 'export')
        export_emb(train_preds, info=[train_labels, files_train, train_names], folder=folder, prefix='train', info_header=['class,file,name'])
        export_emb(test_preds, info=[test_labels, files_test, test_names], folder=folder, prefix='test', 
                   info_header=['class,file,name']) 
        
        
        
    ############################################
    # Collect logs for evaluation
    ############################################
    EVAL_LOG = os.path.join(config['train']['exp_dir'],'evaluate_log.csv')    
    result = dict()
    result['date_time'] = datetime.now()
    result['config'] = config_path
    result['experiment_id'] = config['train']['exp_id'] + '-split-' + str(split_num)
    result['mode'] = mode
    
    result['acc1'] = round(acc[1], 2)
    result['acc5'] = round(acc[5], 2)
    result['acc10'] = round(acc[10], 2)
    result['tpr'] = round(val, 3)
    result['far'] = round(far, 3)
    result['auc'] = round(auc, 3)
    
    save_res_csv(result, EVAL_LOG)

    
if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
