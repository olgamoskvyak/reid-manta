import argparse, os, sys, json

import numpy as np
from math import ceil
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

from model.triplet import TripletLoss
from utils.batch_generators import BatchGenerator
from utils.preprocessing import read_dataset, analyse_dataset, split_classes
from utils.utils import print_nested, save_res_csv
from evaluation.evaluate_accuracy import evaluate_1_vs_all


argparser = argparse.ArgumentParser(description='Train and validate a model on any dataset')

argparser.add_argument('-c','--conf', help='path to the configuration file', default='config.json')
argparser.add_argument('-s','--split_num', help='index of split for K-fold: number from [0,4] or -1 if no K-fold', type=int, default=-1)

def _main_(args):
    
    #Record start time:
    startTime = datetime.now()
    
    ###############################
    #  Read config with parameters and command line params
    ###############################
    config_path = args.conf
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    split_num = args.split_num
    
    ###############################
    #  Create folders for logs
    ###############################
    exp_folder = os.path.join(config['train']['exp_dir'], config['train']['exp_id'])
    if split_num >= 0 :
        exp_folder = exp_folder + '-split-' + str(split_num)
    if not os.path.exists(exp_folder): os.makedirs(exp_folder)
       
    ###############################
    #  Redirect print output to logs
    ###############################
    FULL_PRINT_LOG = os.path.join(exp_folder,'full_print_output.log')
    
    if config['general']['stdout-file']:
        sys.stdout = open(FULL_PRINT_LOG, 'a+')
    print('='*40)
    print('Date: {} / Experiment id: {}'.format(datetime.now(), config['train']['exp_id']))
    print('Config parameters:')
    print_nested(config, nesting = -2)
    print('='*40)
                
    ##############################
    #   Construct the model 
    ##############################
    INPUT_SHAPE = (config['model']['input_height'], config['model']['input_width'], 3)
    
    model_args = dict(backend           = config['model']['backend'],
                            frontend          = config['model']['frontend'],
                            input_shape       = INPUT_SHAPE,
                            embedding_size    = config['model']['embedding_size'],
                            connect_layer     = config['model']['connect_layer'],
                            train_from_layer  = config['model']['train_from_layer'],
                            loss_func         = config['model']['loss'],
                            weights            = 'imagenet')
    
    if config['model']['type'] == 'TripletLoss':
        mymodel = TripletLoss(**model_args)
    else:
        raise Exception('Only TripletLoss is supported')    
    
    ##############################
    #   Load initial weights 
    ##############################
    #if continuing experiment => load weights from prev step
    #if new experiment => use pretrained weights (if any)  
    SAVED_WEIGHTS = os.path.join(exp_folder, 'best_weights.h5')
    PRETRAINED_WEIGHTS = config['train']['pretrained_weights']
    
    warm_up_flag = False    
    if os.path.exists(SAVED_WEIGHTS):
        print("Loading saved weights in ", SAVED_WEIGHTS)
        mymodel.load_weights(SAVED_WEIGHTS)
    elif os.path.exists(PRETRAINED_WEIGHTS):
        print("Loading pre-trained weights in ", PRETRAINED_WEIGHTS)
        mymodel.load_weights(PRETRAINED_WEIGHTS, by_name=True)
    else:
        print("No pre-trained weights are found")
        warm_up_flag = True

    ###############################
    #   Get dataset and labels
    ###############################

    #Get test set if exists, otherwise split train set
    if os.path.exists(config['evaluate']['test_set']):
        print('Loading test set from {}'.format(config['evaluate']['test_set']))
        valid_imgs, valid_names, _ = read_dataset(config['evaluate']['test_set'], 
                                                original_labels=True)
        train_imgs, train_names, _ = read_dataset(config['data']['train_image_folder'], 
                                                  original_labels=True)
        overlap = all(np.isin(train_names, valid_names))
        print('Overlap between train and valid set in individual names: ', overlap)
        
        unique_names = np.unique(np.concatenate((valid_names, train_names)))
        name2lab = dict(enumerate(unique_names))
        name2lab.update({v:k for k,v in name2lab.items()})
        train_labels = np.array([name2lab[name] for name in train_names])
        valid_labels = np.array([name2lab[name] for name in valid_names])
        
    else:
        print('No test set. Splitting train set...')
        imgs, labels, _ = read_dataset(config['data']['train_image_folder'])   
        train_imgs, train_labels, valid_imgs, valid_labels = split_classes(imgs, labels, seed=config['data']['split_seed'],
                                                                          split_num=split_num)
    
    
    analyse_dataset(train_imgs, train_labels, 'train')
    analyse_dataset(valid_imgs, valid_labels, 'valid')
            
    ############################################
    # Make train and validation generators
    ############################################ 
    
    if config['train']['aug_rate'] == 'manta':
        gen_args = dict(rotation_range=90,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            data_format=K.image_data_format(),
            fill_mode='nearest',
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=mymodel.backend_class.normalize)
    elif config['train']['aug_rate'] == 'whale':
        gen_args = dict(rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            data_format=K.image_data_format(),
            fill_mode='nearest',
            preprocessing_function=mymodel.backend_class.normalize)
    else:
        raise Exception('Define augmentation rate in config!')
        
    
    if config['model']['type'] == 'TripletLoss':
        gen = ImageDataGenerator(**gen_args)
        train_generator = BatchGenerator(train_imgs, train_labels, 
                                         aug_gen=gen,
                                         p=config['train']['cl_per_batch'], 
                                         k=config['train']['sampl_per_class'])
        valid_generator = BatchGenerator(valid_imgs, valid_labels, 
                                         aug_gen=gen,
                                         p=config['train']['cl_per_batch'], 
                                         k=config['train']['sampl_per_class'])
        
    else:
        raise Exception('Model type is not supported')
        

    #Compute preprocessing time:
    preprocTime = datetime.now() - startTime
    print('Preprocessing time is {}'.format(preprocTime))
    
    ###############################
    #   Training 
    ###############################
    LOGS_FILE = os.path.join(exp_folder, 'history.csv')
    PLOT_FILE = os.path.join(exp_folder, 'plot.png')    
    ALL_EXP_LOG = os.path.join(config['train']['exp_dir'], 'experiments_all.csv')
    
    n_iter = ceil(config['train']['nb_epochs'] / config['train']['log_step'])
    
    if config['model']['type'] == 'TripletLoss':
        batch_size = config['train']['cl_per_batch'] * config['train']['sampl_per_class']
    else:
        raise Exception('Define batch size for a model type!')
    steps_per_epoch = train_imgs.shape[0] // batch_size
        
    print('Steps per epoch: {}'.format(steps_per_epoch))
    
    if warm_up_flag:
        print('-----First training. Warm up epochs to train random weights with higher learning rate--------')
        mymodel.warm_up_train(  train_gen           = train_generator,
                                 valid_gen          = valid_generator,
                                 nb_epochs          = 1, 
                                 batch_size         = config['train']['batch_size'],
                                 learning_rate      = config['train']['learning_rate'] * 10, 
                                 steps_per_epoch    = steps_per_epoch,
                                 distance           = config['train']['distance'],
                                 saved_weights_name = SAVED_WEIGHTS,
                                 logs_file          = LOGS_FILE,
                                 plot_file          = PLOT_FILE,
                                 debug              = config['train']['debug'])
    
    for iteration in range(n_iter):
        print('-------------Starting iteration {} -------------------'.format(iteration+1))
        startTrainingTime = datetime.now()
        
        mymodel.train(  train_gen           = train_generator,
                         valid_gen          = valid_generator,
                         nb_epochs          = config['train']['log_step'], 
                         batch_size         = config['train']['batch_size'],
                         learning_rate      = config['train']['learning_rate'], 
                         steps_per_epoch    = steps_per_epoch,
                         distance           = config['train']['distance'],
                         saved_weights_name = SAVED_WEIGHTS,
                         logs_file          = LOGS_FILE,
                         plot_file          = PLOT_FILE,
                         debug              = config['train']['debug'])
        ############################################
        # Plot training history
        ############################################
        mymodel.plot_history(LOGS_FILE, from_epoch=0, showFig = False, saveFig = True, figName = PLOT_FILE)
       
        print('Evaluating...')
        train_preds = mymodel.preproc_predict(train_imgs, config['train']['batch_size'])
        valid_preds = mymodel.preproc_predict(valid_imgs, config['train']['batch_size'])
        
        acc, stdev = evaluate_1_vs_all(train_preds, train_labels, valid_preds, valid_labels,
                                                 n_eval_runs=config['evaluate']['n_eval_epochs'], 
                                                 move_to_db = config['evaluate']['move_to_dataset'],
                                                 k_list = config['evaluate']['accuracy_at_k'])
    
        #Calc execution time for each iteration
        iterationTime = datetime.now() - startTrainingTime
        print('Iteration {}, time {}'.format(iteration+1, iterationTime))
        
        #Collect data for logs
        result = dict()
        result['date_time'] = datetime.now()
        result['config'] = config_path
        result['experiment_id'] = exp_folder
        result['iteration_time'] = iterationTime
        result['images'] = config['data']['train_image_folder']
        result['input_height'] = config['model']['input_height']
        result['input_width'] = config['model']['input_width']
        result['backend'] = config['model']['backend']
        result['connect_layer'] = config['model']['connect_layer']
        result['frontend'] = config['model']['frontend']
        result['train_from_layer'] = config['model']['train_from_layer']
        result['embedding_size'] = config['model']['embedding_size']
        result['learning_rate'] = config['train']['learning_rate']
        result['nb_epochs'] = config['train']['log_step']
        result['acc1'] = round(acc[1], 2)
        result['acc5'] = round(acc[5], 2)
        result['acc10'] = round(acc[10], 2)
        result['move_to_dataset'] = config['evaluate']['move_to_dataset']

        save_res_csv(result, ALL_EXP_LOG)
                
        if iteration % 50 and iteration > 0:
            time_finish = datetime.now().strftime("%Y%m%d-%H%M%S") + '_iter_' + str(iteration)
            TEMP_WEIGHTS = os.path.join(exp_folder, 'weights_at_' + time_finish + '.h5')
            mymodel.model.save_weights(TEMP_WEIGHTS)
    #------End For each Iteration--------------# 
    
    #Save weights at the end of experiment
    time_finish = datetime.now().strftime("%Y%m%d-%H%M%S") + '_last'
    TEMP_WEIGHTS = os.path.join(exp_folder, 'weights_at_' + time_finish + '.h5')
    mymodel.model.save_weights(TEMP_WEIGHTS)
    
    
if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)