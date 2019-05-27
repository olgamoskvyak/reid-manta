import argparse, os, sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from math import ceil
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.utils import to_categorical

from model.triplet import TripletLoss
from model.siamese import Siamese
from model.triplet_pose_model import TripletLossPoseInv
from model.classification_model import Classification
from utils.batch_generators import BatchGenerator, PairsImageDataGenerator
from utils.preprocessing import read_dataset, analyse_dataset, split_classes, split_classification
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
        #if Classification => convert to one-hot encoding
        if config['model']['type'] == 'Classification':
            train_labels = to_categorical(train_labels)
            valid_labels = to_categorical(valid_labels)
    else:
        print('No test set. Splitting train set...')
        imgs, labels, label_dict = read_dataset(config['data']['train_image_folder'])   
        print('Label encoding: ', label_dict)
        if config['model']['type'] in ('TripletLoss', 'TripletPose', 'Siamese'):
            train_imgs, train_labels, valid_imgs, valid_labels = split_classes(imgs, labels, seed=config['data']['split_seed'],
                                                                          split_num=split_num)
        elif config['model']['type'] == 'Classification':
            train_imgs, train_labels, valid_imgs, valid_labels = \
                                    split_classification(imgs, labels, min_imgs=config['evaluate']['move_to_dataset'], 
                                                                                      return_mask=False)
        #Convert labels to one-hot encoding:
            train_labels = to_categorical(train_labels)
            valid_labels = to_categorical(valid_labels)
        else:
            raise Exception('Define Data Split for the model type')
        #Delete futher unused variables to clear space
        del imgs
        del labels


    analyse_dataset(train_imgs, train_labels, 'train')
    analyse_dataset(valid_imgs, valid_labels, 'valid')
                       
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
    elif config['model']['type'] == 'Siamese':
            mymodel = Siamese(**model_args)
    elif config['model']['type'] == 'TripletPose':
        model_args['n_poses'] = config['model']['n_poses']
        model_args['bs'] = config['train']['cl_per_batch'] * config['train']['sampl_per_class']
        mymodel = TripletLossPoseInv(**model_args)
    elif config['model']['type'] == 'Classification':
        model_args['embedding_size'] = train_labels.shape[1]
        mymodel = Classification(**model_args)
    else:
        raise Exception('Model type {} is not supported'.format(config['model']['type']))
    
    ##############################
    #   Load initial weights 
    ##############################
    SAVED_WEIGHTS = os.path.join(exp_folder, 'best_weights.h5')
    PRETRAINED_WEIGHTS = config['train']['pretrained_weights']
    
    if os.path.exists(SAVED_WEIGHTS):
        print("Loading saved weights in ", SAVED_WEIGHTS)
        mymodel.load_weights(SAVED_WEIGHTS)
        warm_up_flag = False
    elif os.path.exists(PRETRAINED_WEIGHTS):
        warm_up_flag == False
    else:
        print("No pre-trained weights are found")
        warm_up_flag = True

    
    ############################################
    # Make train and validation generators
    ############################################ 
    
    if config['train']['aug_rate'] == 'manta':
        gen_args = dict(rotation_range=360,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            data_format=K.image_data_format(),
            fill_mode='nearest',
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
                                        aug_gen = gen,
                                        p = config['train']['cl_per_batch'], 
                                        k = config['train']['sampl_per_class'],
                                        equal_k = config['train']['equal_k'])
        valid_generator = BatchGenerator(valid_imgs, valid_labels, 
                                        aug_gen = gen,
                                        p = config['train']['cl_per_batch'], 
                                        k = config['train']['sampl_per_class'],
                                        equal_k = config['train']['equal_k'])
        
    elif config['model']['type'] == 'TripletPose':
        gen = ImageDataGenerator(**gen_args)

        gen_params = dict( aug_gen = gen,
                            p = config['train']['cl_per_batch'], 
                            k = config['train']['sampl_per_class'],
                            equal_k = config['train']['equal_k'],
                            n_poses = config['model']['n_poses'],
                            rotate_poses = config['model']['rotate_poses'],
                            flatten_batch= True,
                            perspective = config['model']['perspective'])

        train_generator = BatchGenerator(train_imgs, train_labels, **gen_params)
        valid_generator = BatchGenerator(valid_imgs, valid_labels, **gen_params)

    elif config['model']['type'] == 'Siamese':
        gen = PairsImageDataGenerator(**gen_args)
        train_generator = gen.flow(train_imgs, train_labels,
                                        batch_size=config['train']['batch_size'], seed=0)
        valid_generator = gen.flow(valid_imgs, valid_labels,
                                        batch_size=config['train']['batch_size'], seed=1)

    elif config['model']['type'] == 'Classification':
        gen = ImageDataGenerator(**gen_args)
        train_generator = gen.flow(train_imgs, train_labels, batch_size=config['train']['batch_size'])
        valid_generator = gen.flow(valid_imgs, valid_labels, batch_size=config['train']['batch_size'])
    else:
        raise Exception('Define Data Generator for the model type')
        

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
    
    if config['model']['type'] in ('TripletLoss', 'TripletPose'):
            batch_size = config['train']['cl_per_batch'] * config['train']['sampl_per_class']
    elif config['model']['type'] in ('Siamese', 'Classification'):
        batch_size = config['train']['batch_size']
    else:
        raise Exception('Define batch size for a model type!')
    steps_per_epoch = train_imgs.shape[0] // batch_size + 1
        
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
                                 debug              = config['train']['debug'])
    
    for iteration in range(n_iter):
        print('-------------Starting iteration {} -------------------'.format(iteration+1))
        startTrainingTime = datetime.now()
        
        #Add weights to balance losses if required
        weights = [1., 1.]

        mymodel.train(  train_gen           = train_generator,
                         valid_gen          = valid_generator,
                         nb_epochs          = config['train']['log_step'], 
                         batch_size         = config['train']['batch_size'],
                         learning_rate      = config['train']['learning_rate'], 
                         steps_per_epoch    = steps_per_epoch,
                         distance           = config['train']['distance'],
                         saved_weights_name = SAVED_WEIGHTS,
                         logs_file          = LOGS_FILE,
                         debug              = config['train']['debug'],
                         weights            = weights)
        ############################################
        # Plot training history
        ############################################
        mymodel.plot_history(LOGS_FILE, from_epoch=0, showFig = False, saveFig = True, figName = PLOT_FILE)
       
        if config['model']['type'] in ('TripletLoss', 'TripletPose', 'Siamese'):
            print('Evaluating...')
            train_preds = mymodel.preproc_predict(train_imgs, config['train']['batch_size'])
            valid_preds = mymodel.preproc_predict(valid_imgs, config['train']['batch_size'])
            
            print('Shape of computed predictions', train_preds.shape, valid_preds.shape)

            acc, stdev = evaluate_1_vs_all(train_preds, train_labels, valid_preds, valid_labels,
                                                    n_eval_runs=config['evaluate']['n_eval_epochs'], 
                                                    move_to_db = config['evaluate']['move_to_dataset'],
                                                    k_list = config['evaluate']['accuracy_at_k'])
        
            #Calc execution time for each iteration
            iterationTime = datetime.now() - startTrainingTime
            print('Iteration {} finished, time {}'.format(iteration+1, iterationTime))
            
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