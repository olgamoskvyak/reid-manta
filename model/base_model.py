import os
import numpy as np
from keras.models import Model

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from backend import DummyNetFeature, InceptionV3Feature, VGG16Feature, ResNet50Feature, MobileNetV2Feature
from backend import InceptionResNetV2Feature
from utils.utils import make_batches
from top_models import glob_pool_norm, glob_pool, glob_softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

class BaseModel(object):
    def __init__(self, input_shape, backend, frontend, embedding_size, 
                 connect_layer=-1, train_from_layer = 0, distance = 'l2', weights='imagenet'):
        """Base model consists of backend feature extractor (pretrained model) and a front-end model.
        
        Input:
        backend: string: one of predefined features extractors. Name matches model name from keras.applications
        input_shape: 3D tuple of integers, shape of input tuple
        frontend: string, name of a function to define top model from top_models.py
        embedding_size: ingeter, size of produced embedding, eg. 256
        connect_layer: integer (positive or negative) or a string: either index of a layer or name of a layer
                        that is used to connect base model with top model
        train_from_layer: integer (positive or negative) or a string: either index of a layer or name of a layer 
                           to train the model from.
        distance: string, distance function to calculate distance between embeddings. TODO: implement
        
        """
        self.input_shape = input_shape
        self.embedding_size = embedding_size
        self.weights = weights
        self.backend = backend
        self.frontend = frontend
        self.feature_extractor()
        self.connect_layer = self.get_connect_layer(connect_layer)
        self.backend_model()
        self.features_shape()
        self.train_from_layer = self.get_train_from_layer(train_from_layer)
        self.top_model()
        self.distance = distance
        
        
    def feature_extractor(self):
        """ Base feature extractor """
        if self.backend == 'InceptionV3':
            self.backend_class = InceptionV3Feature(self.input_shape, self.weights) 
        elif self.backend == 'VGG16':
            self.backend_class = VGG16Feature(self.input_shape, self.weights)
        elif self.backend == 'ResNet50':
            self.backend_class = ResNet50Feature(self.input_shape, self.weights)
        elif self.backend == 'InceptionResNetV2':
            self.backend_class = InceptionResNetV2Feature(self.input_shape, self.weights)
        elif self.backend == 'DummyNet':
            self.backend_class = DummyNetFeature(self.input_shape, self.weights)
        elif self.backend == 'MobileNetV2':
            self.backend_class = MobileNetV2Feature(self.input_shape, self.weights)
        else:
            raise Exception('Architecture is not supported! Use only MobileNet, VGG16, ResNet50, and Inception3.')
            
        self.feature_extractor = self.backend_class.feature_extractor
        
    def normalize_input(self, image):
        '''Normalise input to a CNN depending on a backend'''
        return self.backend_class.normalize(image)        
        
    def backend_model(self):
        """ Model to obtain features from a specific layer of feature extractor."""
        self.backend_model = Model(inputs=self.feature_extractor.get_input_at(0), 
                                outputs=self.feature_extractor.layers[self.connect_layer].get_output_at(0), name='features_model')
        
    def features_shape(self):
        self.features_shape = self.backend_model.get_output_shape_at(0)[1:]
        print('Shape of base features: {}'.format(self.features_shape))
        
    def preproc_predict(self, imgs, batch_size=32):
        """Preprocess images and predict with the model (no batch processing for first step)
        Input:
        imgs: 4D float or int array of images
        batch_size: integer, size of the batch
        Returns:
        predictions: numpy array with predictions (num_images, len_model_output)
        """
        batch_idx = make_batches(imgs.shape[0], batch_size)
        imgs_preds = np.zeros((imgs.shape[0],)+self.model.get_output_shape_at(0)[1:])
        print('Computing predictions with the shape {}'.format(imgs_preds.shape))

        for sid, eid in batch_idx:
            preproc = self.backend_class.normalize(imgs[sid:eid])
            imgs_preds[sid:eid] = self.model.predict_on_batch(preproc)   

        return imgs_preds
        
    def top_model(self, verbose=1):
        """Model on top of features."""
        if self.frontend == 'glob_pool_norm':
            self.top_model = glob_pool_norm(embedding_size = self.embedding_size, 
                                           backend_model = self.backend_model)
        elif self.frontend == 'glob_pool':
            self.top_model = glob_pool(embedding_size = self.embedding_size, 
                                           backend_model = self.backend_model)
        elif self.frontend == 'glob_softmax':
            self.top_model = glob_softmax(embedding_size = self.embedding_size, 
                                           backend_model = self.backend_model)
        else:
            raise Exception('{} is not supported'.format(self.frontend))

        # Freeze layers as per config
        self.set_trainable()
        
    
    def get_connect_layer(self, connect_layer):
        """If connect_layer is a string (layer name), return layer index.
           If connect layer is a negative integer, return positive layer index."""
        index = None
        if isinstance(connect_layer, str):
            for idx, layer in enumerate(self.feature_extractor.layers):
                if layer.name == connect_layer:
                    index = idx
                    break
        elif isinstance(connect_layer, int):
            if connect_layer >= 0:
                index = connect_layer
            else:
                index = connect_layer + len(self.feature_extractor.layers)
        else:
            raise ValueError
            print('Check type of connect_layer')
        print('Connecting layer {} - {}'.format(index, self.feature_extractor.layers[index].name))
        return index
    
    def get_train_from_layer(self, train_from_layer):
        """If train_from_layer is a string (layer name), return layer index.
           If train_from_layer layer is a negative integer, return positive layer index."""
        index = None
        if isinstance(train_from_layer, str):
            for idx, layer in enumerate(self.feature_extractor.layers):
                if layer.name == train_from_layer:
                    index = idx
                    break
        if isinstance(train_from_layer, int):
            if train_from_layer >= 0:
                index = train_from_layer
            else:
                index = train_from_layer + len(self.feature_extractor.layers)
        print('Train network from layer {} - {}'.format(index, self.feature_extractor.layers[index].name))
        return index
            
        
    def load_weights(self, weight_path, by_name=False):
        self.model.load_weights(weight_path, by_name)
        
    def set_all_layers_trainable(self):
        for i in range(len(self.top_model.layers)):
            self.top_model.layers[i].trainable = True
            
    def set_trainable(self):
        self.set_all_layers_trainable()
        for i in range(self.train_from_layer):
            self.top_model.layers[i].trainable = False
        print('Layers are frozen as per config. Non-trainable layers are till layer {} - {}'.format(self.train_from_layer, self.top_model.layers[self.train_from_layer].name))
        

    def warm_up_train(self, train_gen,     
                            valid_gen,     
                            nb_epochs,
                            batch_size,
                            learning_rate,
                            steps_per_epoch,
                            distance = 'l2',
                            saved_weights_name='best_weights.h5',
                            logs_file = 'history.csv',
                            plot_file = 'plot.png',
                            debug=False):
        """Train only randomly initialised layers of top model"""
        # Freeze base model
        self.set_all_layers_trainable()
        
        backend_model_len = len(self.backend_model.layers)
        print('Freezeing layers before warm-up training')
        for i in range(backend_model_len):
            self.top_model.layers[i].trainable = False
        for layer in self.top_model.layers:
            print(layer.name, layer.trainable)
            
        # Compile the model
        self.compile_model(learning_rate)
        
        # Warm-up training   
        csv_logger = CSVLogger(logs_file, append = True)
        
        self.model.fit_generator(generator        = train_gen, 
                                 steps_per_epoch  = steps_per_epoch, 
                                 epochs           = nb_epochs, 
                                 verbose          = 2 if debug else 1,
                                 validation_data  = valid_gen,
                                 validation_steps = steps_per_epoch // 5 + 1,
                                 callbacks        = [csv_logger])
        
        self.top_model.save_weights(saved_weights_name)
        
        # Freeze layers as per config
        self.set_trainable()
               
    
    def train(self, train_gen,     
                    valid_gen,     
                    nb_epochs,
                    batch_size,
                    learning_rate,
                    steps_per_epoch,
                    distance = 'l2',
                    saved_weights_name='best_weights.h5',
                    logs_file = 'history.csv',
                    debug=False,
                    weights=None):     

        # Compile the model
        if weights is None:
            self.compile_model(learning_rate)
        else:
            self.compile_model(learning_rate, weights = weights)
        
        # Make a few callbacks
        early_stop = EarlyStopping(monitor='val_loss', 
                           patience=5, #changed from 3
                           min_delta=0.001, 
                           mode='min', 
                           verbose=1)
        reduce_plateau = ReduceLROnPlateau(monitor='val_loss', 
                                           patience=5, 
                                           min_delta=0.01, 
                                           factor=0.2, 
                                           min_lr=1e-7, 
                                           verbose=1)
        checkpoint = ModelCheckpoint(saved_weights_name, 
                                     monitor='val_loss', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     save_weights_only=False,
                                     mode='min', 
                                     period=1)        
        csv_logger = CSVLogger(logs_file, append = True)

        ############################################
        # Start the training process
        ############################################        
        self.model.fit_generator(generator        = train_gen, 
                                 steps_per_epoch  = steps_per_epoch, 
                                 epochs           = nb_epochs, 
                                 verbose          = 1 if debug else 2,
                                 validation_data  = valid_gen,
                                 validation_steps = steps_per_epoch // 5 + 1,
                                 callbacks        = [early_stop, checkpoint, csv_logger])      
        

    def precompute_features(self, imgs, batch_size):
        imgs = self.backend_class.preprocess_imgs(imgs)
        features = self.backend_model.predict(imgs, batch_size)
        return features
