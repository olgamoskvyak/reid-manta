import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.optimizers import Adam

from scipy.special import comb

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from base_model import BaseModel
from utils.tensorflow_losses import triplet_semihard_loss, lifted_struct_loss
from utils.utils import plot_model_loss_csv

from evaluation.metrics import distance


class TripletLoss(BaseModel):
    def __init__(self, backend, input_shape, frontend, embedding_size, 
                 connect_layer=-1, train_from_layer = 0, distance = 'l2', loss_func = 'semi_hard_triplet',
                 weights='imagenet', show_summary=False):
        self.loss_func = loss_func
        super(TripletLoss, self).__init__(input_shape, backend, frontend, embedding_size, 
                                 connect_layer, train_from_layer, distance, weights)
        self.model = self.top_model
        if show_summary:          
            print('Triplet Loss Model summary:')
            self.model.summary()
        
        
    def compile_model(self, learning_rate):
        '''Compile the model'''
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        if self.loss_func == 'semi_hard_triplet':
            self.model.compile(loss=triplet_semihard_loss, optimizer=optimizer)
        elif self.loss_func == 'lifted_struct_loss':
            self.model.compile(loss=lifted_struct_loss, optimizer=optimizer)
        else:
            raise Exception('{} loss is not supported'.format(self.loss_func))
        
    def plot_history(self, file, from_epoch=0, showFig = True, saveFig = False, figName = 'plot.png'):
        plot_model_loss_csv(file, from_epoch, showFig, saveFig, figName)
        
    
    def compute_dist(self, images, labels, sample_size = None):
        """Compute distances for pairs
        
        sample_size: None or integer, number of pairs as all pairs can be large number. 
                        If None, by default all possible pairs are considered.
                        
        Returns:
        dist_embed:  array of distances
        actual_issame: array of booleans, True if positive pair, False if negative pair
        """
        print('Generating pairs and computing embeddings...')

        #Define generator to loop over data
        gen = ImageDataGenerator(data_format=K.image_data_format(),
                                 preprocessing_function=self.backend_class.normalize)
        features_shape = self.model.get_output_shape_at(0)[1:]
        
        if sample_size is None:
            n_pairs = comb(images.shape[0], 2, exact=True)
        else:
            n_pairs = int(sample_size)
                                 
        embeddings = np.zeros(shape = (2, n_pairs,) + features_shape)
        actual_issame = np.full(shape = (n_pairs,), fill_value=True, dtype = np.bool)

        #Create all possible combinations of images (no repeats)
        idx = 0      
        for i in range(images.shape[0]):
            for j in range(i):
                img_1 = gen.random_transform(images[i].astype(K.floatx()))
                img_1 = gen.preprocessing_function(img_1)

                img_2 = gen.random_transform(images[j].astype(K.floatx()))
                img_2 = gen.preprocessing_function(img_2)

                embeddings[0, idx] = self.model.predict_on_batch(np.expand_dims(img_1, 0))
                embeddings[1, idx] = self.model.predict_on_batch(np.expand_dims(img_2, 0))

                if labels[i] != labels[j]:
                    actual_issame[idx] = False
                idx += 1
                                 
                if idx >= n_pairs:
                     break
                        
            if idx >= n_pairs:
                     break

        print('Number of pairs in evaluation {}, number of positive {}'.format(len(actual_issame), np.sum(actual_issame)))
        dist_emb = distance(embeddings[0], embeddings[1], distance_metric=0)
        return dist_emb, actual_issame