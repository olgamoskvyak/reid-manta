import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda
from keras.preprocessing.image import ImageDataGenerator
from base_model import BaseModel
import keras.backend as K
from keras.optimizers import Adam
from scipy.special import comb

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from evaluation.metrics import contrastive_loss
from utils.utils import make_batches, plot_model_loss_acc_csv
    
    
class Siamese(BaseModel):
    def __init__(self, backend, input_shape, frontend, embedding_size, 
                 connect_layer=-1, train_from_layer = 0, distance = 'l2', loss_func = 'contrastive', show_summary=True, weights='imagenet'):
        
        self.loss_func = loss_func
        super(Siamese, self).__init__(input_shape, backend, frontend, embedding_size, 
                                 connect_layer, train_from_layer, distance, weights)
        self.model = self.siamese_block()
        if show_summary:
            print('Siamese model summary:')
            self.model.summary()
            print('Base model summary:')
            self.top_model.summary()
        
    def siamese_block(self):
        '''Siamese block. Connects two copies of a shared model. Currently uses only L2 distance function.'''
        print('Constructing Siamese model...')
        image_a = Input(shape=self.input_shape, name='input_a')
        image_b = Input(shape=self.input_shape, name='input_b')

        out_a = self.top_model(image_a)
        out_b = self.top_model(image_b)
        #Define distance function
        out = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([out_a, out_b])

        return Model([image_a, image_b], out, name='out_siamese')
    

    def compile_model(self, learning_rate, loss_func=None):
        '''Compile the model'''
        loss_func = loss_func or self.loss_func
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        if self.loss_func == 'binary_crossentropy':
            self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        elif self.loss_func == 'contrastive':
            self.model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=['accuracy'])
        else:
            raise Exception('Only binary_crossentropy or contrastive losses are supported for Siamese network')
            
    def plot_history(self, file, from_epoch=0, showFig = True, saveFig = False, figName = 'plot.png'):
        plot_model_loss_acc_csv(file, from_epoch, showFig, saveFig, figName)
        
    def preproc_predict(self, imgs, batch_size=32):
        """Preprocess images and predict with the model (no batch processing for first step)
        Input:
        imgs: 4D float or int array of images
        batch_size: integer, size of the batch
        Returns:
        predictions: numpy array with predictions (num_images, len_model_output)
        """
        batch_idx = make_batches(imgs.shape[0], batch_size)
        imgs_preds = np.zeros((imgs.shape[0],)+self.top_model.get_output_shape_at(0)[1:])
        print('Computing predictions with the shape {}'.format(imgs_preds.shape))

        for sid, eid in batch_idx:
            preproc = self.backend_class.normalize(imgs[sid:eid])
            imgs_preds[sid:eid] = self.top_model.predict_on_batch(preproc)   

        return imgs_preds
            

    def compute_dist(self, images, labels, sample_size = None):
        """Compute distances for pairs
        
        sample_size: None or integer, number of pairs as all pairs can be large number. 
                        If None, by default all possible pairs are considered.
        """
        # Run forward pass to calculate embeddings
        print('Generating pairs and computing embeddings...')

        #Define generator to loop over data
        gen = ImageDataGenerator(data_format=K.image_data_format(),
                                    preprocessing_function=self.backend_class.normalize)
        
        if sample_size is None:
            n_pairs = comb(images.shape[0], 2, exact=True)
        else:
            n_pairs = int(sample_size)
        print('Computing distances for {} pairs...'.format(n_pairs))
                                 
        distances = np.zeros(shape = (n_pairs,))
        actual_issame = np.full(shape = (n_pairs,), fill_value=True, dtype = np.bool)

        #Create all possible combinations of images (no repeats)
        idx = 0      
        for i in range(images.shape[0]):
            for j in range(i):
                #Preprocess each image
                img_1 = gen.random_transform(images[i].astype(K.floatx()))
                img_1 = gen.preprocessing_function(img_1)

                img_2 = gen.random_transform(images[j].astype(K.floatx()))
                img_2 = gen.preprocessing_function(img_2)
                        
                #Predict distance using Siamese model
                distances[idx] = self.model.predict_on_batch([np.expand_dims(img_1, 0), np.expand_dims(img_2, 0)])

                #Change flag to False for negative pairs
                if labels[i] != labels[j]:
                    actual_issame[idx] = False
                
                if idx>0 and idx%100==0:
                    print('Computed distances for {} pairs'.format(idx))
                    
                #print('Distance: {:.2f} actual: {}. labels {}, {}'.format(distances[idx], actual_issame[idx],
                #                                                          labels[i], labels[j]))
                
                idx += 1
                if idx >= n_pairs:
                    break
                    
            if idx >= n_pairs:
                break


        print('Number of pairs in evaluation {}, number of positive {}'.format(len(actual_issame), np.sum(actual_issame)))
        return distances, actual_issame

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)        
