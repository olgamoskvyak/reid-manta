import os
from keras.optimizers import Adam

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from base_model import BaseModel
from utils.utils import plot_model_loss_acc_csv

class Classification(BaseModel):
    def __init__(self, backend, input_shape, frontend, embedding_size, 
                 connect_layer=-1, train_from_layer = 0, distance = 'l2', loss_func = 'semi_hard_triplet',
                 weights='imagenet', show_summary=True):
        self.loss_func = loss_func
        super(Classification, self).__init__(input_shape, backend, frontend, embedding_size, 
                                 connect_layer, train_from_layer, distance, weights)
        self.model = self.top_model   
        if show_summary:
            print('Triplet Loss Model summary:')
            self.model.summary()

            
    def compile_model(self, learning_rate):
        '''Compile the model'''
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        if self.loss_func == 'categorical_crossentropy':
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        else:
            raise Exception('Only categorical_crossentropy is supported')
            
    def plot_history(self, file, from_epoch=0, showFig = True, saveFig = False, figName = 'plot.png'):
        plot_model_loss_acc_csv(file, from_epoch=from_epoch, showFig=showFig, saveFig=saveFig, figName=figName)