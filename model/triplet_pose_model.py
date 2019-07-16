import numpy as np
from keras.optimizers import Adam
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator

from model.base_model import BaseModel
from utils.tensorflow_losses import triplet_semihard_loss
from utils.custom_losses import triplet_pose_loss, semihard_pose_loss, triplet_loss_mult, pose_variance
from utils.utils import plot_model_loss_csv
from utils.utils import make_batches
from scipy.special import comb
from evaluation.metrics import distance


class TripletLossPoseInv(BaseModel):
    def __init__(self, backend, input_shape, frontend, embedding_size, 
                 connect_layer=-1, train_from_layer = 0, distance = 'l2', loss_func = 'semi_hard_triplet',
                 weights='imagenet', n_poses = 5, show_summary=True, bs=40):
        self.loss_func = loss_func
        self.n_poses = n_poses
        self.bs = bs
        
        #Define main branch
        super(TripletLossPoseInv, self).__init__(input_shape, backend, frontend, embedding_size, 
                                 connect_layer, train_from_layer, distance, weights)
        
        self.model = self.top_model  
            
        if show_summary:
            print('TripletPose Loss Model summary:')
            self.model.summary() 
            
    def compile_model(self, lr, margin=0.5, weights=[1., 1.], loss_func=None):
        '''Compile the model'''
        loss_func = loss_func or self.loss_func
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        if loss_func == 'triplet_loss_mult':
            self.model.compile(loss=lambda y_true,y_pred: triplet_loss_mult(y_true, y_pred, margin, n_poses = self.n_poses, n_imgs = self.bs), optimizer=optimizer)
            print('Model is compiled with triplet_loss_mult')
            
        elif loss_func == 'pose_variance':
            #This loss is mostly used to calculate the weights
            self.model.compile(loss=lambda y_true,y_pred: pose_variance(y_true, y_pred, n_poses = self.n_poses, n_imgs = self.bs), optimizer=optimizer)
            print('Model is compiled with pose_variance')

        elif loss_func == 'triplet_pose_loss':
            self.model.compile(loss = lambda y_true,y_pred: triplet_pose_loss(y_true, y_pred, margin, weights, n_poses = self.n_poses, n_imgs = self.bs), optimizer=optimizer)
            print('Model is compiled with triplet_pose_loss = triplet_loss_mult + pose_invariant loss, weights {}'.format(weights))
                
        elif loss_func == 'semi_hard_triplet':
            self.model.compile(loss=triplet_semihard_loss, optimizer=optimizer)
            print('Model is compiled with triplet semi-hard loss')
            
        elif loss_func == 'semihard_pose_loss':
            self.model.compile(loss=lambda y_true,y_pred: semihard_pose_loss(y_true, y_pred, margin, weights, n_poses = self.n_poses, n_imgs = self.bs), optimizer=optimizer)
            print('Model is compiled with semihard_pose_loss: semihard over the whole batch and pose invariant term')
        else:
            raise Exception('Only semi_hard_triplet, triplet_loss_mult and triplet_pose_loss are supported')
        
            
    def plot_history(self, file, from_epoch=0, showFig = True, saveFig = False, figName = 'plot.png'):
        plot_model_loss_csv(file, from_epoch, showFig, saveFig, figName)
         
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
    
    def compute_dist(self, images, labels, sample_size = None):
        """Compute distances for pairs
        
        sample_size: None or integer, number of pairs as all pairs can be large number. 
                        If None, by default all possible pairs are considered.
                        
        Returns:
        dist_embed:  array of distances
        actual_issame: array of booleans, True if positive pair, False if negative pair
        """
        # Run forward pass to calculate embeddings
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
    
            