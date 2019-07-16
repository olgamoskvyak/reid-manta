import os, random
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from skimage import transform
from keras_preprocessing.image.affine_transformations import apply_affine_transform

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils.utils import rgb2gray


class BatchGenerator:
    """Create a generator thay yelds batches of images B = P * K where P - number of persons (unique class), 
    K - number of examples per class. Classes are selected randomly and can be repeated from batch to batch.
    """
    def __init__(self, images, classes, aug_gen=None, p=20, k=3, seed=None, equal_k=True, to_gray=False, n_poses=1,
                dupl_labels=False, rotate_poses=False, flatten_batch= True, perspective=False):
        self.seed = seed
        self.aug_gen = aug_gen
        self.equal_k =equal_k
        self.to_gray = to_gray
        self.n_poses = n_poses
        self.dupl_labels = dupl_labels
        self.rotate_poses = rotate_poses
        self.flatten_batch = flatten_batch
        self.perspective = perspective
        
        if self.dupl_labels:
            print('Duplicating labels for network branches')
      
        self.total_samples_seen = 0
        self.unique_classes, counts = np.unique(classes, return_counts=True)
        min_samples_per_class = min(counts)
        print('Number of unique classes {}, min images per class {}'.format(counts.shape[0], min_samples_per_class))
        
        if equal_k and min_samples_per_class < k:
            self.k = min_samples_per_class
            print('Number of samples per class is reduced to {}'.format(self.k))
        else:
            self.k = k            
        
        self.img_set = images
        self.class_set = classes
        
        if p>self.unique_classes.shape[0]:
            self.p = self.unique_classes.shape[0]
            print('Number of unique classes per batch is reduced to {}'.format(self.p))
        else:
            self.p = p
    
    def __iter__(self):
        return self
    
    def set_seed(self):
        if self.seed is not None:
            local_seed= self.seed + self.total_samples_seen
        else:
            local_seed = self.total_samples_seen 
        np.random.seed(local_seed)
        return local_seed
    
    def _get_batches_of_transformed_samples(self):
        #set seed
        self.set_seed()
        
        #initialise empty arrays for batches of images and classes
        batch_img = np.zeros(shape=(self.p*self.k, self.n_poses)+self.img_set.shape[1:], dtype='float32')
        batch_class = np.empty(shape=(self.p*self.k, ), dtype=self.class_set.dtype)
        
        #selected classes - random choice from an array of unique
        sel_classes = np.random.choice(self.unique_classes, self.p, replace=False)
        
        #select images
        for i, sel_class in enumerate(sel_classes):
            start_idx = i*self.k
            end_idx = (i+1)*self.k
            
            num_img_sel_class = self.img_set[self.class_set==sel_class].shape[0]
            if self.equal_k:
                sel_idx = np.random.choice(num_img_sel_class, self.k, replace=False)
            else:
                sel_idx = np.random.choice(num_img_sel_class, self.k, replace=True)
            
            for pose in range(self.n_poses):
                batch_img[start_idx:end_idx, pose] = self.img_set[self.class_set==sel_class][sel_idx]
            
            batch_class[start_idx:end_idx] = self.class_set[self.class_set==sel_class][sel_idx]
            
            self.total_samples_seen = self.total_samples_seen + 1
            self.set_seed()
            
        #print(batch_img[0,0])
        if self.perspective:
            #Apply one perspective transform and then rotate the transformed image
            angle_step = 360 // self.n_poses
            #augment images if generator is defined
            for j in range(batch_img.shape[0]):
                if self.aug_gen is not None:
                    temp = self.aug_gen.random_transform(batch_img[j,0], seed=None)
                else:
                    temp = batch_img[j,0]
                for pose in range(batch_img.shape[1]):
                    projected = projective_transformation(temp.astype('uint8'), var=0.15)
                    angle = int(random.gauss(angle_step * pose, 10))
                    batch_img[j,pose] = apply_affine_transform(projected, theta=angle)
                    batch_img[j,pose] = self.aug_gen.preprocessing_function(batch_img[j,pose] * 255) 
            
        else:
            rot_angle = 360 // self.n_poses
            #augment images if generator is defined
            if self.aug_gen is not None:
                for j in range(batch_img.shape[0]):
                    if self.rotate_poses:
                        temp = self.aug_gen.random_transform(batch_img[j,0], seed=None)
                        for pose in range(batch_img.shape[1]):
                            batch_img[j,pose] = apply_affine_transform(temp, theta=rot_angle * pose)
                            batch_img[j,pose] = self.aug_gen.preprocessing_function(batch_img[j,pose])
                    else:
                        for pose in range(batch_img.shape[1]):
                            #In half cases convert to grayscale
                            if self.to_gray:
                                if np.random.random()>0.5:
                                    batch_img[j,pose] = rgb2gray(batch_img[j,pose], 'float32')
                            temp = self.aug_gen.random_transform(batch_img[j,pose], seed=None)
                            batch_img[j,pose] = self.aug_gen.preprocessing_function(temp)
        
        if self.dupl_labels:
            return [batch_img[:,pose] for pose in range(batch_img.shape[1])], [batch_class] * self.n_poses
        elif self.flatten_batch:
            #Batch size = (#img, #poses, h,w,ch)
            total_images = batch_img.shape[0]*batch_img.shape[1]
            new_shape = (total_images,) + batch_img.shape[2:]
            tiled_classes = np.reshape(np.array([batch_class]*self.n_poses), (total_images,), 'C')
            return np.reshape(batch_img, new_shape, order='F'), tiled_classes
        else:
            return np.squeeze(batch_img), batch_class
    
    def __next__(self):
        return self._get_batches_of_transformed_samples()
    


def randomProjection(variation, image_size, random_seed = None):
    '''Generate geometrical projection by defining transformation of 4 points
       ------
       Input:
       variation:    percentage (in decimal notation from 0 to 1)
                     relative size of a circle region where centre is projected
                     
       image_size:   integer
                     size of image in pixels
       random_seed:  integer
                     initialize internal state of the random number generator      
       ------
       Return:
       tform:        object from skimage.transromf
    
    '''
    d = image_size * variation
    
    if random_seed is not None:
        random.seed(random_seed)
    
    top_left =    (random.uniform(-0.5*d, d), random.uniform(-0.5*d, d))  # Top left corner
    bottom_left = (random.uniform(-0.5*d, d), random.uniform(-0.5*d, d))   # Bottom left corner
    top_right =   (random.uniform(-0.5*d, d), random.uniform(-0.5*d, d))     # Top right corner
    bottom_right =(random.uniform(-0.5*d, d), random.uniform(-0.5*d, d))  # Bottom right corner

    tform = transform.ProjectiveTransform()
    tform.estimate(np.array((
            top_left,
            (bottom_left[0], image_size - bottom_left[1]),
            (image_size - bottom_right[0], image_size - bottom_right[1]),
            (image_size - top_right[0], top_right[1])
        )), np.array((
            (0, 0),
            (0, image_size),
            (image_size, image_size),
            (image_size, 0)
        )))       

    return tform

def projective_transformation(img, var = 0.15, random_seed = None):
    """Additional preprocessing function for data augmentation: random projective transformations over input image.
    Input:
    img: 3D tensor, image (integers [0,255])
    random_seed: integer
    Returns:
    img_transformed: 3D tensor, image (dtype float64, [0,1])
    """
    projection = randomProjection(var, min(img.shape[0], img.shape[1]), random_seed = random_seed)
                                  
    img_transformed = transform.warp(img, projection, mode='edge')
    
    return img_transformed
       
    
class PairsImageDataGenerator(ImageDataGenerator):
    """Generate minibatches of image PAIRS data with real-time data augmentation.
    """
    
    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        """Generate minibatches of image pairs from a numpy array
        """
        return PairsNumpyArrayIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)
    
class PairsNumpyArrayIterator(NumpyArrayIterator):
    """Iterator yielding pairs of images from a Numpy array"""
    
    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        #self.y = y
        self.seed = seed
        self.classes = np.unique(y) 
        
        super(PairsNumpyArrayIterator, self).__init__(x, y, image_data_generator,
                 batch_size=batch_size, shuffle=shuffle, seed=seed,
                 data_format=data_format,
                 save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)
    
    def _get_class_indices(self):
        """Indices of same class elements in X
        ______________________________________
        Example: 
        y = [2,2,0,1,0]
        classes = [0,1,2]
        num_classes = 3
        class_indices = { 0: [2,4], 1: [3], 2: [0,1]]       
        """
        class_indices = {cl: np.where(self.y == cl)[0] for cl in self.classes}
        return class_indices
        
    
    def _get_batches_of_transformed_samples(self, index_array, return_classes = False):
        batch_pairs = []
        batch_pair_labels = []
        batch_classes = []
        class_indices = self._get_class_indices()
        
        #set seed
        if self.seed is not None:
            local_seed= self.seed + self.total_batches_seen
        else:
            local_seed = self.total_batches_seen
        
        random.seed(local_seed)
        
        for step, idx in enumerate(index_array):
            #get an anchor image
            x_a = self.x[idx]
            x_a = self.image_data_generator.random_transform(x_a)
            x_a = self.image_data_generator.preprocessing_function(x_a)

            #Half positive and half negative pairs are generated for a batch
            same_class = int(self.y[idx])     
                
            #get different class
            while True:
                diff_class = random.choice(self.classes)
                if same_class != diff_class:
                    break
            
            neg_pair_index = random.choice(class_indices[diff_class])
            
            while True:
                pos_pair_index = random.choice(class_indices[same_class])
                if pos_pair_index != idx:
                    break
            
            x_n = self.x[neg_pair_index]
            #augment pair image
            #TODO do I need to cast to float? x2.astype(K.floatx())
            x_n = self.image_data_generator.random_transform(x_n)
            x_n = self.image_data_generator.preprocessing_function(x_n)
            
            x_p = self.x[pos_pair_index]
            x_p = self.image_data_generator.random_transform(x_p)
            x_p = self.image_data_generator.preprocessing_function(x_p)
            
            batch_pairs += [[x_a, x_p]]
            batch_pairs += [[x_a, x_n]]
            batch_classes += [[same_class, same_class]]
            batch_classes += [[same_class, diff_class]]
            batch_pair_labels += [0 , 1]
        
        #Shuffle pairs and labels in unison
        if self.shuffle == True:
            batch_pairs, batch_pair_labels, batch_classes = shuffle(batch_pairs, batch_pair_labels, batch_classes, random_state=local_seed)
        
        if return_classes:
            return ([np.array(batch_pairs, dtype='float32')[:,0], np.array(batch_pairs, dtype='float32')[:,1]], np.array(batch_pair_labels), np.array(batch_classes))
        else:
            return ([np.array(batch_pairs, dtype='float32')[:,0], np.array(batch_pairs, dtype='float32')[:,1]], np.array(batch_pair_labels))
    
    def __next__(self):
        """Returns
            The next batch with classes.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        #print(index_array)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        #class_indices = self._get_class_indices()
        return self._get_batches_of_transformed_samples(index_array, return_classes = True)