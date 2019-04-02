import random
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator



class BatchGenerator:
    """Create a generator thay yelds batches of images B = P * K where P - number of persons (unique class), 
    K - number of examples per class. Classes are selected randomly and can be repeated from batch to batch.
    """
    def __init__(self, images, classes, aug_gen=None, p=20, k=3, seed=None):
        self.seed = seed
        self.aug_gen = aug_gen
   
        self.total_samples_seen = 0
        self.unique_classes, counts = np.unique(classes, return_counts=True)
        min_samples_per_class = min(counts)
        print('Number of unique classes {}, min images per class {}'.format(counts.shape[0], min_samples_per_class))
        
        if min_samples_per_class < k:
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
        batch_img = np.zeros(shape=(self.p*self.k, )+self.img_set.shape[1:], dtype='float32') #self.img_set.dtype)
        batch_class = np.empty(shape=(self.p*self.k, ), dtype=self.class_set.dtype)
        
        #selected classes - random choice from an array of unique
        sel_classes = np.random.choice(self.unique_classes, self.p, replace=False)
        
        for i, sel_class in enumerate(sel_classes):
            start_idx = i*self.k
            end_idx = (i+1)*self.k
            
            num_img_sel_class = self.img_set[self.class_set==sel_class].shape[0]
            sel_idx = np.random.choice(num_img_sel_class, self.k, replace=False)
            
            batch_img[start_idx:end_idx] = self.img_set[self.class_set==sel_class][sel_idx]
            batch_class[start_idx:end_idx] = self.class_set[self.class_set==sel_class][sel_idx]
            
            self.total_samples_seen = self.total_samples_seen + 1
            self.set_seed()
            
        #augment images if generator is defined
        #TODO check seed for random transform
        if self.aug_gen is not None:
            for j in range(batch_img.shape[0]):
                #astype(K.floatx()) - is simply casting to float without rescaling
                temp = self.aug_gen.random_transform(batch_img[j], seed=None)
                batch_img[j] = self.aug_gen.preprocessing_function(temp)
          
        return batch_img, batch_class
    
    def __next__(self):
        return self._get_batches_of_transformed_samples()
    
    
#NOT USED
'''class BatchGeneratorAug:
    def __init__(self, images, classes, image_data_generator=None, p=40, k=2, seed=None):
        self.seed = seed
        self.image_data_generator = image_data_generator
        self.total_samples_seen = 0
        self.unique_classes, counts = np.unique(classes, return_counts=True)
        min_samples_per_class = min(counts) 
        print('Number of unique classes {}, min images per class {}'.format(counts.shape[0], min_samples_per_class))

        self.k = k
        self.img_set = images
        self.class_set = classes
        self.img_copies = self.img_set.shape[0]
        
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
    
    def _get_batches_of_transformed_samples(self):
        #set seed
        self.set_seed()
        
        #initialise empty arrays for batches of images and classes
        batch_img = np.zeros(shape=(self.p*self.k, )+self.img_set.shape[2:], dtype=self.img_set.dtype)
        batch_class = np.empty(shape=(self.p*self.k, ), dtype=self.class_set.dtype)
        
        #selected classes - random choice from an array of unique
        sel_classes = np.random.choice(self.unique_classes, self.p, replace=False)
        
        for i, sel_class in enumerate(sel_classes):
            start_idx = i*self.k
            end_idx = (i+1)*self.k
            
            sel_class_subset = self.img_set[:, self.class_set==sel_class]
            sel_idx = np.random.choice(sel_class_subset.shape[1], self.k, replace=False)
            sel_copy = np.random.choice(self.img_copies, self.k, replace=True)
            #print(sel_copy)
            
            #batch_img[start_idx:end_idx] = self.img_set[self.class_set==sel_class][sel_copy, sel_idx]
            for i in range(self.k):
                batch_img[start_idx+i] = sel_class_subset[sel_copy[i], sel_idx[i]]
                
            batch_class[start_idx:end_idx] = self.class_set[self.class_set==sel_class][sel_idx]
            
            self.total_samples_seen = self.total_samples_seen + 1
            self.set_seed()
          
        #augment images if generator is defined
        if self.image_data_generator is not None:
            for j in range(batch_img.shape[0]):
                batch_img[j] = self.image_data_generator.random_transform(batch_img[j].astype(K.floatx()))
                batch_img[j] = self.image_data_generator.preprocessing_function(batch_img[j])
        
        # Convert labels to categorical one-hot encoding
        #batch_class_cat = np.array([np.where(self.unique_classes == cl)[0][0] for cl in batch_class])
        #oh_batch_class = keras.utils.to_categorical(batch_class_cat, len(self.unique_classes))

        return batch_img, batch_class
    
    def __next__(self):
        return self._get_batches_of_transformed_samples()
        '''
    
    
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