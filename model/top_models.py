from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Dense, Lambda, GlobalAveragePooling2D, Dropout
from keras import regularizers
import keras.backend as K


def glob_pool_norm(embedding_size, backend_model = None, features_shape = None):
    """Global Pooling of feature maps with normalisation layer."""
    if backend_model is not None:
        input_layer = backend_model.output
    elif features_shape is not None:
        input_layer = Input(shape=features_shape)
    else:
        raise ValueError('Provide a base model or a shape of features.')

    x = GlobalAveragePooling2D(name='top_global_pool')(input_layer)

    dense_layer = Dense(embedding_size, 
                        name='top_dense_layer', 
                        kernel_regularizer=regularizers.l2(0.01), 
                        bias_regularizer=regularizers.l2(0.01))(x)

    norm_layer = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='top_norm_layer')(dense_layer)

    if backend_model is not None:
        return Model(backend_model.input, norm_layer, name='top_model')  
    else:
        return Model(input_layer, norm_layer, name='top_model')    

def glob_pool(embedding_size, backend_model = None, features_shape = None):
    """Global Pooling of feature maps without normalisation."""
    if backend_model is not None:
        input_layer = backend_model.output
    elif features_shape is not None:
        input_layer = Input(shape=features_shape)
    else:
        raise ValueError('Provide a base model or a shape of features.')

    x = GlobalAveragePooling2D(name='top_global_pool')(input_layer)

    dense_layer = Dense(embedding_size, 
                        name='top_dense_layer', 
                        kernel_regularizer=regularizers.l2(0.01), 
                        bias_regularizer=regularizers.l2(0.01))(x)

    if backend_model is not None:
        return Model(backend_model.input, dense_layer, name='top_model')  
    else:
        return Model(input_layer, dense_layer, name='top_model')
    