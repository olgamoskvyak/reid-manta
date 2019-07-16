from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, Dense, Lambda, GlobalAveragePooling2D
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

def glob_softmax(embedding_size, backend_model = None, features_shape = None):
    """Global Pooling of feature maps without normalisation."""
    if backend_model is not None:
        input_layer = backend_model.output
    elif features_shape is not None:
        input_layer = Input(shape=features_shape)
    else:
        raise ValueError('Provide a base model or a shape of features.')

    output_size = embedding_size
    x = GlobalAveragePooling2D(name='top_global_pool')(input_layer)
  
    output_layer = Dense(output_size, activation = 'softmax',
                        name='top_dense_layer_2', 
                        kernel_regularizer=regularizers.l2(0.01), 
                        bias_regularizer=regularizers.l2(0.01))(x)
    
    

    if backend_model is not None:
        return Model(backend_model.input, output_layer, name='top_model')  
    else:
        return Model(input_layer, output_layer, name='top_model')
    

def conv_norm(embedding_size, backend_model = None):
    """Max pool, conv, glob pooling and normalisation layers on top of features. """
    input_layer  = backend_model.output

    x = MaxPooling2D((2, 2), strides=(2, 2), name='top_pool')(input_layer)
    x = Conv2D(embedding_size, (3, 3), activation='relu', padding='same', name='top_conv1')(x)
    x = BatchNormalization(name='top_batchnorm1')(x)
    x = GlobalAveragePooling2D(name='top_global_pool')(x)

    dense_layer = Dense(embedding_size, 
                        name='top_dense_layer', 
                        kernel_regularizer=regularizers.l2(0.01), 
                        bias_regularizer=regularizers.l2(0.01))(x)

    norm_layer = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='top_norm_layer')(dense_layer)
    
    model = Model(backend_model.input, norm_layer, name="top_model")
    return model   
    
def vgg16_block_5(features_shape, embedding_size, vgg_weights = None):
    """Last conv block (block 5) of VGG.
    Loads pretrained weights if available"""
    input_image     = Input(shape=features_shape)

    # finetuning of Block 5 VGG
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(input_image)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = BatchNormalization(name='batch_norm_1')(x)
    x = GlobalAveragePooling2D(name='global_pooling')(x)

    dense_layer = Dense(embedding_size, 
                        name='dense_layer', 
                        kernel_regularizer=regularizers.l2(0.01), 
                        bias_regularizer=regularizers.l2(0.01))(x)

    norm_layer = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)
    
    model = Model(input_image, norm_layer)
    if vgg_weights is not None:
        model.load_weights(vgg_weights, by_name=True)
        print('Top model is initialised with VGG pretrained weights')

    return model       
