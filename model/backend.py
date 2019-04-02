from keras.models import Model
import numpy as np
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications import InceptionResNetV2, DenseNet121, InceptionV3


class BaseFeatureExtractor(object):
    """CNN pretrained model for extracting features. Imported directly from Keras"""

    # to be defined in each subclass
    def __init__(self, input_shape, weights):
        raise NotImplementedError("error message")

    # to be defined in each subclass
    def normalize(self, image):
        raise NotImplementedError("error message")       

    def get_output_shape(self):
        return self.feature_extractor.get_output_shape_at(-1)[1:3]

    def extract(self, input_image):
        return self.feature_extractor(input_image)
    
class DummyNetFeature(BaseFeatureExtractor):
    """Small CNN to test on local machine"""
    def __init__(self, input_shape, weights):
        input_image = Input(shape=input_shape)
        
        # Layer 1
        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = MaxPooling2D(pool_size=(4, 4))(x)
        
        # Layer 2
        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = MaxPooling2D(pool_size=(4, 4))(x)
        
        output = MaxPooling2D(pool_size=(4, 4))(x)

        self.feature_extractor = Model(input_image, output)  

    def normalize(self, image):
        image = image / 255.
        return image


class InceptionV3Feature(BaseFeatureExtractor):
    """Inception V3 model pretrained on Imagenet"""
    def __init__(self, input_shape, weights):
       
        self.feature_extractor = InceptionV3(input_shape=input_shape, include_top=False, weights=weights)

    def normalize(self, image):
        image = image.astype('uint8')
        image = image / 255.
        image = image - 0.5
        image = image * 2.

        return image
    
class InceptionResNetV2Feature(BaseFeatureExtractor):
    """Inception V3 model pretrained on Imagenet"""
    def __init__(self, input_shape, weights):
        
        self.feature_extractor = InceptionResNetV2(input_shape=input_shape, include_top=False, weights=weights)

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.

        return image

class MobileNetV2Feature(BaseFeatureExtractor):
    """Inception V3 model pretrained on Imagenet"""
    def __init__(self, input_shape, weights):
        self.feature_extractor = MobileNetV2(input_shape=input_shape, include_top=False, weights=weights)
        
    def normalize(self, image):
        image = image / 128.
        image = image - 1.
        return image.astype(np.float32)

class VGG16Feature(BaseFeatureExtractor):
    """VGG16 Model pretrained on Imagenet"""
    def __init__(self, input_shape, weights):
        self.feature_extractor = VGG16(input_shape=input_shape, include_top=False, weights=weights)

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image

class ResNet50Feature(BaseFeatureExtractor):
    """ResNet50 model pretrained on Imagenet with the last pooling layer removed"""
    def __init__(self, input_shape, weights):
        self.feature_extractor = ResNet50(input_shape=input_shape, include_top=False, weights=weights)
        
    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image
    
    def preprocess_imgs(self, imgs):
        return self.normalize(imgs)