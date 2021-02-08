from functools import reduce
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import models
from tensorflow.keras.layers import Input

from core.backbone.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from core.backbone.efficientnet import EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from utils.anchors import anchors_for_shape
from core.utils_layers import wBiFPNAdd, SeparableConvBlock, ConvBlock, RegressBoxes, ClipBoxes, FilterDetections
from core.utils import PriorProbability
phi2config = [
            { 'r_input': 512,  'backbone': EfficientNetB0(input_tensor = Input((512, 512, 3))),   'w_bifpn': 64,  'd_bifpn': 3, 'd_class': 3},  #phi = 0
            { 'r_input': 640,  'backbone': EfficientNetB1(input_tensor = Input((640, 640, 3))),   'w_bifpn': 88,  'd_bifpn': 4, 'd_class': 3},  #phi = 1
            { 'r_input': 768,  'backbone': EfficientNetB2(input_tensor = Input((768, 768, 3))),   'w_bifpn': 112, 'd_bifpn': 5, 'd_class': 3},  #phi = 2
            { 'r_input': 896,  'backbone': EfficientNetB3(input_tensor = Input((896, 896, 3))),   'w_bifpn': 160, 'd_bifpn': 6, 'd_class': 4},  #phi = 3
            { 'r_input': 1024, 'backbone': EfficientNetB4(input_tensor = Input((1024, 1024, 3))), 'w_bifpn': 224, 'd_bifpn': 7, 'd_class': 4},  #phi = 4
            { 'r_input': 1280, 'backbone': EfficientNetB5(input_tensor = Input((1280, 1280, 3))), 'w_bifpn': 288, 'd_bifpn': 7, 'd_class': 4},  #phi = 5
            { 'r_input': 1280, 'backbone': EfficientNetB6(input_tensor = Input((1280, 1280, 3))), 'w_bifpn': 384, 'd_bifpn': 8, 'd_class': 5},  #phi = 6
            ]

MOMENTUM = 0.997
EPSILON = 1e-4


class EfficientDet(models.Model):
    def __init__(self, phi, num_classes, num_anchors = 9, score_threshold = 0.01, anchor_parameters = None, train = True):
        super(EfficientDet, self).__init__()

        self.train = train
        config = phi2config[phi]
        
        r_input = config['r_input']
        w_bifpn = config['w_bifpn']
        d_bifpn = config['d_bifpn']
        w_head = w_bifpn
        d_class = config['d_class']

        self.img_shape = (r_input, r_input, 3)

        self.backbone = config['backbone']
        
        self.bifpn_list = []
        for i in range(d_bifpn):
            if i == 0:
                self.bifpn_list.append(BiFPN(num_channels = w_bifpn ,first = True))
            else:
                self.bifpn_list.append(BiFPN(num_channels = w_bifpn ,first = False))
        
        self.level = 0

        self.class_net = ClassNet(w_head, d_class,  name = 'classnet', num_classes = num_classes, num_anchors = num_anchors)
        self.box_net = BoxNet(w_head, d_class, name = 'boxnet', num_anchors = num_anchors)

        self.class_out_concat = layers.Concatenate(axis = 1, name='class_concat')
        self.box_out_concat = layers.Concatenate(axis = 1, name='box_concat')

        self.anchors = anchors_for_shape((r_input, r_input), anchor_params = anchor_parameters)
        self.anchors_input = np.expand_dims(self.anchors, axis=0)
        
        self.regressor_boxes = RegressBoxes(name = 'boxes')
        self.clip_boxes = ClipBoxes(name = 'clipped_boxes')

        self.filter_detector = FilterDetections(name = 'filtered_detections', score_threshold = score_threshold)
        
    def call(self, inputs):
        out = self.backbone(inputs)

        for bifpn in self.bifpn_list:
            out = bifpn(out)

        class_out = [self.class_net(feature, self.level) for feature in out]
        class_out = self.class_out_concat(class_out)    

        box_out = [self.box_net(feature, self.level) for feature in out]
        box_out = self.box_out_concat(box_out)

        out_dict= {'classification': class_out, 'regression': box_out}

        if self.train == False:
            delta_boxes = self.regressor_boxes([self.anchors_input, box_out[..., :4]])
            delta_boxes = self.clip_boxes([self.img_shape, delta_boxes])

            detections = self.filter_detector([delta_boxes, class_out])
            
            return class_out, box_out, detections
        else:
            return out_dict
        
class EfficientDet_P(models.Model):
    def __init__(self, phi, efficientdet_model, anchor_parameters = None, score_threshold = 0.01):
        super(EfficientDet_P, self).__init__()

        config = phi2config[phi]
        r_input = config['r_input']
        self.img_shape = (r_input, r_input, 3)
        
        self.efficientdet_model = efficientdet_model

        self.anchors = anchors_for_shape((r_input, r_input), anchor_params = anchor_parameters)
        self.anchors_input = np.expand_dims(self.anchors, axis=0)
        
        self.regressor_boxes = RegressBoxes(name = 'boxes')
        self.clip_boxes = ClipBoxes(name = 'clipped_boxes')

        self.filter_detector = FilterDetections(name = 'filtered_detections', score_threshold = score_threshold)
    
    def call(self, inputs):
        
        dict_out = self.efficientdet_model(inputs)
        
        class_out = dict_out['classification']
        box_out = dict_out['regression']

        delta_boxes = self.regressor_boxes([self.anchors_input, box_out[..., :4]])
        delta_boxes = self.clip_boxes([self.img_shape, delta_boxes])

        detections = self.filter_detector([delta_boxes, class_out])

        return class_out, box_out, detections

class BiFPN(models.Model):
    def __init__(self, num_channels, first):
        super(BiFPN, self).__init__()

        self.first = first
            
        self.p5_to_p6_conv2d = layers.Conv2D(num_channels, kernel_size = 1, padding = 'same', name = 'resample_p6/conv2d')
        self.p5_to_p6_bn     = layers.BatchNormalization(momentum = MOMENTUM, epsilon = EPSILON, name='resample_p6/bn')
        self.p5_to_p6_mxpool = layers.MaxPooling2D(pool_size = 3, strides = 2, padding = 'same', name='resample_p6/maxpool')

        self.p6_to_p7 = layers.MaxPooling2D(pool_size = 3, strides = 2, padding = 'same', name = 'resample_p7/maxpool')

        self.p5_down_channel_conv2d = layers.Conv2D(num_channels, kernel_size = 1, padding = 'same', name = f'resample_p5/conv2d')
        self.p5_down_channel_bn     = layers.BatchNormalization(momentum = MOMENTUM, epsilon = EPSILON, name = f'resample_p5/bn')

        self.p4_down_channel_conv2d = layers.Conv2D(num_channels, kernel_size = 1, padding = 'same', name = f'resample_p4/conv2d')
        self.p4_down_channel_bn     = layers.BatchNormalization(momentum = MOMENTUM, epsilon = EPSILON, name = f'resample_p4/bn')

        self.p3_down_channel_conv2d = layers.Conv2D(num_channels, kernel_size = 1, padding = 'same', name = f'resample_p3/conv2d')
        self.p3_down_channel_bn     = layers.BatchNormalization(momentum = MOMENTUM, epsilon = EPSILON, name = f'resample_p3/bn')

        self.p4_down_channel2_conv2d = layers.Conv2D(num_channels, kernel_size = 1, padding = 'same', name = f'resample_p4_2/conv2d')
        self.p4_down_channel2_bn     = layers.BatchNormalization(momentum = MOMENTUM, epsilon = EPSILON, name = f'resample_p4_2/bn')


        self.p5_down_channel2_conv2d = layers.Conv2D(num_channels, kernel_size = 1, padding = 'same', name = f'resample_p5_2/conv2d')
        self.p5_down_channel2_bn     = layers.BatchNormalization(momentum = MOMENTUM, epsilon = EPSILON, name = f'resample_p5_2/bn')

        self.p6_separableconv_up = SeparableConvBlock(num_channels = num_channels, kernel_size = 3, strides = 1, \
                                            momentum = MOMENTUM, epsilon = EPSILON, name = f'separableconv_up_p6')
        self.p5_separableconv_up = SeparableConvBlock(num_channels = num_channels, kernel_size = 3, strides = 1, \
                                            momentum = MOMENTUM, epsilon = EPSILON, name = f'separableconv_up_p5')
        self.p4_separableconv_up = SeparableConvBlock(num_channels = num_channels, kernel_size = 3, strides = 1, \
                                            momentum = MOMENTUM, epsilon = EPSILON, name = f'separableconv_up_p4')
        self.p3_separableconv_up = SeparableConvBlock(num_channels = num_channels, kernel_size = 3, strides = 1, \
                                            momentum = MOMENTUM, epsilon = EPSILON, name = f'separableconv_up_p3')

        self.p4_separableconv_down = SeparableConvBlock(num_channels = num_channels, kernel_size = 3, strides = 1, \
                                            momentum = MOMENTUM, epsilon = EPSILON, name = f'separableconv_down_p4')
        self.p5_separableconv_down = SeparableConvBlock(num_channels = num_channels, kernel_size = 3, strides = 1, \
                                            momentum = MOMENTUM, epsilon = EPSILON, name = f'separableconv_down_p5')                                                            
        self.p6_separableconv_down = SeparableConvBlock(num_channels = num_channels, kernel_size = 3, strides = 1, \
                                            momentum = MOMENTUM, epsilon = EPSILON, name = f'separableconv_down_p6')
        self.p7_separableconv_down = SeparableConvBlock(num_channels = num_channels, kernel_size = 3, strides = 1, \
                                            momentum = MOMENTUM, epsilon = EPSILON, name = f'separableconv_down_p7')            

        self.p7_upsample = layers.UpSampling2D()
        self.p6_upsample = layers.UpSampling2D()
        self.p5_upsample = layers.UpSampling2D()
        self.p4_upsample = layers.UpSampling2D()
        
        self.p6_downsample = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='downsample_p6/maxpool')
        self.p5_downsample = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='downsample_p5/maxpool')
        self.p4_downsample = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='downsample_p4/maxpool')
        self.p3_downsample = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='downsample_p3/maxpool')

        self.p6_td_wbifpn_add = wBiFPNAdd(name='wbifpn_p6_td/add')
        self.p5_td_wbifpn_add = wBiFPNAdd(name='wbifpn_p5_td/add')
        self.p4_td_wbifpn_add = wBiFPNAdd(name='wbifpn_p4_td/add')

        self.p3_out_wbifpn_add = wBiFPNAdd(name='wbifpn_p3_out/add')
        self.p4_out_wbifpn_add = wBiFPNAdd(name='wbifpn_p4_out/add')
        self.p5_out_wbifpn_add = wBiFPNAdd(name='wbifpn_p5_out/add')
        self.p6_out_wbifpn_add = wBiFPNAdd(name='wbifpn_p6_out/add')
        self.p7_out_wbifpn_add = wBiFPNAdd(name='wbifpn_p7_out/add')


        self.swish = layers.Activation(lambda x: tf.nn.swish(x))
            
    def call(self, inputs):
        if self.first == True:
            _, _, p3, p4, p5 = inputs
            
            p6_in = self.p5_to_p6_conv2d(p5)
            p6_in = self.p5_to_p6_bn(p6_in)
            p6_in = self.p5_to_p6_mxpool(p6_in)

            p7_in = self.p6_to_p7(p6_in)
            
            p5_in = self.p5_down_channel_conv2d(p5)
            p5_in = self.p5_down_channel_bn(p5_in)

            p4_in = self.p4_down_channel_conv2d(p4)
            p4_in = self.p4_down_channel_bn(p4_in)

            p3_in = self.p3_down_channel_conv2d(p3)
            p3_in = self.p3_down_channel_bn(p3_in)

        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs
        

        p6_td = self.p6_td_wbifpn_add([p6_in, self.p7_upsample(p7_in)])
        p6_td = self.p6_separableconv_up(self.swish(p6_td))

        p5_td = self.p5_td_wbifpn_add([p5_in, self.p6_upsample(p6_td)])
        p5_td = self.p5_separableconv_up(self.swish(p5_td))

        p4_td = self.p4_td_wbifpn_add([p4_in, self.p5_upsample(p5_td)])
        p4_td = self.p4_separableconv_up(self.swish(p4_td))

        p3_out = self.p3_out_wbifpn_add([p3_in, self.p4_upsample(p4_td)])
        p3_out = self.p3_separableconv_up(self.swish(p3_out))


        if self.first == True:
            p4_in = self.p4_down_channel2_conv2d(p4)
            p4_in = self.p4_down_channel2_bn(p4_in)

            p5_in = self.p5_down_channel2_conv2d(p5)
            p5_in = self.p5_down_channel2_bn(p5_in)
        
        p3_down = self.p3_downsample(p3_out)
        p4_out = self.p4_out_wbifpn_add([p4_in, p4_td, p3_down])
        p4_out = self.p4_separableconv_down(self.swish(p4_out))

        p4_down = self.p4_downsample(p4_out)
        p5_out = self.p5_out_wbifpn_add([p5_in, p5_td, p4_down])
        p5_out = self.p4_separableconv_down(self.swish(p5_out))

        p5_down = self.p5_downsample(p5_out)
        p6_out = self.p6_out_wbifpn_add([p6_in, p6_td, p5_down])
        p6_out = self.p6_separableconv_down(self.swish(p6_out))

        p6_down = self.p6_downsample(p6_out)
        p7_out = self.p7_out_wbifpn_add([p7_in, p6_down])
        p7_out = self.p7_separableconv_down(self.swish(p7_out))

        return p3_out, p4_out, p5_out, p6_out, p7_out


class BoxNet(models.Model):
    def __init__(self, width, depth, num_anchors=9, **kwargs):
        super(BoxNet, self).__init__(**kwargs)

        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        num_values = 4

        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)

        self.separable_convs = [layers.SeparableConv2D(filters = self.width, name = f'{self.name}/box-{i}', **options) for i in range(self.depth)]
        self.head = layers.SeparableConv2D(filters = num_anchors * num_values, name = f'{self.name}/box-predict', **options)

        self.bns = [[layers.BatchNormalization(momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/box-{i}-bn-{j}')
                    for j in range(3, 8)] for i in range(depth)]

        self.swish = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = layers.Reshape((-1, num_values))
        self.level = 0

    def call(self, inputs, level, **kwargs):
        feature = inputs
        for i in range(self.depth):
            feature = self.separable_convs[i](feature)
            feature = self.bns[i][level](feature)
            feature = self.swish(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        level += 1
        
        return outputs


class ClassNet(models.Model):
    def __init__(self, width, depth,  num_classes=20, num_anchors=9, **kwargs):
        super(ClassNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)

        self.separable_convs = [layers.SeparableConv2D(filters = self.width, bias_initializer = 'zeros', name = f'{self.name}/class-{i}', **options)
                                for i in range(depth)]
        self.head = layers.SeparableConv2D(filters = num_classes * num_anchors, bias_initializer = PriorProbability(probability = 0.01),
                                name = f'{self.name}/class-predict', **options)

        self.bns = [[layers.BatchNormalization(momentum = MOMENTUM, epsilon = EPSILON, name = f'{self.name}/class-{i}-bn-{j}') 
                    for j in range(3, 8)] for i in range(depth)]

        self.swish = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = layers.Reshape((-1, num_classes))
        self.activation = layers.Activation('sigmoid')

    def call(self, inputs, level, **kwargs):
        feature = inputs
        for i in range(self.depth):
            feature = self.separable_convs[i](feature)
            feature = self.bns[i][level](feature)
            feature = self.swish(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        outputs = self.activation(outputs)
        level += 1
        return outputs
