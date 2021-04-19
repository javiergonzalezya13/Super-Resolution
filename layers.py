'''
Implementation of neural networks internal layers
'''

import keras.backend as K
import tensorflow as tf
from keras.layers import (Add, AveragePooling2D, BatchNormalization,
                          Concatenate, Conv2D, Conv2DTranspose, Dense, Flatten,
                          Input, Lambda, LeakyReLU, MaxPool2D, MaxPooling2D,
                          Multiply, ReLU, UpSampling2D)
from keras.layers.core import Activation
from keras.models import Model

# FRVSR and TecoGAN internal layers 
class FNet(object):
    def __init__(self, LR_shape):
        self.LR_shape = LR_shape

    def build(self):
        def Residual_Block(input_layer, filters, kernel=3, strides=1, leakage=0.2):
            x = Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding='same')(input_layer)
            x = LeakyReLU(leakage)(x)
            x = Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding='same')(x)
            output_layer = LeakyReLU(alpha=leakage)(x)
            return output_layer

        I_LR_t = Input(shape=self.LR_shape)
        I_LR_t_1 = Input(shape=self.LR_shape)
        mid_layer = Concatenate(axis=-1)([I_LR_t, I_LR_t_1])
        mid_layer = Residual_Block(mid_layer, 32)
        mid_layer = MaxPool2D(padding='same')(mid_layer)
        mid_layer = Residual_Block(mid_layer, 64)
        mid_layer = MaxPool2D(padding='same')(mid_layer)
        mid_layer = Residual_Block(mid_layer, 128)
        mid_layer = MaxPool2D(padding='same')(mid_layer)
        mid_layer = Residual_Block(mid_layer, 256)
        mid_layer = UpSampling2D(interpolation='bilinear')(mid_layer)
        mid_layer = Residual_Block(mid_layer, 128)
        mid_layer = UpSampling2D(interpolation='bilinear')(mid_layer)
        mid_layer = Residual_Block(mid_layer, 64)
        mid_layer = UpSampling2D(interpolation='bilinear')(mid_layer)
        mid_layer = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(mid_layer)
        mid_layer = LeakyReLU(0.2)(mid_layer)
        mid_layer = Conv2D(filters=2, kernel_size=3, strides=1, padding='same')(mid_layer)
        output = Activation('tanh')(mid_layer)

        model = Model([I_LR_t, I_LR_t_1], output)
        model.name = 'FNet'
        return model

class SRNet(object):
    def __init__(self, LR_shape):
        self.lr_shape = LR_shape
        self.Ss_shape = (LR_shape[0], LR_shape[1], LR_shape[2] * 16)

    def build(self):
        def Residual_Block(input_layer, filters=64, kernel=3, strides=1):
            x = Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding='same')(input_layer)
            x = ReLU()(x)
            x = Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding='same')(x)
            output_layer = Add()([input_layer, x])
            return output_layer

        I_LR_t = Input(shape=self.lr_shape)
        S_s = Input(shape=self.Ss_shape)
        mid_layer = Concatenate(axis=-1)([I_LR_t, S_s])
        mid_layer = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(mid_layer)
        mid_layer = ReLU()(mid_layer)
        for _ in range(10):
            mid_layer = Residual_Block(mid_layer)
        mid_layer = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(mid_layer)
        mid_layer = ReLU()(mid_layer)
        mid_layer = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(mid_layer)
        mid_layer = ReLU()(mid_layer)
        output = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(mid_layer)
        output = ReLU(max_value=1.0)(output)

        model = Model([I_LR_t, S_s], output)
        model.name = 'SRNet'
        return model

class Upscaling(object):
    def __init__(self, LR_shape, idx=0):
        self.lr_shape = LR_shape
        self.idx = idx

    def build(self):
        F_LR = Input(shape=self.lr_shape)
        output = UpSampling2D(size=4, interpolation='bilinear')(F_LR)

        model = Model(F_LR, output)
        model.name = 'Upscaling_' + str(self.idx)
        return model

class Space2Depth(object):
    def __init__(self, HR_shape):
        self.hr_shape = HR_shape

    def build(self):
        def space2depth_custom(input_shape, idx, scale=4):
            def space2depth_shape(input_shape):
                dims = [input_shape[0],
                        int(input_shape[1] / scale),
                        int(input_shape[2] / scale),
                        int(input_shape[3] * (scale ** 2))]
                output_shape = tuple(dims)
                return output_shape
            def subpixel(x):
                return K.tf.space_to_depth(x, scale)
            return Lambda(subpixel, output_shape=space2depth_shape, name='Space2Depth_'+str(idx))

        I_hat_est_t_1 = Input(shape=self.hr_shape)
        output = space2depth_custom(self.hr_shape, 1)(I_hat_est_t_1)

        model = Model(I_hat_est_t_1, output)
        model.name = 'Space_to_Depth'
        return model

class Warp(object):
    def __init__(self, input_shape, idx):
        self.input_shape = input_shape
        self.idx = idx

    def build(self):
        def warp_custom(input_shape, idx):
            def warp_shape(input_shape):
                dims = input_shape[0]
                output_shape = tuple(dims)
                return output_shape
            def subwarp(x):
                return K.tf.contrib.image.dense_image_warp(x[0], x[1])
            return Lambda(subwarp, output_shape=warp_shape, name='Warp_'+str(idx))

        flow = Input(shape=(self.input_shape[0], self.input_shape[1], 2))
        I_est_t_1 = Input(shape=self.input_shape)
        output = warp_custom(self.input_shape, self.idx)([I_est_t_1, flow])

        model = Model([I_est_t_1, flow], output)
        model.name = 'Warp_' + str(self.idx)
        return model

class Discriminator(object):
    def __init__(self, HR_shape):
        self.HR_shape = HR_shape

    def build(self):
        def Discriminator_Block(input_layer, filters):
            x = Conv2D(filters=filters, kernel_size=4, strides=2, padding='same')(input_layer)
            x = BatchNormalization()(x)
            output_layer = LeakyReLU(0.2)(x)
            return output_layer

        I_hr_past = Input(shape=self.HR_shape)
        I_hr_pres = Input(shape=self.HR_shape)
        I_hr_fut = Input(shape=self.HR_shape)

        I_hr_warp_past = Input(shape=self.HR_shape)
        I_hr_warp_pres = Input(shape=self.HR_shape)
        I_hr_warp_fut = Input(shape=self.HR_shape)

        I_bic_past = Input(shape=self.HR_shape)
        I_bic_pres = Input(shape=self.HR_shape)
        I_bic_fut = Input(shape=self.HR_shape)

        mid_layer = Concatenate(axis=-1)([I_hr_past, I_hr_pres, I_hr_fut,
                                          I_hr_warp_past, I_hr_warp_pres, I_hr_warp_fut,
                                          I_bic_past, I_bic_pres, I_bic_fut,])
        mid_layer = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(mid_layer)
        mid_layer = LeakyReLU(0.2)(mid_layer)
        mid_layer_1 = Discriminator_Block(mid_layer, 64)
        mid_layer_2 = Discriminator_Block(mid_layer_1, 64)
        mid_layer_3 = Discriminator_Block(mid_layer_2, 128)
        mid_layer_4 = Discriminator_Block(mid_layer_3, 256)
        mid_layer = Flatten()(mid_layer_4)
        mid_layer = Dense(1)(mid_layer)
        output = Activation('sigmoid')(mid_layer)

        model = Model([I_hr_past, I_hr_pres, I_hr_fut,
                       I_hr_warp_past, I_hr_warp_pres, I_hr_warp_fut,
                       I_bic_past, I_bic_pres, I_bic_fut],
                      [output,
                       mid_layer_1, mid_layer_2,
                       mid_layer_3, mid_layer_4])

        model.name = 'Discriminator'
        return model

# EDVR internal layers 
class PreDeblur(object):
    def __init__(self, lr_shape):
        self.lr_shape = lr_shape

    def build(self):
        def Conv_Block(input_layer, stride=1):
            x = Conv2D(filters=64, kernel_size=3, strides=stride, padding='same')(input_layer)
            output_layer = LeakyReLU(0.1)(x)
            return output_layer

        def Residual_Block(input_layer):            
            x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(input_layer)
            x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
            output_layer = Add()([input_layer, x])
            return output_layer
        
        i_lr = Input(shape=self.lr_shape)
        mid_layer = Conv_Block(i_lr)
        mid_layer_1 = Conv_Block(mid_layer, stride=2)
        mid_layer_2 = Conv_Block(mid_layer_1, stride=2)
        mid_layer_2 = Residual_Block(mid_layer_2)
        mid_layer_2 = UpSampling2D(interpolation='bilinear')(mid_layer_2)
        mid_layer_1 = Residual_Block(mid_layer_1)
        mid_layer_1 = Add()([mid_layer_1, mid_layer_2])
        mid_layer_1 = Residual_Block(mid_layer_1)
        mid_layer_1 = UpSampling2D(interpolation='bilinear')(mid_layer_1)
        mid_layer = Residual_Block(mid_layer)
        mid_layer = Add()([mid_layer, mid_layer_1])
        for _ in range(3):
            output = Residual_Block(mid_layer)

        model = Model(i_lr, output)
        model.name = 'PreDeblur'
        return model

class PCDAlign(object):
    def __init__(self, lr_shape):
        self.lr_shape = lr_shape

    def build(self):
        def Conv_Block(input_layer, stride=1):
            x = Conv2D(filters=64, kernel_size=3, strides=stride, padding='same')(input_layer)
            output_layer = LeakyReLU(0.1)(x)
            return output_layer

        def Residual_Block(input_layer):            
            x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(input_layer)
            x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
            output_layer = Add()([input_layer, x])
            return output_layer

        ref_fea_1 = Input(shape=(self.lr_shape[0], self.lr_shape[1], 64))
        ref_fea_2 = Input(shape=(self.lr_shape[0] // 2, self.lr_shape[1] // 2, 64))
        ref_fea_3 = Input(shape=(self.lr_shape[0] // 4, self.lr_shape[1] // 4, 64))

        nbr_fea_1 = Input(shape=(self.lr_shape[0], self.lr_shape[1], 64))
        nbr_fea_2 = Input(shape=(self.lr_shape[0] // 2, self.lr_shape[1] // 2, 64))
        nbr_fea_3 = Input(shape=(self.lr_shape[0] // 4, self.lr_shape[1] // 4, 64))

        middle_layer = Concatenate()([ref_fea_1, nbr_fea_1])
        middle_layer = Conv_Block(middle_layer)

        middle_layer_1 = Concatenate()([ref_fea_3, nbr_fea_3])
        middle_layer_1 = Conv2D(filters=64, kernel_size=3, padding='same')(middle_layer_1)
        middle_layer_1 = LeakyReLU(0.1)(middle_layer_1)
        middle_layer_1 = Conv2D(filters=64, kernel_size=3, padding='same')(middle_layer_1)
        middle_layer_1 = LeakyReLU(0.1)(middle_layer_1)

        middle_layer_2 = Conv2D(filters=64, kernel_size=3, padding='same')(middle_layer_1)
        middle_layer_2 = UpSampling2D(interpolation='bilinear')(middle_layer_2)

        middle_layer_1 = UpSampling2D(interpolation='bilinear')(middle_layer_1)
        middle_layer_1 = Lambda(lambda x: x*2)(middle_layer_1)

        middle_layer_3 = Concatenate()([ref_fea_2, nbr_fea_2])
        middle_layer_3 = Conv_Block(middle_layer_3)
        middle_layer_3 = Concatenate()([middle_layer_3, middle_layer_1])
        middle_layer_3 = Conv_Block(middle_layer_3)
        middle_layer_3 = Conv_Block(middle_layer_3)

        middle_layer_4 = Conv2D(filters=64, kernel_size=3, padding='same')(middle_layer_3)

        middle_layer_3 = UpSampling2D(interpolation='bilinear')(middle_layer_3)
        middle_layer_3 = Lambda(lambda x: x*2)(middle_layer_3)

        middle_layer = Concatenate()([middle_layer, middle_layer_3])
        middle_layer = Conv_Block(middle_layer)
        middle_layer = Conv_Block(middle_layer)
        middle_layer = Conv2D(filters=64, kernel_size=3, padding='same')(middle_layer)

        middle_layer_2 = Concatenate()([middle_layer_2, middle_layer_4])
        middle_layer_2 = Conv_Block(middle_layer_2)
        middle_layer_2 = UpSampling2D(interpolation='bilinear')(middle_layer_2)

        middle_layer = Concatenate()([middle_layer, middle_layer_2])
        middle_layer = Conv_Block(middle_layer)
        middle_layer = Concatenate()([middle_layer, ref_fea_1])
        middle_layer = Conv_Block(middle_layer)
        middle_layer = Conv_Block(middle_layer)
        middle_layer = Conv2D(filters=64, kernel_size=3, padding='same')(middle_layer)
        output = LeakyReLU(0.1)(middle_layer)

        model = Model([ref_fea_1, ref_fea_2, ref_fea_3,
                       nbr_fea_1, nbr_fea_2, nbr_fea_3],
                      output)
        model.name = 'PCD_Align'
        return model

class TSAFusion(object):
    def __init__(self, n_frames, lr_shape):
        self.n_frames = n_frames
        self.lr_shape = lr_shape

    def build(self):
        def Conv_Block(input_layer, stride=1):
            x = Conv2D(filters=64, kernel_size=3, strides=stride, padding='same')(input_layer)
            output_layer = LeakyReLU(0.1)(x)
            return output_layer

        def Residual_Block(input_layer):
            x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(input_layer)
            x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
            output_layer = Add()([input_layer, x])
            return output_layer

        aligned_fea = Input(shape=(self.n_frames, self.lr_shape[0], self.lr_shape[1], 64))
        middle_layer_1 = Lambda(lambda x: x[:, self.n_frames // 2, :, :, :])(aligned_fea)
        middle_layer_1 = Conv2D(filters=64, kernel_size=3, padding='same')(middle_layer_1)

        middle_layer_2 = Lambda(lambda x: K.reshape(x, (-1, self.lr_shape[0], self.lr_shape[1], 64)))(aligned_fea)
        middle_layer_2 = Conv2D(filters=64, kernel_size=3, padding='same')(middle_layer_2)
        middle_layer_2 = Lambda(lambda x: K.reshape(x, (-1, self.n_frames, self.lr_shape[0], self.lr_shape[1], 64)))(middle_layer_2)

        middle_layer_l = []

        for j in range(self.n_frames):
            middle_layer_3 = Lambda(lambda x: x[:, j, :, :, :])(middle_layer_2)
            middle_layer_3 = Multiply()([middle_layer_3, middle_layer_1])
            middle_layer_3 = Lambda(lambda x: K.sum(x, axis=-1))(middle_layer_3)
            middle_layer_3 = Lambda(lambda x: K.expand_dims(x, -1))(middle_layer_3)
            middle_layer_l.append(middle_layer_3)

        middle_layer = Concatenate()(middle_layer_l)
        middle_layer = Lambda(lambda x: K.sigmoid(x))(middle_layer)
        middle_layer = Lambda(lambda x: K.expand_dims(x, axis=-1))(middle_layer)
        middle_layer = 64 * [middle_layer]
        middle_layer = Concatenate()(middle_layer)
        middle_layer = Lambda(lambda x: K.reshape(x, (-1, self.lr_shape[0], self.lr_shape[1], self.n_frames * 64)))(middle_layer)

        middle_layer_1 = Lambda(lambda x: K.reshape(x, (-1, self.lr_shape[0], self.lr_shape[1], self.n_frames * 64)))(aligned_fea)
        
        middle_layer = Multiply()([middle_layer_1, middle_layer])
        
        middle_layer_1 = Conv2D(filters=64, kernel_size=1)(middle_layer)
        middle_layer_1 = LeakyReLU(0.1)(middle_layer_1)

        middle_layer = Conv2D(filters=64, kernel_size=1)(middle_layer)
        middle_layer = LeakyReLU(0.1)(middle_layer)

        middle_layer_2 = Lambda(lambda x: K.spatial_2d_padding(x))(middle_layer)
        middle_layer_2 = AveragePooling2D(pool_size=3, strides=2)(middle_layer_2)

        middle_layer_3 = Lambda(lambda x: K.spatial_2d_padding(x))(middle_layer)
        middle_layer_3 = MaxPool2D(pool_size=3, strides=2)(middle_layer_3)

        middle_layer = Concatenate()([middle_layer_2, middle_layer_3])
        middle_layer = Conv2D(filters=64, kernel_size=1)(middle_layer)
        middle_layer = LeakyReLU(0.1)(middle_layer)

        middle_layer_2 = Conv2D(filters=64, kernel_size=1)(middle_layer)
        middle_layer_2 = LeakyReLU(0.1)(middle_layer_2)

        middle_layer_3 = Lambda(lambda x: K.spatial_2d_padding(x))(middle_layer_2)
        middle_layer_3 = MaxPool2D(pool_size=3, strides=2)(middle_layer_3)

        middle_layer_2 = Lambda(lambda x: K.spatial_2d_padding(x))(middle_layer_2)
        middle_layer_2 = AveragePooling2D(pool_size=3, strides=2)(middle_layer_2)

        middle_layer_2 = Concatenate()([middle_layer_2, middle_layer_3])
        middle_layer_2 = Conv_Block(middle_layer_2)
        middle_layer_2 = UpSampling2D(interpolation='bilinear')(middle_layer_2)

        middle_layer = Add()([middle_layer, middle_layer_2])
        middle_layer = Conv2D(filters=64, kernel_size=1)(middle_layer)
        middle_layer = LeakyReLU(0.1)(middle_layer)
        middle_layer = UpSampling2D(interpolation='bilinear')(middle_layer)
        middle_layer = Conv2D(filters=64, kernel_size=3, padding='same')(middle_layer)
        
        middle_layer_2 = Lambda(lambda x: K.sigmoid(x))(middle_layer)

        middle_layer = Conv2D(filters=64, kernel_size=1)(middle_layer)
        middle_layer = LeakyReLU(0.1)(middle_layer)
        middle_layer = Conv2D(filters=64, kernel_size=1)(middle_layer)

        output = Lambda(lambda x: x[0] * x[1] * 2 + x[2])([middle_layer_1, middle_layer_2, middle_layer])

        model = Model(aligned_fea, output)
        model.name = 'TSA_Fusion'
        return model

class Reconstruction(object):
    def __init__(self, lr_shape):
        self.lr_shape = lr_shape

    def build(self):
        def SubpixelConv2D(input_shape, scale=2, idx=0):
            def subpixel_shape(input_shape):
                dims = [input_shape[0],
                        input_shape[1] * scale,
                        input_shape[2] * scale,
                        int(input_shape[3] / (scale ** 2))]
                output_shape = tuple(dims)
                return output_shape

            def subpixel(x):
                return tf.depth_to_space(x, scale)

            return Lambda(subpixel, output_shape=subpixel_shape, name='subpixel_' + str(idx))

        def Residual_Block(input_layer):
            x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(input_layer)
            x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
            output_layer = Add()([input_layer, x])
            return output_layer

        input_layer = Input(shape=(self.lr_shape[0], self.lr_shape[1], 64))

        mid_layer = input_layer

        for _ in range(10):
            mid_layer = Residual_Block(mid_layer)

        mid_layer = SubpixelConv2D((self.lr_shape[0], self.lr_shape[1], 64))(mid_layer)
        mid_layer = LeakyReLU(0.1)(mid_layer)
        mid_layer = SubpixelConv2D((self.lr_shape[0] * 2, self.lr_shape[1] * 2, 16), idx=1)(mid_layer)
        mid_layer = LeakyReLU(0.1)(mid_layer)
        mid_layer = Conv2D(filters=64, kernel_size=3, padding='same')(mid_layer)
        mid_layer = LeakyReLU(0.1)(mid_layer)
        output = Conv2D(filters=3, kernel_size=3, padding='same')(mid_layer)

        model = Model(input_layer, output)
        model.name = 'Reconstruction'
        return model
