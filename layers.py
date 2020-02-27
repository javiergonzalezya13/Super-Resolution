from keras.layers import (Add, BatchNormalization, Concatenate, Conv2D,
                          Conv2DTranspose, Dense, Input, Lambda, LeakyReLU,
                          MaxPool2D, ReLU, UpSampling2D, Flatten)
from keras.layers.core import Activation
from keras.models import Model
import keras.backend as K

# --------------------------------FRVSR and TecoGAN layers--------------------------------

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

# ------------------------------------EDVR layers--------------------------------
class PreDeblur(object):
    def __init__(self):
        pass
    def build(self):
        return 0

class PCDAlign(object):
    def __init__(self):
        pass
    def build(self):
        return 0

class TSAFusion(object):
    def __init__(self):
        pass
    def build(self):
        return 0

class Reconstruction(object):
    def __init__(self):
        pass
    def build(self):
        return 0