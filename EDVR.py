'''
EDVR implementation
'''
import os
import sys

import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

from layers import *

from utils import *
from skimage.measure import compare_ssim as ssim


class EDVR(object):
    def __init__(self, lr_shape, hr_shape, output_dir, configs):
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        self.output_dir = output_dir
        self.configs = configs

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

        print('[INFO] Creating Pre Deblur...')
        self.predeblur = PreDeblur(self.lr_shape).build()
        print('[INFO] Pre Deblur ready.')

        print('[INFO] Creating PCD Align...')
        self.pcdalign = PCDAlign(self.lr_shape).build()
        print('[INFO] PCD Align ready.')

        print('[INFO] Creating TSA Fusion...')
        self.tsafusion = TSAFusion(self.configs['train']['n_frames'], self.lr_shape).build()
        print('[INFO] TSA Fusion ready.')

        print('[INFO] Creating Reconstruction...')
        self.reconstruction = Reconstruction(self.lr_shape).build()
        print('[INFO] Reconstruction ready.')

        if self.configs['stage']['train']:
            os.makedirs(self.output_dir)
            plot_model(self.predeblur, to_file=os.path.join(self.output_dir, 'PreDeblur.jpg'),
                       show_shapes=True, show_layer_names=True)
            plot_model(self.pcdalign, to_file=os.path.join(self.output_dir, 'PCD_Align.jpg'),
                       show_shapes=True, show_layer_names=True)
            plot_model(self.tsafusion, to_file=os.path.join(self.output_dir, 'TSA_Fusion.jpg'),
                       show_shapes=True, show_layer_names=True)
            plot_model(self.reconstruction, to_file=os.path.join(self.output_dir, 'Reconstruction.jpg'),
                       show_shapes=True, show_layer_names=True)

        print('[INFO] Creating Video Restoration with Enhanced Deformable Convolutional Network...')

        input_shape = (self.configs['train']['n_frames'],
                       self.lr_shape[0],
                       self.lr_shape[1],
                       self.lr_shape[2])

        i_lr = Input(shape=input_shape, name='LR_frames')

        i_lr_reshaped = Lambda(lambda x: K.reshape(x, (-1, self.lr_shape[0], self.lr_shape[1], self.lr_shape[2])))(i_lr)

        i_lr_t = Lambda(lambda x: x[:, self.configs['train']['n_frames'] // 2, :, :, :])(i_lr)

        mid_layer_1 = self.predeblur(i_lr_reshaped)
        mid_layer_1 = Conv2D(filters=64, kernel_size=1)(mid_layer_1)
        for _ in range(5):
            mid_layer_1 = Residual_Block(mid_layer_1)

        mid_layer_2 = Conv_Block(mid_layer_1, 2)
        mid_layer_2 = Conv_Block(mid_layer_2)

        mid_layer_3 = Conv_Block(mid_layer_1, 2)
        mid_layer_3 = Conv_Block(mid_layer_3)

        mid_layer_1 = Lambda(lambda x: K.reshape(x, (-1, self.configs['train']['n_frames'],
                                            self.lr_shape[0], self.lr_shape[1], 64)))(mid_layer_1)
        mid_layer_2 = Lambda(lambda x: K.reshape(x, (-1, self.configs['train']['n_frames'],
                                            self.lr_shape[0] // 2, self.lr_shape[1] // 2, 64)))(mid_layer_2)
        mid_layer_3 = Lambda(lambda x: K.reshape(x, (-1, self.configs['train']['n_frames'],
                                            self.lr_shape[0] // 4, self.lr_shape[1] // 4, 64)))(mid_layer_3)

        ref_fea = [Lambda(lambda x: x[:, self.configs['train']['n_frames'] // 2, :, :, :])(mid_layer_1),
                     Lambda(lambda x: x[:, self.configs['train']['n_frames'] // 2, :, :, :])(mid_layer_2),
                     Lambda(lambda x: x[:, self.configs['train']['n_frames'] // 2, :, :, :])(mid_layer_3)]

        aligned_fea = []
        for j in range(self.configs['train']['n_frames']):
            nbr_fea = [Lambda(lambda x: x[:, j, :, :, :])(mid_layer_1),
                         Lambda(lambda x: x[:, j, :, :, :])(mid_layer_2),
                         Lambda(lambda x: x[:, j, :, :, :])(mid_layer_3)]
            
            pcd = self.pcdalign([ref_fea[0], ref_fea[1], ref_fea[2],
                                 nbr_fea[0], nbr_fea[1], nbr_fea[2]])
            aligned_fea.append(pcd)

        aligned_fea = Lambda(lambda x: K.stack(x, axis=1), name='PCD_aligned_features')(aligned_fea)

        features = self.tsafusion(aligned_fea)

        features = self.reconstruction(features)

        i_interp_t = UpSampling2D(size=4, interpolation='bilinear')(i_lr_t)
        i_hr_t = Add()([i_interp_t, features])

        self.edvr = Model(i_lr, i_hr_t)

        self.opt = Adam(lr=4e-4)

        self.edvr.compile(loss=self.charbonnier, optimizer=self.opt)

        if self.configs['stage']['train']:
            plot_model(self.edvr, to_file=os.path.join(self.output_dir, 'EDVR.jpg'),
                       show_shapes=True, show_layer_names=True)

        print('[INFO] Video Restoration with Enhanced Deformable Convolutional Network ready.')
        
    def train(self):
        i = 0
        samples_dir = os.path.join(self.output_dir, 'Samples')
        os.makedirs(samples_dir)
        checkpoints_dir = os.path.join(self.output_dir, 'Checkpoints')
        os.makedirs(checkpoints_dir)
        videos = []
        if isinstance(self.configs['data']['videos'], list):
            for data_dir in self.configs['data']['videos']:
                for f in os.listdir(data_dir):
                    if os.path.isfile(os.path.join(data_dir, f)):
                        videos.append(os.path.join(data_dir, f))

        else:
            data_dir = self.configs['data']['videos']
            for f in os.listdir(data_dir):
                if os.path.isfile(os.path.join(data_dir, f)):
                    videos.append(os.path.join(data_dir, f))

        print(len(videos), ' videos found.')

        if self.configs['train']['pretrained_model']:
            print('[INFO] Loading pretrained moddel...')
            self.edvr.load_weights(self.configs['train']['pretrained_model'])
            name = os.path.splitext(self.configs['train']['pretrained_model'])[0]
            i = int(name.split('_')[-1])
            print('[INFO] Model ready.')

        print('[INFO] Starting training process...')
        rand_batch = np.array([])
        while i < self.configs['train']['iterations'] + 1:
            rand_batch = np.random.randint(0, len(videos), size=self.configs['train']['batch_size'])

            for j in range(self.configs['train']['batch_size']):
                hr_single = get_frames(videos[rand_batch[j]], self.configs)
                if j == 0:
                    hr_batch = np.array([hr_single])
                else:
                    hr_batch = np.append(hr_batch, [hr_single], axis=0)

            hr_batch = normalize(hr_batch)
            # LR batch
            for b in range(hr_batch.shape[0]):
                lr_frames = []
                for j in range(self.configs['data']['c_frames']):
                    lr_frame = cv2.resize(cv2.GaussianBlur(hr_batch[b][j], (5, 5), 0),
                                          (self.lr_shape[0], self.lr_shape[1]),
                                          interpolation=cv2.INTER_CUBIC)
                    # if j == -self.configs['data']['c_frames']//2 + 1:
                    #     lr_frames = np.array([lr_frame])
                    # else:
                    #     lr_frames = np.append(lr_frames, [lr_frame], axis=0)
                    lr_frames.append(lr_frame)
                lr_frames = np.stack(lr_frames)

                if b == 0:
                    lr_batch = np.array([lr_frames])
                    hr_batch_2 = np.array([hr_batch[b][self.configs['data']['c_frames']//2]])
                else:
                    lr_batch = np.append(lr_batch, [lr_frames], axis=0)
                    hr_batch_2 = np.append(hr_batch_2, [hr_batch[b][self.configs['data']['c_frames']//2]], axis=0)
            
            hr_batch = hr_batch_2

            net_input = lr_batch
            net_output = hr_batch

            loss = self.edvr.train_on_batch(net_input, net_output)

            if i % self.configs['train']['info_freq'] == 0:
                print('[INFO]', '-'*15, 'Iteration %d' % i, '-'*15)
                print('[INFO] Charbonnier loss: %f' % (loss))

            if i % self.configs['train']['sample_freq'] == 0:
                img_window_est = np.array([])
                img_window_hr = np.array([])
                img_window_lr = np.array([])
                lr_batch = np.array([lr_batch[0]])
                est_batch = self.edvr.predict(lr_batch)
                for n in range(len(lr_batch)):
                    try:
                        img_window_est = np.concatenate((img_window_est, denormalize(est_batch[n])), axis=1)
                        img_window_hr = np.concatenate((img_window_hr, denormalize(hr_batch[n])), axis=1)
                        img_lr = cv2.resize(denormalize(test_batch[n][self.configs['data']['c_frames'] // 2]), (self.hr_shape[0], self.hr_shape[1]), interpolation=cv2.INTER_CUBIC)
                        img_window_lr = np.concatenate((img_window_lr, img_lr), axis=1)
                    except ValueError:
                        img_window_est = denormalize(est_batch[n])
                        img_window_hr = denormalize(hr_batch[n])
                        img_lr = cv2.resize(denormalize(lr_batch[n][self.configs['data']['c_frames']//2]), (self.hr_shape[0], self.hr_shape[1]), interpolation=cv2.INTER_CUBIC)
                        img_window_lr = img_lr
                    edvr_psnr = psnr(est_batch[n], hr_batch[n])
                    bicubic_psnr = psnr(cv2.resize(lr_batch[n][self.configs['data']['c_frames'] // 2], (self.hr_shape[0], self.hr_shape[1]), interpolation=cv2.INTER_CUBIC), hr_batch[n])
                    print('[INFO] EDVR PSNR: %f \t Bicubic PSNR: %f' % (edvr_psnr, bicubic_psnr))
                img_window = np.concatenate((img_window_lr, img_window_est), axis=0)
                img_window = np.concatenate((img_window, img_window_hr), axis=0)

                cv2.imwrite(os.path.join(samples_dir, 'Sample_%d.jpg' % i), img_window)

            if i % self.configs['train']['checkpoint_freq'] == 0:
                print('[INFO] Saving model...')
                self.edvr.save_weights(os.path.join(checkpoints_dir, 'edvr_model_weights_%d.h5' % i))
                print('[INFO] Model saved.')

            i += 1

    def eval(self):
        pass

    def run(self):
        pass

    def charbonnier(self, y_true, y_pred):
        square_diff = K.square(y_pred - y_true)
        charbonnier_loss = K.sqrt(square_diff + 1e-6)
        return K.mean(charbonnier_loss)
