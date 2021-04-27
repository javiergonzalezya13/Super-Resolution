'''
Temporally Coherent GAN for Video Super-Resolution
'''
import datetime
import os
import sys

import cv2
import numpy as np
import yaml
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

from layers import *
from utils import *


class TecoGAN(object):
    def __init__(self, lr_shape, hr_shape, output_dir, configs):
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        self.output_dir = output_dir
        self.configs = configs

    # Building TecoGAN architecture process
    def build(self):
        # Build generator blocks
        print('[INFO] Creating FNet ...')
        self.fnet = FNet(self.lr_shape).build()
        print('[INFO] FNet ready.')

        print('[INFO] Creating SRNet ...')
        self.srnet = SRNet(self.lr_shape).build()
        print('[INFO] SRNet ready.')

        print('[INFO] Creating Upscaling ...')
        self.upscaling = Upscaling(self.lr_shape).build()
        print('[INFO] Upscaling ready.')

        print('[INFO] Creating High Resolution Warp ...')
        self.hr_warp = Warp(self.hr_shape, 1).build()
        print('[INFO] High Resolution Warp ready.')

        print('[INFO] Creating Low Resolution Warp ...')
        self.lr_warp = Warp(self.lr_shape, 2).build()
        print('[INFO] Low Resolution Warp ready.')

        print('[INFO] Creating Space to Depth ...')
        self.space2depth = Space2Depth(self.hr_shape).build()
        print('[INFO] Space to Depth ready.')

        print('[INFO] Creating Bicubic Upscaling ...')
        self.bicubic_upscale = Upscaling(self.lr_shape, 1).build()
        print('[INFO] Bicubic Upscaling ready.')

        # Build discriminator
        print('[INFO] Creating Discriminator ...')
        self.discriminator = Discriminator(self.hr_shape).build()
        print('[INFO] Discriminator ready.')

        # Save generator and discriminator blocks images
        if self.configs['stage']['train']:
            os.makedirs(self.output_dir)
            plot_model(self.fnet, to_file=os.path.join(self.output_dir, 'FNet.jpg'),
                       show_shapes=True, show_layer_names=True)
            plot_model(self.srnet, to_file=os.path.join(self.output_dir, 'SRNet.jpg'),
                       show_shapes=True, show_layer_names=True)
            plot_model(self.upscaling, to_file=os.path.join(self.output_dir, 'Upscaling.jpg'),
                       show_shapes=True, show_layer_names=True)
            plot_model(self.hr_warp, to_file=os.path.join(self.output_dir, 'HR_Warp.jpg'),
                       show_shapes=True, show_layer_names=True)
            plot_model(self.lr_warp, to_file=os.path.join(self.output_dir, 'LR_Warp.jpg'),
                       show_shapes=True, show_layer_names=True)
            plot_model(self.space2depth, to_file=os.path.join(self.output_dir, 'Space2Depth.jpg'),
                       show_shapes=True, show_layer_names=True)
            plot_model(self.bicubic_upscale, to_file=os.path.join(self.output_dir, 'BicubicUpscale.jpg'),
                       show_shapes=True, show_layer_names=True)
            plot_model(self.discriminator, to_file=os.path.join(self.output_dir, 'Discriminator.jpg'),
                       show_shapes=True, show_layer_names=True)

        # Assemble generator
        print('[INFO] Creating Temporally Coherent GAN ...')
        self.I_LR_t = Input(shape=self.lr_shape, name='I_LR_t')
        self.I_LR_t_1 = Input(shape=self.lr_shape, name='I_LR_t_1')
        self.I_est_t_1 = Input(shape=self.hr_shape, name='I_est_t_1')
        self.F_LR = self.fnet([self.I_LR_t, self.I_LR_t_1])
        self.I_LR_est_t_1 = self.lr_warp([self.I_LR_t_1, self.F_LR])
        self.F_HR = self.upscaling(self.F_LR)
        self.I_hat_est_t_1 = self.hr_warp([self.I_est_t_1, self.F_HR])
        self.S_s = self.space2depth(self.I_hat_est_t_1)
        self.I_res_t = self.srnet([self.I_LR_t, self.S_s])
        self.I_bic_t = self.bicubic_upscale(self.I_LR_t)
        self.I_est_t_plus = Add()([self.I_res_t, self.I_bic_t])
        self.I_est_t = Activation('tanh')(self.I_est_t_plus)

        self.generator = Model([self.I_LR_t, self.I_LR_t_1, self.I_est_t_1],
                               [self.I_est_t, self.I_LR_est_t_1])
        self.generator.name = 'Generator'

        # Assemble TecoGAN
        self.hr_warping = Model([self.I_LR_t, self.I_LR_t_1, self.I_est_t_1],
                                self.I_hat_est_t_1)
        self.hr_warping.name = 'HR_Warping'

        self.I_hr_pres, self.I_lr_warped = self.generator([self.I_LR_t, self.I_LR_t_1, self.I_est_t_1]) # Training
        self.I_hr_fut = Input(shape=self.hr_shape, name='I_LR_t_fut') # Discrminator input

        self.I_hr_warp_past = Input(shape=self.hr_shape, name='I_est_warp_t_1') # Discriminator input
        self.I_hr_warp_fut = Input(shape=self.hr_shape, name='I_est_warp_t_fut') # Discriminator input

        self.I_bic_past = Input(shape=self.hr_shape, name='I_bic_t_1') # Discriminator input, lr past
        self.I_bic_pres = Input(shape=self.hr_shape, name='I_bic_t') # Discriminator input, lr present
        self.I_bic_fut = Input(shape=self.hr_shape, name='I_bic_t_fut') # Discriminator input, lr future

        self.D_st, self.D_1, self.D_2, self.D_3, self.D_4 = self.discriminator([self.I_est_t_1, self.I_hr_pres, self.I_hr_fut,
                                                                                self.I_hat_est_t_1, self.I_hr_pres, self.I_hr_warp_fut,
                                                                                self.I_bic_past, self.I_bic_pres, self.I_bic_fut])

        self.tecogan = Model([self.I_LR_t, self.I_LR_t_1, self.I_est_t_1,
                              self.I_hr_fut,
                              self.I_hr_warp_fut,
                              self.I_bic_past, self.I_bic_pres, self.I_bic_fut],
                             [self.I_hr_pres, self.D_st,
                              self.D_1, self.D_2, self.D_3, self.D_4,
                              self.I_hr_pres, self.I_hr_pres, self.I_hr_pres, self.I_hr_pres,
                              self.I_hr_pres, self.I_lr_warped])

        # Set loss functions and learning rate
        self.opt = Adam(lr=5e-5)

        self.generator.compile(loss=['mean_squared_error', 'mean_squared_error'],
                               loss_weights=[1., 1.], optimizer=self.opt)

        self.discriminator.compile(loss=['binary_crossentropy',
                                         'mean_squared_error', 'mean_squared_error',
                                         'mean_squared_error', 'mean_squared_error'],
                                   loss_weights=[1., 0., 0., 0., 0.],
                                   optimizer=self.opt)

        self.tecogan.compile(loss=['mean_squared_error', 'binary_crossentropy', # GT, Dst
                                   'mean_squared_error', 'mean_squared_error', # Ds1, Ds2
                                   'mean_squared_error', 'mean_squared_error', # Ds3, Ds4
                                   self.vgg_22, self.vgg_34, self.vgg_44, self.vgg_54, # VGG 22, 34, 44, 54
                                   'mean_squared_error', 'mean_squared_error'], # Ping-Pong, Warp
                             loss_weights=[1., -1e-3, # GT, Dst
                                           1.67e-3, 1.43e-3, 8.33e-4, 2e-5, #  Ds1, Ds2, Ds3, Ds4
                                           3e-5, 1.4e-6, 6e-6, 2e-3, # VGG 22, 34, 44, 54
                                           0.25, 1.], # Ping-Pong, Warp
                             optimizer=self.opt)

        # Save TecoGAN images
        if self.configs['stage']['train']:
            plot_model(self.generator, to_file=os.path.join(self.output_dir, 'Generator.jpg'),
                       show_shapes=True, show_layer_names=True)
            plot_model(self.discriminator,
                       to_file=os.path.join(self.output_dir, 'Discriminator.jpg'),
                       show_shapes=True, show_layer_names=True)
            plot_model(self.tecogan, to_file=os.path.join(self.output_dir, 'TecoGAN.jpg'),
                       show_shapes=True, show_layer_names=True)
            plot_model(self.hr_warping, to_file=os.path.join(self.output_dir, 'HRWarping.jpg'),
                       show_shapes=True, show_layer_names=True)

        print('[INFO] Temporally Coherent GAN ready.')

    # Training process
    def train(self):
        i = 0
        samples_dir = os.path.join(self.output_dir, 'Samples')
        os.makedirs(samples_dir)
        self.checkpoints_dir = os.path.join(self.output_dir, 'Checkpoints')
        os.makedirs(self.checkpoints_dir)

        videos = get_videos(self.configs) 

        # Load pretrained generator and discriminator, and set iteration current iteration
        pretrain_gen = True

        if self.configs['train']['pretrained_disc']:
            print('[INFO] Loading pretrained discriminator ...')
            pretrained_disc_file = os.path.join(self.configs['root_dir'], self.configs['train']['pretrained_disc'])
            self.discriminator.load_weights(pretrained_disc_file)
            pretrain_gen = False
            print('[INFO] Discriminator ready.')

        if self.configs['cnn']['pretrained_model']:
            print('[INFO] Loading pretrained generator ...')
            pretrained_gen_file = os.path.join(self.configs['root_dir'], self.configs['cnn']['pretrained_model'])
            self.generator.load_weights(pretrained_gen_file)
            name = os.path.splitext(self.configs['cnn']['pretrained_model'])[0]
            i_init = int(name.split('_')[-1])
            i = i_init
            print('[INFO] Generator ready.')


        print('[INFO] Starting generator training process ...')

        self.configs['train']['iterations'] += 5e4

        gen_learning_rate = 1e-4 / 2 ** (i // 5e4)
        if gen_learning_rate < 2.5e-5:
            gen_learning_rate = 2.5e-5

        rand_batch = np.array([])

        # Pretrain generator 
        while i <= self.configs['train']['gen_iterations'] and pretrain_gen:
            # Choose batch index
            rand_batch = np.random.randint(0, len(videos), size=self.configs['train']['batch_size'])

            # Generate HR and LR batch samples at t
            for j in range(self.configs['train']['batch_size']):
                hr_single = get_frames(videos[rand_batch[j]], self.configs)
                if j == 0:
                    hr_batch = np.array(hr_single)
                else:
                    hr_batch = np.append(hr_batch, hr_single, axis=0)

            hr_batch = normalize(hr_batch)

            # Generate LR at t, and LR and estimated HR at t-1 for batch samples
            for b in range(hr_batch.shape[0]):
                est_frame = np.array([])
                single_lr = cv2.resize(cv2.GaussianBlur(hr_batch[b], (5, 5), 0),
                                       (self.lr_shape[0], self.lr_shape[1]),
                                       interpolation=cv2.INTER_CUBIC)

                # Initialize batch and frame zero of the batch
                if b == 0:
                    lr_batch = np.array([single_lr])
                    prev_lr_batch = np.array([np.zeros(self.lr_shape)])
                    prev_est_batch = np.array([np.zeros(self.hr_shape)])

                # Frame zero of the second and onward sequences
                elif b % self.configs['train']['c_frames'] == 0:
                    lr_batch = np.append(lr_batch, [single_lr], axis=0)
                    prev_lr_batch = np.append(prev_lr_batch, [np.zeros(self.lr_shape)], axis=0)
                    prev_est_batch = np.append(prev_est_batch, [np.zeros(self.hr_shape)], axis=0)

                # Frame one of the sequence
                else:
                    lr_batch = np.append(lr_batch, [single_lr], axis=0)
                    prev_lr_batch = np.append(prev_lr_batch, [lr_batch[b-1]], axis=0)
                    lr_frame = np.array([lr_batch[b-1]])
                    prev_lr_frame = np.array([prev_lr_batch[b-1]])
                    prev_est_frame = np.array([prev_est_batch[b-1]])
                    est_frame, _ = self.generator.predict([lr_frame, prev_lr_frame, prev_est_frame])
                    prev_est_batch = np.append(prev_est_batch, [est_frame[0]], axis=0)

            # Train generator
            net_input = [lr_batch, prev_lr_batch, prev_est_batch]
            net_output = [hr_batch, lr_batch]

            gen_learning_rate = 1e-4 / 2 ** (i // 5e4)
            if gen_learning_rate < 2.5e-5:
                gen_learning_rate = 2.5e-5
            K.set_value(self.generator.optimizer.lr, gen_learning_rate)

            loss = self.generator.train_on_batch(net_input, net_output)

            # Training information
            if i % 100 == 0:
                print('[INFO]', '-'*15, 'Iteration %d' % i, '-'*15)
                print('[INFO] Total loss: %f \t Flow loss: %f \t SR loss: %f \t Learning rate: %f' % (loss[0], loss[1], loss[2], K.eval(self.generator.optimizer.lr)))

            # Generate sample
            if i % self.configs['train']['sample_freq'] == 0:
                img_window_est = np.array([])
                img_window_hr = np.array([])
                img_window_lr = np.array([])
                est_batch, _ = self.generator.predict([lr_batch[0:5], prev_lr_batch[0:5], prev_est_batch[0:5]])

                for n in range(5):
                    try:
                        img_window_est = np.concatenate((img_window_est, denormalize(est_batch[n])), axis=1)
                        img_window_hr = np.concatenate((img_window_hr, denormalize(hr_batch[n])), axis=1)
                        img_lr = cv2.resize(denormalize(lr_batch[n]), (self.hr_shape[0], self.hr_shape[1]), interpolation=cv2.INTER_CUBIC)
                        img_window_lr = np.concatenate((img_window_lr, img_lr), axis=1)
                    except ValueError:
                        img_window_est = denormalize(est_batch[n])
                        img_window_hr = denormalize(hr_batch[n])
                        img_lr = cv2.resize(denormalize(lr_batch[n]), (self.hr_shape[0], self.hr_shape[1]), interpolation=cv2.INTER_CUBIC)
                        img_window_lr = img_lr
                    gen_psnr = psnr(est_batch[n], hr_batch[n])
                    bicubic_psnr = psnr(cv2.resize(lr_batch[n], (self.hr_shape[0], self.hr_shape[1]), interpolation=cv2.INTER_CUBIC), hr_batch[n])
                    print('[INFO] Generator PSNR: %f \t Bicubic PSNR: %f' % (gen_psnr, bicubic_psnr))
                img_window = np.concatenate((img_window_lr, img_window_est), axis=0)
                img_window = np.concatenate((img_window, img_window_hr), axis=0)

                cv2.imwrite(os.path.join(samples_dir, 'Sample_%d.jpg' % i), img_window)

            # Save checkpoint
            if i % self.configs['train']['checkpoint_freq'] == 0:
                print('[INFO] Saving model...')
                gen_weights = os.path.join(self.checkpoints_dir, 'generator_model_weights_%d.h5' % i)
                self.generator.save_weights(gen_weights)
                print('[INFO] Model saved.')
                yaml_file = os.path.join(self.output_dir, 'TecoGAN.yaml')
                self.configs['cnn']['pretrained_model'] = gen_weights
                with open(yaml_file, 'w') as file:
                    yaml.dump(self.configs, file, default_flow_style=False)

            i += 1

        # Train TecoGAN 
        K.set_value(self.generator.optimizer.lr, 5e-5)

        print('\n[INFO] Generator learning rate: %f' % (K.eval(self.generator.optimizer.lr)))
        print('[INFO] Discriminator learning rate: %f' % (K.eval(self.discriminator.optimizer.lr)))
        print('[INFO] TecoGAN learning rate: %f' % (K.eval(self.tecogan.optimizer.lr)))

        lr_zeros = np.zeros(self.lr_shape)
        hr_zeros = np.zeros(self.hr_shape)
        D_ones = np.ones((self.configs['train']['batch_size'] * (self.configs['train']['c_frames'] - 2)) * 2)
        D_zeros = np.zeros((self.configs['train']['batch_size'] * (self.configs['train']['c_frames'] - 2)) * 2)

        print('[INFO] Starting TecoGAN training process ...')

        while i <= self.configs['train']['iterations']:
            # Choose batch index
            rand_batch = np.random.randint(0, len(videos), size=self.configs['train']['batch_size'])

            # Generate HR at t batch samples
            for j in range(self.configs['train']['batch_size']):
                hr_single = get_frames(videos[rand_batch[j]], self.configs)
                if j == 0:
                    hr_batch = np.array(hr_single)
                else:
                    hr_batch = np.append(hr_batch, hr_single, axis=0)
            hr_batch = np.append(hr_batch, hr_batch[::-1], axis=0)

            hr_batch = normalize(hr_batch)
            # Generate batches at t and t-1
            for b in range(hr_batch.shape[0]):
                single_lr = cv2.resize(cv2.GaussianBlur(hr_batch[b], (5, 5), 0),
                                       (self.lr_shape[0], self.lr_shape[1]),
                                       interpolation=cv2.INTER_CUBIC)
                single_bic = cv2.resize(single_lr, (self.hr_shape[0], self.hr_shape[1]),
                                        interpolation=cv2.INTER_CUBIC)

                # Initialize batch and frame zero of the batch
                if b == 0:
                    lr_batch = np.array([single_lr])
                    prev_lr_batch = np.array([np.zeros(self.lr_shape)])
                    prev_est_batch = np.array([np.zeros(self.hr_shape)])
                    bic_batch = np.array([single_bic])
                    prev_bic_batch = np.array([np.zeros(self.hr_shape)])
                    prev_hr_batch = np.array([np.zeros(self.hr_shape)])

                    lr_frame = np.array([lr_batch[b]])
                    prev_lr_frame = np.array([prev_lr_batch[b]])
                    prev_hr_frame = np.array([prev_hr_batch[b]])
                    prev_est_frame = np.array([prev_est_batch[b]])
                    est_frame, _ = self.generator.predict([lr_frame, prev_lr_frame, prev_est_frame])
                    est_batch = np.array([est_frame[0]])
                    est_frame = self.hr_warping.predict([lr_frame, prev_lr_frame, prev_est_frame])
                    prev_hat_est_batch = np.array([est_frame[0]])
                    hr_frame = self.hr_warping.predict([lr_frame, prev_lr_frame, prev_hr_frame])
                    prev_hat_hr_batch = np.array([hr_frame[0]])

                # First frame of the second and onward sequences
                elif b % self.configs['train']['c_frames'] == 0:
                    lr_batch = np.append(lr_batch, [single_lr], axis=0)
                    prev_lr_batch = np.append(prev_lr_batch, [lr_zeros], axis=0)
                    prev_est_batch = np.append(prev_est_batch, [hr_zeros], axis=0)
                    bic_batch = np.append(bic_batch, [single_bic], axis=0)
                    prev_bic_batch = np.append(prev_bic_batch, [np.zeros(self.hr_shape)], axis=0)
                    prev_hr_batch = np.append(prev_hr_batch, [np.zeros(self.hr_shape)], axis=0)

                    lr_frame = np.array([lr_batch[b]])
                    prev_lr_frame = np.array([prev_lr_batch[b]])
                    prev_hr_frame = np.array([prev_hr_batch[b]])
                    prev_est_frame = np.array([prev_est_batch[b]])
                    est_frame, _ = self.generator.predict([lr_frame, prev_lr_frame, prev_est_frame])
                    est_batch = np.append(est_batch, [est_frame[0]], axis=0)
                    est_frame = self.hr_warping.predict([lr_frame, prev_lr_frame, prev_est_frame])
                    prev_hat_est_batch = np.append(prev_hat_est_batch, [est_frame[0]], axis=0)
                    hr_frame = self.hr_warping.predict([lr_frame, prev_lr_frame, prev_hr_frame])
                    prev_hat_hr_batch = np.append(prev_hat_hr_batch, [hr_frame[0]], axis=0)

                # Other cases
                else:
                    lr_batch = np.append(lr_batch, [single_lr], axis=0)
                    prev_lr_batch = np.append(prev_lr_batch, [lr_batch[b-1]], axis=0)
                    prev_est_batch = np.append(prev_est_batch, [est_batch[b-1]], axis=0)
                    bic_batch = np.append(bic_batch, [single_bic], axis=0)
                    prev_bic_batch = np.append(prev_bic_batch, [bic_batch[b-1]], axis=0)
                    prev_hr_batch = np.append(prev_hr_batch, [hr_batch[b-1]], axis=0)

                    lr_frame = np.array([lr_batch[b]])
                    prev_lr_frame = np.array([prev_lr_batch[b]])
                    prev_hr_frame = np.array([prev_hr_batch[b]])
                    prev_est_frame = np.array([prev_est_batch[b]])
                    est_frame, _ = self.generator.predict([lr_frame, prev_lr_frame, prev_est_frame])
                    est_batch = np.append(est_batch, [est_frame[0]], axis=0)
                    est_frame = self.hr_warping.predict([lr_frame, prev_lr_frame, prev_est_frame])
                    prev_hat_est_batch = np.append(prev_hat_est_batch, [est_frame[0]], axis=0)
                    hr_frame = self.hr_warping.predict([lr_frame, prev_lr_frame, prev_hr_frame])
                    prev_hat_hr_batch = np.append(prev_hat_hr_batch, [hr_frame[0]], axis=0)

            # Generate batches at t+1
            for b in range(hr_batch.shape[0]):
                if b == 0:
                    next_bic_batch = np.array([bic_batch[b+1]])
                    next_est_batch = np.array([est_batch[b+1]])
                    next_hr_batch = np.array([hr_batch[b+1]])

                    lr_frame = np.array([lr_batch[b]])
                    prev_lr_frame = np.array([lr_batch[b+1]])
                    prev_est_frame = np.array([next_est_batch[b]])
                    prev_hr_frame = np.array([next_hr_batch[b]])
                    est_frame = self.hr_warping.predict([lr_frame, prev_lr_frame, prev_est_frame])
                    next_hat_est_batch = np.array([est_frame[0]])
                    hr_frame = self.hr_warping.predict([lr_frame, prev_lr_frame, prev_hr_frame])
                    next_hat_hr_batch = np.array([hr_frame[0]])

                # Last frame of the sequence
                elif (b + 1) % self.configs['train']['c_frames'] == 0:
                    next_bic_batch = np.append(next_bic_batch, [np.zeros(self.hr_shape)], axis=0)
                    next_est_batch = np.append(next_est_batch, [np.zeros(self.hr_shape)], axis=0)
                    next_hr_batch = np.append(next_hr_batch, [np.zeros(self.hr_shape)], axis=0)
                    next_hat_est_batch = np.append(next_hat_est_batch, [np.zeros(self.hr_shape)], axis=0)
                    next_hat_hr_batch = np.append(next_hat_hr_batch, [np.zeros(self.hr_shape)], axis=0)

                # Other cases
                else:
                    next_bic_batch = np.append(next_bic_batch, [bic_batch[b+1]], axis=0)
                    next_est_batch = np.append(next_est_batch, [est_batch[b+1]], axis=0)
                    next_hr_batch = np.append(next_hr_batch, [hr_batch[b+1]], axis=0)

                    lr_frame = np.array([lr_batch[b]])
                    prev_lr_frame = np.array([lr_batch[b+1]])
                    prev_est_frame = np.array([next_est_batch[b]])
                    prev_hr_frame = np.array([next_hr_batch[b]])
                    est_frame = self.hr_warping.predict([lr_frame, prev_lr_frame, prev_est_frame])
                    next_hat_est_batch = np.append(next_hat_est_batch, [est_frame[0]], axis=0)
                    hr_frame = self.hr_warping.predict([lr_frame, prev_lr_frame, prev_hr_frame])
                    next_hat_hr_batch = np.append(next_hat_hr_batch, [hr_frame[0]], axis=0)

            # Remove useless images from batch
            for b in range(hr_batch.shape[0]-1, -1, -1):
                if (b % self.configs['train']['c_frames'] == 0) or ((b+1) % self.configs['train']['c_frames'] == 0):
                    prev_hr_batch = np.delete(prev_hr_batch, b, axis=0)
                    hr_batch = np.delete(hr_batch, b, axis=0)
                    next_hr_batch = np.delete(next_hr_batch, b, axis=0)
                    prev_lr_batch = np.delete(prev_lr_batch, b, axis=0)
                    lr_batch = np.delete(lr_batch, b, axis=0)
                    prev_bic_batch = np.delete(prev_bic_batch, b, axis=0)
                    bic_batch = np.delete(bic_batch, b, axis=0)
                    next_bic_batch = np.delete(next_bic_batch, b, axis=0)
                    prev_est_batch = np.delete(prev_est_batch, b, axis=0)
                    est_batch = np.delete(est_batch, b, axis=0)
                    next_est_batch = np.delete(next_est_batch, b, axis=0)
                    prev_hat_est_batch = np.delete(prev_hat_est_batch, b, axis=0)
                    next_hat_est_batch = np.delete(next_hat_est_batch, b, axis=0)
                    prev_hat_hr_batch = np.delete(prev_hat_hr_batch, b, axis=0)
                    next_hat_hr_batch = np.delete(next_hat_hr_batch, b, axis=0)

            # Discriminator training
            self.discriminator.trainable = True
            self.discriminator.compile(loss=['binary_crossentropy',
                                             'mean_squared_error', 'mean_squared_error',
                                             'mean_squared_error', 'mean_squared_error'],
                                       loss_weights=[1., 0., 0., 0., 0.],
                                       optimizer=self.opt)

            net_input = [prev_hr_batch, hr_batch, next_hr_batch,
                         prev_hat_hr_batch, hr_batch, next_hat_hr_batch,
                         prev_bic_batch, bic_batch, next_bic_batch]

            _, d_1, d_2, d_3, d_4 = self.discriminator.predict(net_input)

            net_output = [D_ones, d_1, d_2, d_3, d_4]

            self.discriminator.train_on_batch(net_input, net_output)

            net_input = [prev_est_batch, est_batch, next_est_batch,
                         prev_hat_est_batch, est_batch, next_hat_est_batch,
                         prev_bic_batch, bic_batch, next_bic_batch]

            net_output = [D_zeros, d_1, d_2, d_3, d_4]

            self.discriminator.train_on_batch(net_input, net_output)

            # GAN training
            self.discriminator.trainable = False

            net_input = [prev_hr_batch, hr_batch, next_hr_batch,
                         prev_hat_hr_batch, hr_batch, next_hat_hr_batch,
                         prev_bic_batch, bic_batch, next_bic_batch]

            _, d_1, d_2, d_3, d_4 = self.discriminator.predict(net_input)

            net_input = [lr_batch, prev_lr_batch, prev_est_batch,
                         next_est_batch, next_hat_est_batch,
                         prev_bic_batch, bic_batch, next_bic_batch]

            net_output = [hr_batch, D_ones,
                          d_1, d_2, d_3, d_4,
                          hr_batch, hr_batch, hr_batch, hr_batch,
                          np.flip(hr_batch, axis=0), lr_batch]

            for _ in range(2):
                loss = self.tecogan.train_on_batch(net_input, net_output)
            
            # Training information
            if i % 50 == 0:
                print('[INFO]', '-'*15, 'Iteration %d' % i, '-'*15)
                print('[INFO] Total loss: %f \t Gen loss: %f \t Disc loss: %f' % (loss[0], loss[1], loss[2]))

            # Generate sample
            if i % self.configs['train']['sample_freq'] == 0:
                img_window_est = np.array([])
                img_window_hr = np.array([])
                img_window_lr = np.array([])
                est_batch, _ = self.generator.predict([lr_batch[0:5], prev_lr_batch[0:5], prev_est_batch[0:5]])

                for n in range(5):
                    try:
                        img_window_est = np.concatenate((img_window_est, denormalize(est_batch[n])), axis=1)
                        img_window_hr = np.concatenate((img_window_hr, denormalize(hr_batch[n])), axis=1)
                        img_lr = cv2.resize(denormalize(lr_batch[n]), (self.hr_shape[0], self.hr_shape[1]), interpolation=cv2.INTER_CUBIC)
                        img_window_lr = np.concatenate((img_window_lr, img_lr), axis=1)
                    except ValueError:
                        img_window_est = denormalize(est_batch[n])
                        img_window_hr = denormalize(hr_batch[n])
                        img_lr = cv2.resize(denormalize(lr_batch[n]), (self.hr_shape[0], self.hr_shape[1]), interpolation=cv2.INTER_CUBIC)
                        img_window_lr = img_lr
                    gen_psnr = psnr(est_batch[n], hr_batch[n])
                    bicubic_psnr = psnr(cv2.resize(lr_batch[n], (self.hr_shape[0], self.hr_shape[1]), interpolation=cv2.INTER_CUBIC), hr_batch[n])
                    print('[INFO] Generator PSNR: %f \t Bicubic PSNR: %f' % (gen_psnr, bicubic_psnr))
                img_window = np.concatenate((img_window_lr, img_window_est), axis=0)
                img_window = np.concatenate((img_window, img_window_hr), axis=0)

                cv2.imwrite(os.path.join(samples_dir, 'Sample_%d.jpg' % i), img_window)

            # Save checkpoints
            if i % self.configs['train']['checkpoint_freq'] == 0:
                print('[INFO] Saving model ...')
                gen_weights = os.path.join(self.checkpoints_dir, 'generator_model_weights_%d.h5' % i)
                disc_weights = os.path.join(self.checkpoints_dir, 'discriminator_model_weights_%d.h5' % i)
                self.generator.save_weights(gen_weights)
                self.discriminator.save_weights(disc_weights)
                print('[INFO] Model saved.')
                yaml_file = os.path.join(self.output_dir, 'TecoGAN.yaml')
                self.configs['cnn']['pretrained_model'] = gen_weights
                self.configs['train']['pretrained_disc'] = disc_weights
                with open(yaml_file, 'w') as file:
                    yaml.dump(self.configs, file, default_flow_style=False)

            i += 1
        print('[INFO] Training process ready.')

    # Evaluation process
    def eval(self):
        # Load pretrained model
        print('[INFO] Loading pretrained model ...')

        if self.configs['cnn']['pretrained_model']:
            self.configs['cnn']['pretrained_model'] = os.path.join(self.configs['root_dir'],
                                                                    self.configs['cnn']['pretrained_model'])
            self.frvsr.load_weights(self.configs['cnn']['pretrained_model'])

        # Initialize variables and directories
        print('[INFO] Model ready.')
        print('[INFO] Evaluating model ...')
        os.makedirs(self.configs['eval']['output_dir'], exist_ok=True)
        f = open(os.path.join(self.configs['eval']['output_dir'], 'metrics.txt'), 'w+')

        apply_yolo = False
        if self.configs['eval']['yolo_model']:
            apply_yolo = True
            yolo = YoloV3(self.configs)

        rows = self.configs['data']['rows']
        cols = self.configs['data']['cols']

        window_rows = 2
        window_cols = 2

        est_frame = np.array([np.zeros((self.hr_shape[0]*rows, self.hr_shape[1]*cols, self.hr_shape[2]))])

        sub_prev_lr_frames = np.repeat(np.array([np.zeros(self.lr_shape)]), rows*cols, axis=0)
        sub_est_frames = np.repeat(np.array([np.zeros(self.hr_shape)]), rows*cols, axis=0)

        t = 0
        t1 = datetime.datetime.now()

        total_psnr = 0
        total_ssim = 0
        total_bic_psnr = 0
        total_bic_ssim = 0

        os.makedirs(self.configs['eval']['output_dir'], exist_ok=True)

        # Process videos
        videos = get_videos(self.configs)
        for video in videos:

            cap = cv2.VideoCapture(video)
            video_basename = os.path.basename(video)
            video_file = os.path.join(self.configs['eval']['output_dir'], video_basename)
            video_out = cv2.VideoWriter(video_file,
                                        cv2.VideoWriter_fourcc(*'DIVX'),
                                        10,
                                        (self.hr_shape[0]*rows,
                                         self.hr_shape[1]*cols))
            t_video = 0
            while cap.isOpened:
                img_windows = []
                img_window = np.array([])

                # Resize to high and low resolution 
                ret, frame = cap.read()
                if not ret:
                    break
                frame = normalize(frame)

                hr_frame = cv2.resize(frame,
                                      (self.hr_shape[0]*rows,
                                       self.hr_shape[1]*cols),
                                       interpolation=cv2.INTER_CUBIC)

                lr_frame = np.array([cv2.resize(cv2.GaussianBlur(hr_frame, (5, 5), 0),
                                                (self.lr_shape[0]*rows,
                                                 self.lr_shape[1]*cols),
                                    interpolation=cv2.INTER_CUBIC)])

                # Get low resolution sub images
                sub_lr_frames = np.array([])
                for i in range(rows):
                    for j in range(cols):
                        sub_lr_frame = lr_frame[0, i*self.lr_shape[0]:(i+1)*self.lr_shape[0],
                                                j*self.lr_shape[1]:(j+1)*self.lr_shape[1]]
                        try:
                            sub_lr_frames = np.append(sub_lr_frames, [sub_lr_frame], axis=0)

                        except ValueError:
                            sub_lr_frames = np.array([sub_lr_frame])

                # Get estimated sub images
                inference_time_1 = datetime.datetime.now()
                sub_est_frames, _ = self.generator.predict([sub_lr_frames,
                                                        sub_prev_lr_frames,
                                                        sub_est_frames])
                inference_time_2 = datetime.datetime.now()

                # Arrange estimated sub images
                for i in range(rows):
                    for j in range(cols):
                        est_frame[:, i*self.hr_shape[0]:(i+1)*self.hr_shape[0], j*self.hr_shape[1]:(j+1)*self.hr_shape[1]] = sub_est_frames[i*cols+j]

                # Calculate interpolations
                bicubic_frame = cv2.resize(lr_frame[0], (self.hr_shape[0]*rows, self.hr_shape[1]*cols),
                                           interpolation=cv2.INTER_CUBIC)
                nearest_frame = cv2.resize(lr_frame[0], (self.hr_shape[0]*rows, self.hr_shape[1]*cols),
                                           interpolation=cv2.INTER_NEAREST)

                sub_prev_lr_frames = sub_lr_frames
                
                t2 = datetime.datetime.now()
                delta_t = t2 - t1
                t1 = datetime.datetime.now()

                fps = int(1 / delta_t.total_seconds())
                delta_inference = inference_time_2 - inference_time_1
                inference_time = delta_inference.total_seconds() * 1000

                # Calculate metrics
                if t % 10 == 0:
                    nearest_psnr = psnr(nearest_frame, hr_frame)
                    bic_psnr = psnr(bicubic_frame, hr_frame)
                    est_psnr = psnr(hr_frame, est_frame[0])
                    total_psnr += est_psnr
                    total_bic_psnr += bic_psnr

                    nearest_ssim = ssim(nearest_frame, hr_frame)
                    bic_ssim = ssim(bicubic_frame, hr_frame)
                    est_ssim = ssim(hr_frame, est_frame[0])
                    total_ssim += est_ssim
                    total_bic_ssim += bic_ssim

                    if apply_yolo:
                        yolo_frame = yolo.predict_image(denormalize(est_frame[0]), yolo_f)

                    print('\n[INFO] Running at: %d[fps] \t TecoGAN inference time: %d[ms] \t Total inference time: %d[ms]' % (fps, inference_time, delta_t.total_seconds() * 1000))
                    print('[INFO] Nearest PSNR: %f \t Bicubic PSNR: %f \t TecoGAN PSNR: %f' % (nearest_psnr, bic_psnr, est_psnr))
                    print('[INFO] Nearest SSIM: %f \t Bicubic SSIM: %f \t TecoGAN SSIM: %f' % (nearest_ssim, bic_ssim, est_ssim))
                    f.write('Video:\t%s\tFrame:\t%d\tInference time:\t%d\tPSNR:\t%f\tSSIM:\t%f\n' % (video_basename, t_video, inference_time, est_psnr, est_ssim))
                t += 1
                t_video += 1

                # Show and write video
                img_window = denormalize(est_frame[0])

                if self.configs['eval']['watch']:
                    window_title = 'TecoGAN evaluation'
                    cv2.waitKey(1)
                    cv2.imshow(window_title, img_window)

                video_out.write(img_window)

            video_out.release()
            f.write('\n')

            if apply_yolo:
                yolo_f.write('\n')

        # Show average metrics        
        total_psnr = total_psnr / (t // 10)
        total_ssim = total_ssim / (t // 10)

        total_bic_psnr = total_bic_psnr / (t // 10)
        total_bic_ssim = total_bic_ssim / (t // 10)
        
        print('\n[INFO] Avg. Bicubic PSNR: %f\tAvg. TecoGAN PSNR: %f' % (total_bic_psnr, total_psnr))
        print('[INFO] Avg. Bicubic SSIM: %f\tAvg. TecoGAN SSIM: %f' % (total_bic_ssim, total_ssim))
        f.write('Total PSNR:\t%f\tTotal SSIM:\t%f' % (total_psnr, total_ssim))
        f.write('Total bicubuc PSNR:\t%f\tTotal bicubic SSIM:\t%f' % (total_bic_psnr, total_bic_ssim))

        f.close()
        if apply_yolo:
            yolo_f.close()

        cv2.destroyAllWindows()
        print('\n[INFO] Video stopped.')

    # Run TecoGAN on single video or camera
    def run(self):
        # Load pretrained generator
        print('[INFO] Loading pretrained model ...')

        if self.configs['cnn']['pretrained_model']:
            self.generator.load_weights(self.configs['cnn']['pretrained_model'])

        # Initialize variables
        print('[INFO] Model ready.')
        print('[INFO] Running model ...')

        rows = self.configs['data']['rows']
        cols = self.configs['data']['cols']

        window_rows = 2
        window_cols = 2

        prev_lr_frame = np.array([np.zeros((self.lr_shape[0] * rows, self.lr_shape[1] * cols, self.lr_shape[2]))])
        prev_est_frame = np.array([np.zeros((self.hr_shape[0] * rows, self.hr_shape[1] * cols, self.hr_shape[2]))])

        est_frame = np.array([np.zeros((self.hr_shape[0] * rows, self.hr_shape[1] * cols, self.hr_shape[2]))])

        sub_prev_lr_frames = np.repeat(np.array([np.zeros(self.lr_shape)]), rows * cols, axis=0)
        sub_est_frames = np.repeat(np.array([np.zeros(self.hr_shape)]), rows * cols, axis=0)

        t = 0
        t1 = datetime.datetime.now()
        
        total_psnr = 0
        total_ssim = 0
        total_bic_psnr = 0
        total_bic_ssim = 0

        if self.configs['run']['video']:
            cap = cv2.VideoCapture(self.configs['run']['video'])
        else:
            cap = cv2.VideoCapture(0)

        # Process video
        while cap.isOpened:
            img_windows = []
            img_window = np.array([])

            # Resize to high and low resolution
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = normalize(frame)

            hr_frame = cv2.resize(frame,
                                  (self.hr_shape[0]*rows,
                                   self.hr_shape[1]*cols),
                                  interpolation=cv2.INTER_CUBIC)

            lr_frame = np.array([cv2.resize(cv2.GaussianBlur(hr_frame, (5, 5), 0),
                                            (self.lr_shape[0]*rows,
                                             self.lr_shape[1]*cols),
                                interpolation=cv2.INTER_CUBIC)])

            # Get low resolution sub images 
            sub_lr_frames = np.array([])
            for i in range(rows):
                for j in range(cols):
                    sub_lr_frame = lr_frame[0, i*self.lr_shape[0]:(i+1)*self.lr_shape[0],
                                            j*self.lr_shape[1]:(j+1)*self.lr_shape[1]]

                    try:
                        sub_lr_frames = np.append(sub_lr_frames, [sub_lr_frame], axis=0)

                    except ValueError:
                        sub_lr_frames = np.array([sub_lr_frame])

            # Get estimated sub images
            inference_time_1 = datetime.datetime.now()
            sub_est_frames, _ = self.generator.predict([sub_lr_frames,
                                                        sub_prev_lr_frames,
                                                        sub_est_frames])
            inference_time_2 = datetime.datetime.now()

            # Arrange estimated sub images
            for i in range(rows):
                for j in range(cols):
                    est_frame[:, i*self.hr_shape[0]:(i+1)*self.hr_shape[0], j*self.hr_shape[1]:(j+1)*self.hr_shape[1]] = sub_est_frames[i*cols+j]

            # Calculate interpolations            
            bicubic_frame = cv2.resize(lr_frame[0], (self.hr_shape[0]*rows, self.hr_shape[1]*cols),
                                       interpolation=cv2.INTER_CUBIC)
            nearest_frame = cv2.resize(lr_frame[0], (self.hr_shape[0]*rows, self.hr_shape[1]*cols),
                                       interpolation=cv2.INTER_NEAREST)
            
            sub_prev_lr_frames = sub_lr_frames

            border = (self.hr_shape[0] - self.lr_shape[0]) * rows  // 2

            lr_scale = cv2.copyMakeBorder(lr_frame[0], border, border, border, border, cv2.BORDER_CONSTANT)

            img_windows.append(lr_scale)
            img_windows.append(est_frame[0])
            img_windows.append(bicubic_frame)
            img_windows.append(hr_frame)

            img_window = np.zeros((window_rows * self.hr_shape[0] * rows,
                                   window_cols * self.hr_shape[1] * cols,
                                   self.hr_shape[2]))

            for i in range(window_rows):
                for j in range(window_cols):
                    img_window[self.hr_shape[0] * rows * i:self.hr_shape[0] * rows * (i+1), self.hr_shape[1]  * cols * j:self.hr_shape[1] * cols * (j+1)] = img_windows[i*window_cols + j]

            t2 = datetime.datetime.now()
            delta_t = t2 - t1
            t1 = datetime.datetime.now()

            fps = int(1 / delta_t.total_seconds())
            delta_inference = inference_time_2 - inference_time_1
            inference_time = delta_inference.total_seconds() * 1000

            # Calculate metrics
            if t % 10 == 0:
                 nearest_psnr = psnr(nearest_frame, hr_frame)
                 bic_psnr = psnr(bicubic_frame, hr_frame)
                 est_psnr = psnr(hr_frame, est_frame[0])
                 total_psnr += est_psnr
                 total_bic_psnr += bic_psnr

                 nearest_ssim = ssim(nearest_frame, hr_frame)
                 bic_ssim = ssim(bicubic_frame, hr_frame)
                 est_ssim = ssim(hr_frame, est_frame[0])
                 total_ssim += est_ssim
                 total_bic_ssim += bic_ssim
                 
                 print('\n[INFO] Running at: %d[fps] \t TecoGAN inference time: %d[ms] \t Total inference time: %d[ms]' % (fps, inference_time, delta_t.total_seconds() * 1000))
                 print('[INFO] Nearest PSNR: %f \t Bicubic PSNR: %f \t TecoGAN PSNR: %f' % (nearest_psnr, bic_psnr, est_psnr))
                 print('[INFO] Nearest SSIM: %f \t Bicubic SSIM: %f \t TecoGAN SSIM: %f' % (nearest_ssim, bic_ssim, est_ssim))


            t += 1

            # Show video
            text_window = 'FPS:%d' % fps

            img_window = denormalize(img_window)
            
            window_title = 'Original / TecoGAN / Bicubic / High resolution'
            cv2.putText(img_window, text_window, (0, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_4)

            cv2.imshow(window_title, img_window)

            key = cv2.waitKey(1)
            if key == ord('p'):
                while True:
                    cv2.imshow(window_title, img_window)
                    if cv2.waitKey(1) == ord('p'):
                        break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print('\n[INFO] Video stopped.')

    # Loss function VGG networks
    def vgg_22(self, y_true, y_pred):
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.hr_shape)
        vgg19.trainable = False
        for l in vgg19.layers:
            l.trainable = False
        model22 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block2_conv2').output)
        model22.trainable = False
        return K.mean(K.square(model22(y_true) - model22(y_pred)))

    def vgg_34(self, y_true, y_pred):
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.hr_shape)
        vgg19.trainable = False
        for l in vgg19.layers:
            l.trainable = False
        model34 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block3_conv4').output)
        model34.trainable = False
        return K.mean(K.square(model34(y_true) - model34(y_pred)))

    def vgg_44(self, y_true, y_pred):
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.hr_shape)
        vgg19.trainable = False
        for l in vgg19.layers:
            l.trainable = False
        model44 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block4_conv4').output)
        model44.trainable = False
        return K.mean(K.square(model44(y_true) - model44(y_pred)))

    def vgg_54(self, y_true, y_pred):
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.hr_shape)
        vgg19.trainable = False
        for l in vgg19.layers:
            l.trainable = False
        model54 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model54.trainable = False
        return K.mean(K.square(model54(y_true) - model54(y_pred)))
