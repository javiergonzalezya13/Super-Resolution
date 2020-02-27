import datetime
import os
import sys
from multiprocessing import Process, Queue

import cv2
import numpy as np
from keras.applications.vgg19 import VGG19
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

    def build(self):
        print('[INFO] Creating FNet...')
        self.fnet = FNet(self.lr_shape).build()
        print('[INFO] FNet ready.')

        print('[INFO] Creating SRNet...')
        self.srnet = SRNet(self.lr_shape).build()
        print('[INFO] SRNet ready.')

        print('[INFO] Creating Upscaling...')
        self.upscaling = Upscaling(self.lr_shape).build()
        print('[INFO] Upscaling ready.')

        print('[INFO] Creating High Resolution Warp...')
        self.hr_warp = Warp(self.hr_shape, 1).build()
        print('[INFO] High Resolution Warp ready.')

        print('[INFO] Creating Low Resolution Warp...')
        self.lr_warp = Warp(self.lr_shape, 2).build()
        print('[INFO] Low Resolution Warp ready.')

        print('[INFO] Creating Space to Depth...')
        self.space2depth = Space2Depth(self.hr_shape).build()
        print('[INFO] Space to Depth ready.')

        print('[INFO] Creating Bicubic Upscaling...')
        self.bicubic_upscale = Upscaling(self.lr_shape, 1).build()
        print('[INFO] Bicubic Upscaling ready.')

        print('[INFO] Creating Discriminator...')
        self.discriminator = Discriminator(self.hr_shape).build()
        print('[INFO] Discriminator ready.')

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

        print('[INFO] Creating Temporally Coherent GAN...')

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

        self.hr_warping = Model([self.I_LR_t, self.I_LR_t_1, self.I_est_t_1],
                                self.I_hat_est_t_1)
        self.hr_warping.name = 'HR_Warping'

        # self.I_hr_past = Input(shape=self.hr_shape, name='I_est_t_1') # Generator input
        # self.I_hr_pres = Input(shape=self.hr_shape) # Training
        self.I_hr_pres, self.I_lr_warped = self.generator([self.I_LR_t, self.I_LR_t_1, self.I_est_t_1]) # Training
        self.I_hr_fut = Input(shape=self.hr_shape, name='I_LR_t_fut') # Discrminator input

        self.I_hr_warp_past = Input(shape=self.hr_shape, name='I_est_warp_t_1') # Discriminator input
        # self.I_hr_warp_pres = Input(shape=self.hr_shape) # Training
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
                                           0.25, 1.], # Warp
                             optimizer=self.opt)

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

    def train_gen(self, queue):
        while True:
            i, learning_rate, input_net, output_net = queue.get()
            print(i)
            print(input_net[0].shape)
            print(output_net[0].shape)
            print(learning_rate)
            print('Training % d' % (i))
            loss = self.generator.train_on_batch(input_net, output_net)
            print('Training done for iteration %d' % (i))
            # Training information
            if i % 100 == 0:
                print('[INFO]', '-'*15, 'Iteration %d' % i, '-'*15)
                print('[INFO] Total loss: %f \t Flow loss: %f \t SR loss: %f \t Learning rate: %f' % (loss[0], loss[1], loss[2], learning_rate))

            if i % 1000 == 0:
                print('[INFO] Saving model...')
                self.generator.save_weights(os.path.join(self.checkpoints_dir, 'generator_model_weights_%d.h5' % i))
                print('[INFO] Model saved.')
                # print('[INFO] Saving model time: %f seconds' % t_total.total_seconds())

    def train(self):
        i = 0
        # generator_iterations = 5e5
        samples_dir = os.path.join(self.output_dir, 'Samples')
        os.makedirs(samples_dir)
        self.checkpoints_dir = os.path.join(self.output_dir, 'Checkpoints')
        os.makedirs(self.checkpoints_dir)
        
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
        

        pretrain_gen = True

        if self.configs['train']['pretrained_disc']:
            print('[INFO] Loading pretrained discriminator...')
            pretrained_disc_file = os.path.join(self.configs['root_dir'], self.configs['train']['pretrained_disc'])
            self.discriminator.load_weights(pretrained_disc_file)
            pretrain_gen = False
            print('[INFO] Discriminator ready.')

        if self.configs['train']['pretrained_gen']:
            print('[INFO] Loading pretrained generator...')
            pretrained_gen_file = os.path.join(self.configs['root_dir'], self.configs['train']['pretrained_gen'])
            self.generator.load_weights(pretrained_gen_file)
            name = os.path.splitext(self.configs['train']['pretrained_gen'])[0]
            i_init = int(name.split('_')[-1])
            if (pretrain_gen) and (i_init == self.configs['train']['gen_iterations']):
                pretrain_gen = False
                i = 0
            else:
                i = i_init
            print('[INFO] Generator ready.')


        print('[INFO] Starting generator training process...')

        self.configs['train']['iterations'] += 5e4 # Add iterations for pretraining stage

        gen_learning_rate = 1e-4 / 2 ** (i // 5e4)
        if gen_learning_rate < 2.5e-5:
            gen_learning_rate = 2.5e-5

        rand_batch = np.array([])
        # q = Queue(1)
        # p = Process(target=self.train_gen, args=(q, ))
        # p.start()

        # ---------------------------------------------Pretrain generator------------------------------------------
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
                elif b % self.configs['data']['c_frames'] == 0:
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

            # Train net
            net_input = [lr_batch, prev_lr_batch, prev_est_batch]
            net_output = [hr_batch, lr_batch]

            gen_learning_rate = 1e-4 / 2 ** (i // 5e4)
            if gen_learning_rate < 2.5e-5:
                gen_learning_rate = 2.5e-5
            K.set_value(self.generator.optimizer.lr, gen_learning_rate)
            # print('Processing % d' % (i))
            # q.put([i, gen_learning_rate, net_input, net_output])

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

            # Save model
            if i % self.configs['train']['checkpoint_freq'] == 0:
                print('[INFO] Saving model...')
                self.generator.save_weights(os.path.join(self.checkpoints_dir, 'generator_model_weights_%d.h5' % i))
                print('[INFO] Model saved.')
                # print('[INFO] Saving model time: %f seconds' % t_total.total_seconds())

            i += 1

        # --------------------------------------------Train TecoGAN------------------------------------------------

        K.set_value(self.generator.optimizer.lr, 5e-5)

        print('\n[INFO] Generator learning rate: %f' % (K.eval(self.generator.optimizer.lr)))
        print('[INFO] Discriminator learning rate: %f' % (K.eval(self.discriminator.optimizer.lr)))
        print('[INFO] TecoGAN learning rate: %f' % (K.eval(self.tecogan.optimizer.lr)))

        lr_zeros = np.zeros(self.lr_shape)
        hr_zeros = np.zeros(self.hr_shape)
        D_ones = np.ones((self.configs['train']['batch_size'] * (self.configs['data']['c_frames'] - 2)) * 2)
        D_zeros = np.zeros((self.configs['train']['batch_size'] * (self.configs['data']['c_frames'] - 2)) * 2)

        print('[INFO] Starting TecoGAN training process...')

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
                elif b % self.configs['data']['c_frames'] == 0:
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
                elif (b + 1) % self.configs['data']['c_frames'] == 0:
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

            for b in range(hr_batch.shape[0]-1, -1, -1):
                if (b % self.configs['data']['c_frames'] == 0) or ((b+1) % self.configs['data']['c_frames'] == 0):
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

            # print('prev hr batch', prev_hr_batch.shape)

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

            # Save model
            if i % self.configs['train']['checkpoint_freq'] == 0:
                print('[INFO] Saving model...')
                self.generator.save_weights(os.path.join(self.checkpoints_dir, 'generator_model_weights_%d.h5' % i))
                self.discriminator.save_weights(os.path.join(self.checkpoints_dir, 'discriminator_model_weights_%d.h5' % i))
                print('[INFO] Model saved.')
                # print('[INFO] Saving model time: %f seconds' % t_total.total_seconds())

            i += 1
        print('[INFO] Training process ready.')

    def eval(self):
        pass
    
    def run(self):
        pass

    def vgg_22(self, y_true, y_pred):
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.hr_shape)
        vgg19.trainable = False
        for l in vgg19.layers:
            l.trainable = False
        
        model_22 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block2_conv2').output)
        model_22.trainable = False
        
        return K.mean(K.square(model_22(y_true) - model_22(y_pred)))

    def vgg_34(self, y_true, y_pred):
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.hr_shape)
        vgg19.trainable = False
        for l in vgg19.layers:
            l.trainable = False

        model_34 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block3_conv4').output)
        model_34.trainable = False

        return K.mean(K.square(model_34(y_true) - model_34(y_pred)))

    def vgg_44(self, y_true, y_pred):
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.hr_shape)
        vgg19.trainable = False
        for l in vgg19.layers:
            l.trainable = False
        
        model_44 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block4_conv4').output)
        model_44.trainable = False

        return K.mean(K.square(model_44(y_true) - model_44(y_pred)))

    def vgg_54(self, y_true, y_pred):
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.hr_shape)
        vgg19.trainable = False
        for l in vgg19.layers:
            l.trainable = False
        
        model_54 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model_54.trainable = False

        return K.mean(K.square(model_54(y_true) - model_54(y_pred)))
