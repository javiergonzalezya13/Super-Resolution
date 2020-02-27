'''
Frame-Recurrent VVideo Super Resolution implementation
'''
import datetime
import os
import sys

import cv2
import numpy as np
from keras.optimizers import Adam
from keras.utils import plot_model

from layers import *
from utils import *
from skimage.measure import compare_ssim as ssim

class FrameRecurrentVideoSR(object):
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

        print('[INFO] Creating Frame-Recurrent Video Super Resolution...')

        I_LR_t = Input(shape=self.lr_shape, name='I_LR_t')
        I_LR_t_1 = Input(shape=self.lr_shape, name='I_LR_t_1')
        I_est_t_1 = Input(shape=self.hr_shape, name='I_est_t_1')
        F_LR = self.fnet([I_LR_t, I_LR_t_1])
        I_LR_est_t_1 = self.lr_warp([I_LR_t_1, F_LR])
        F_HR = self.upscaling(F_LR)
        I_hat_est_t_1 = self.hr_warp([I_est_t_1, F_HR])
        S_s = self.space2depth(I_hat_est_t_1)
        I_est_t = self.srnet([I_LR_t, S_s])

        self.frvsr = Model([I_LR_t, I_LR_t_1, I_est_t_1],
                           [I_est_t, I_LR_est_t_1])
        self.opt = Adam(lr=1e-4)
        self.frvsr.compile(loss=['mean_squared_error', 'mean_squared_error'],
                           loss_weights=[1., 1.], optimizer=self.opt)

        if self.configs['stage']['train']:
            plot_model(self.frvsr, to_file=os.path.join(self.output_dir, 'FRVSR.jpg'),
                        show_shapes=True, show_layer_names=True)

        print('[INFO] Frame-Recurrent Video Super Resolution ready.')

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

        if self.configs['train']['pretrained_model']:
            print('[INFO] Loading pretrained model...')
            self.frvsr.load_weights(self.configs['train']['pretrained_model'])
            name = os.path.splitext(self.configs['train']['pretrained_model'])[0]
            i = int(name.split('_')[-1])
            print('[INFO] Model ready.')

        print('[INFO] Starting training process...')
        rand_batch = np.array([])
        while i < self.configs['train']['iterations'] + 1:

            # Choose batch index
            # t1 = datetime.datetime.now()
            rand_batch = np.random.randint(0, len(videos), size=self.configs['train']['batch_size'])

            # Generate HR batch samples at t
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
                                       (self.lr_shape[0], self.lr_shape[1]), interpolation=cv2.INTER_CUBIC)

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
                elif b % self.configs['data']['c_frames'] == 1:
                    lr_batch = np.append(lr_batch, [single_lr], axis=0)
                    prev_lr_batch = np.append(prev_lr_batch, [lr_batch[b-1]], axis=0)
                    lr_frame = np.array([lr_batch[b-1]])
                    prev_lr_frame = np.array([np.zeros(self.lr_shape)])
                    prev_est_frame = np.array([np.zeros(self.hr_shape)])
                    est_frame, _ = self.frvsr.predict([lr_frame, prev_lr_frame, prev_est_frame])
                    prev_est_batch = np.append(prev_est_batch, [est_frame[0]], axis=0)

                # Other cases
                else:
                    lr_batch = np.append(lr_batch, [single_lr], axis=0)
                    prev_lr_batch = np.append(prev_lr_batch, [lr_batch[b-1]], axis=0)
                    lr_frame = np.array([lr_batch[b-1]])
                    prev_lr_frame = np.array([lr_batch[b-2]])
                    prev_est_frame = np.array([prev_est_batch[b-2]])
                    est_frame, _ = self.frvsr.predict([lr_frame, prev_lr_frame, prev_est_frame])
                    prev_est_batch = np.append(prev_est_batch, [est_frame[0]], axis=0)
            # t2 = datetime.datetime.now()
            # t_total = t2 - t1
            # print('[INFO] Processing time: %f seconds' % t_total.total_seconds())
            # Train net
            net_input = [lr_batch, prev_lr_batch, prev_est_batch]
            net_output = [hr_batch, lr_batch]
            loss = self.frvsr.train_on_batch(net_input, net_output)
            # t2 = datetime.datetime.now()
            # t_total = t2 - t1
            # print('[INFO] Training time: %f seconds' % t_total.total_seconds())

            # Training information
            if i % self.configs['train']['info_freq'] == 0:
                print('[INFO]', '-'*15, 'Iteration %d' % i, '-'*15)
                print('[INFO] Total loss: %f \t SR loss: %f \t Flow loss: %f' % (loss[0], loss[1], loss[2]))

            # Generate sample
            if i % self.configs['train']['sample_freq'] == 0:
                # t1 = datetime.datetime.now()    
                img_window_est = np.array([])
                img_window_hr = np.array([])
                img_window_lr = np.array([])
                est_batch, _ = self.frvsr.predict([lr_batch[0:5], prev_lr_batch[0:5], prev_est_batch[0:5]])

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
                    frvsr_psnr = psnr(est_batch[n], hr_batch[n])
                    bicubic_psnr = psnr(cv2.resize(lr_batch[n], (self.hr_shape[0], self.hr_shape[1]), interpolation=cv2.INTER_CUBIC), hr_batch[n])
                    print('[INFO] FRVSR PSNR: %f \t Bicubic PSNR: %f' % (frvsr_psnr, bicubic_psnr))
                img_window = np.concatenate((img_window_lr, img_window_est), axis=0)
                img_window = np.concatenate((img_window, img_window_hr), axis=0)

                cv2.imwrite(os.path.join(samples_dir, 'Sample_%d.jpg' % i), img_window)

                # t2 = datetime.datetime.now()
                # t_total = t2 - t1
                # print('[INFO] Generated samples time: %f seconds' % t_total.total_seconds())

            if i % self.configs['train']['checkpoint_freq'] == 0:
                # t1 = datetime.datetime.now()
                print('[INFO] Saving model...')
                # frvsr.save('frvsr_model_%d.h5' % i)
                self.frvsr.save_weights(os.path.join(checkpoints_dir, 'frvsr_model_weights_%d.h5' % i))
                print('[INFO] Model saved.')
                # t2 = datetime.datetime.now()
                # t_total = t2 - t1
                # print('[INFO] Saving model time: %f seconds' % t_total.total_seconds())

            i += 1

        print('[INFO] Training process ready.')

    def eval(self):

        print('[INFO] Loading pretrained model...')

        if self.configs['eval']['pretrained_model']:
            self.frvsr.load_weights(self.configs['eval']['pretrained_model'])

        print('[INFO] Model ready.')
        print('[INFO] Evaluating model...')

        rows = self.configs['eval']['rows']
        cols = self.configs['eval']['cols']

        window_rows = 2
        window_cols = 2

        est_frame = np.array([np.zeros((self.hr_shape[0] * rows, self.hr_shape[1] * cols, self.hr_shape[2]))])

        sub_prev_lr_frames = np.repeat(np.array([np.zeros(self.lr_shape)]), rows * cols, axis=0)
        sub_est_frames = np.repeat(np.array([np.zeros(self.hr_shape)]), rows * cols, axis=0)

        t = 0
        t1 = datetime.datetime.now()

        if self.configs['eval']['video']:
            cap = cv2.VideoCapture(self.configs['run']['video'])
        else:
            print('[ERROR] Not a valid video.')
            sys.exit(0)
            # cap = cv2.VideoCapture(0)

        # video_out = cv2.VideoWriter('video_test.avi',
        #                             cv2.VideoWriter_fourcc(*'DIVX'),
        #                             10,
        #                             (self.configs['data']['high_res'] * rows * window_rows,
        #                              self.configs['data']['high_res'] * cols * window_cols))

        video_out = cv2.VideoWriter('video_test.avi',
                            cv2.VideoWriter_fourcc(*'DIVX'),
                            10,
                            (self.configs['data']['high_res'] * rows,
                             self.configs['data']['high_res'] * cols))


        while cap.isOpened:
            img_windows = []
            img_window = np.array([])

            ret, frame = cap.read()
            if not ret:
                break
            frame = normalize(frame)

            hr_frame = cv2.resize(frame,
                                  (self.configs['data']['high_res'] * rows, self.configs['data']['high_res'] * cols),
                                  interpolation=cv2.INTER_CUBIC)

            lr_frame = np.array([cv2.resize(cv2.GaussianBlur(hr_frame, (5, 5), 0),
                                            (self.configs['data']['low_res'] * rows,
                                             self.configs['data']['low_res'] * cols),
                                interpolation=cv2.INTER_CUBIC)])

            sub_hr_frames = np.array([])
            sub_lr_frames = np.array([])
            for i in range(rows):
                for j in range(cols):
                    sub_lr_frame = lr_frame[0, i * self.configs['data']['low_res']: (i+1) * self.configs['data']['low_res'],
                                            j * self.configs['data']['low_res']: (j+1) * self.configs['data']['low_res']]

                    try:
                        sub_lr_frames = np.append(sub_lr_frames, [sub_lr_frame], axis=0)

                    except ValueError:
                        sub_lr_frames = np.array([sub_lr_frame])

            inference_time_1 = datetime.datetime.now()
            sub_est_frames, _ = self.frvsr.predict([sub_lr_frames,
                                                    sub_prev_lr_frames,
                                                    sub_est_frames])
            inference_time_2 = datetime.datetime.now()

            for i in range(rows):
                for j in range(cols):
                    est_frame[:, i * self.configs['data']['high_res']: (i+1) * self.configs['data']['high_res'], j * self.configs['data']['high_res']: (j+1) * self.configs['data']['high_res']] = sub_est_frames[i*cols + j]

            bicubic_frame = cv2.resize(lr_frame[0], (self.configs['data']['high_res'] * rows, self.configs['data']['high_res'] * cols),
                                       interpolation=cv2.INTER_CUBIC)

            sub_prev_lr_frames = sub_lr_frames

            nearest_frame = cv2.resize(lr_frame[0], (self.configs['data']['high_res'] * rows, self.configs['data']['high_res'] * cols),
                                       interpolation=cv2.INTER_NEAREST)

            border = 256 - 64

            lr_scale = cv2.copyMakeBorder(lr_frame[0], border, border, border, border, cv2.BORDER_CONSTANT)

            # img_windows.append(hr_frame)
            # print(hr_frame.dtype)
            img_windows.append(lr_scale)
            img_windows.append(est_frame[0])
            # print(est_frame[0].dtype)
            img_windows.append(bicubic_frame)
            # print(bicubic_frame.dtype)
            img_windows.append(nearest_frame)
            # print(nearest_frame.dtype)

            img_window = np.zeros((window_rows * self.hr_shape[0] * rows,
                                   window_cols * self.hr_shape[1] * cols,
                                   self.hr_shape[2]))
            
            # for img in img_windows:
            #     try:
            #         img_window = np.concatenate((img_window, img), axis=0)
            #     except ValueError:
            #         img_window = np.array(img)

            for i in range(window_rows):
                for j in range(window_cols):
                    img_window[self.hr_shape[0] * rows * i:self.hr_shape[0] * rows * (i+1), self.hr_shape[1]  * cols * j:self.hr_shape[1] * cols * (j+1)] = img_windows[i*window_cols + j]

            # img_window = np.concatenate((hr_frame, est_frame[0]), axis=1)
            # img_window = np.concatenate((img_window, bicubic_frame), axis=1)
            # img_window = np.concatenate((img_window, nearest_frame), axis=1)

            t2 = datetime.datetime.now()
            delta_t = t2 - t1
            t1 = datetime.datetime.now()

            fps = int(1 / delta_t.total_seconds())
            delta_inference = inference_time_2 - inference_time_1
            inference_time = delta_inference.total_seconds() * 1000

            gray_hr_frame = cv2.cvtColor(hr_frame, cv2.COLOR_BGR2GRAY)
            gray_nearest_frame = cv2.cvtColor(nearest_frame, cv2.COLOR_BGR2GRAY)
            gray_bicubic_frame = cv2.cvtColor(bicubic_frame, cv2.COLOR_BGR2GRAY)
            gray_est_frame = cv2.cvtColor(est_frame[0].astype('Float32'), cv2.COLOR_BGR2GRAY)

            if t == 100:
                print('\n[INFO] Running at: %d[fps] \t FRVSR inference time: %d[ms] \t Total inference time: %d[ms]' % (fps, inference_time, delta_t.total_seconds() * 1000))
                print('[INFO] Nearest PSNR: %f \t Bicubic PSNR: %f \t FRVSR PSNR: %f' % (psnr(nearest_frame, hr_frame), psnr(bicubic_frame, hr_frame), psnr(hr_frame, est_frame[0])))
                print('[INFO] Nearest SSIM: %f \t Bicubic SSIM: %f \t FRVSR SSIM: %f' % (ssim(gray_nearest_frame, gray_hr_frame, full=True)[0],
                                                                                         ssim(gray_bicubic_frame, gray_hr_frame, full=True)[0],
                                                                                         ssim(gray_hr_frame, gray_est_frame, full=True)[0]))
                t = 0
            t += 1

            text_window = 'FPS:%d' % fps

            window_title = 'Original / Frame-Recurrent Video Super Resolution / Bicubic / Nearest'
            cv2.putText(img_window, text_window, (0, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 1, 0), 1, cv2.LINE_4)

            # video_out.write(denormalize(est_frame[0]))

            # img_window_2 = denormalize(img_window)

            # img_window_2 = cv2.threshold(est_frame[0], 0.0, 0.0, cv2.THRESH_TOZERO)[1]
            img_window_2 = denormalize(est_frame[0])
            # img_window_2 = cv2.threshold(img_window_2, 255, 255, cv2.THRESH_TRUNC)[1]

            if est_frame[0].max() > 1.0:
                print('[ERROR] Max value exceeded: %d' % est_frame[0].max())            

            video_out.write(img_window_2)
            cv2.imshow(window_title, img_window_2)
            
            key = cv2.waitKey(1)
            if key == ord('p'):
                while True:
                    # cv2.imshow('Low Resolution Frame', lr_frame[0])
                    cv2.imshow(window_title, img_window)
                    if cv2.waitKey(1) == ord('p'):
                        break
            elif key == ord('q'):
                break
            
        # video_out.release()
        print('\n[INFO] Video stopped.')

    def run(self):
        print('[INFO] Loading pretrained model...')

        if self.configs['run']['pretrained_model']:
            self.frvsr.load_weights(self.configs['run']['pretrained_model'])

        print('[INFO] Model ready.')
        print('[INFO] Running model...')

        rows = self.configs['run']['rows']
        cols = self.configs['run']['cols']

        prev_lr_frame = np.array([np.zeros((self.lr_shape[0] * rows, self.lr_shape[1] * cols, self.lr_shape[2]))])
        prev_est_frame = np.array([np.zeros((self.hr_shape[0] * rows, self.hr_shape[1] * cols, self.hr_shape[2]))])

        est_frame = np.array([np.zeros((self.hr_shape[0] * rows, self.hr_shape[1] * cols, self.hr_shape[2]))])

        sub_prev_lr_frames = np.repeat(np.array([np.zeros(self.lr_shape)]), rows * cols, axis=0)
        sub_est_frames = np.repeat(np.array([np.zeros(self.hr_shape)]), rows * cols, axis=0)

        t = 0
        t1 = datetime.datetime.now()

        if self.configs['run']['video']:
            cap = cv2.VideoCapture(self.configs['run']['video'])
        else:
            cap = cv2.VideoCapture(0)

        while cap.isOpened:
            img_windows = []
            img_window = np.array([])

            ret, frame = cap.read()
            if not ret:
                break
            frame = normalize(frame)

            hr_frame = cv2.resize(frame,
                                  (self.configs['data']['high_res'] * rows, self.configs['data']['high_res'] * cols),
                                  interpolation=cv2.INTER_CUBIC)

            lr_frame = np.array([cv2.resize(cv2.GaussianBlur(hr_frame, (5, 5), 0),
                                            (self.configs['data']['low_res'] * rows,
                                             self.configs['data']['low_res'] * cols),
                                interpolation=cv2.INTER_CUBIC)])

            sub_hr_frames = np.array([])
            sub_lr_frames = np.array([])
            for i in range(rows):
                for j in range(cols):
                    sub_lr_frame = lr_frame[0, i * self.configs['data']['low_res']: (i+1) * self.configs['data']['low_res'],
                                            j * self.configs['data']['low_res']: (j+1) * self.configs['data']['low_res']]

                    try:
                        sub_lr_frames = np.append(sub_lr_frames, [sub_lr_frame], axis=0)

                    except ValueError:
                        sub_lr_frames = np.array([sub_lr_frame])

            inference_time_1 = datetime.datetime.now()
            sub_est_frames, _ = self.frvsr.predict([sub_lr_frames,
                                                    sub_prev_lr_frames,
                                                    sub_est_frames])
            inference_time_2 = datetime.datetime.now()

            for i in range(rows):
                for j in range(cols):
                    est_frame[:, i * self.configs['data']['high_res']: (i+1) * self.configs['data']['high_res'], j * self.configs['data']['high_res']: (j+1) * self.configs['data']['high_res']] = sub_est_frames[i*cols + j]

            bicubic_frame = cv2.resize(lr_frame[0], (self.configs['data']['high_res'] * rows, self.configs['data']['high_res'] * cols),
                                       interpolation=cv2.INTER_CUBIC)

            sub_prev_lr_frames = sub_lr_frames

            nearest_frame = cv2.resize(lr_frame[0], (self.configs['data']['high_res'] * rows, self.configs['data']['high_res'] * cols),
                                       interpolation=cv2.INTER_NEAREST)

            border = 256 - 64

            lr_scale = cv2.copyMakeBorder(lr_frame[0], border, border, border, border, cv2.BORDER_CONSTANT)

            img_windows.append(lr_scale)
            img_windows.append(est_frame[0])
            img_windows.append(bicubic_frame)
            img_windows.append(nearest_frame)

            window_rows = 2
            window_cols = 2

            img_window = np.zeros((window_rows * self.hr_shape[0] * rows,
                                   window_cols * self.hr_shape[1] * cols,
                                   self.hr_shape[2]))

            # for img in img_windows:
            #     try:
            #         img_window = np.concatenate((img_window, img), axis=0)
            #     except ValueError:
            #         img_window = np.array(img)

            for i in range(window_rows):
                for j in range(window_cols):
                    img_window[self.hr_shape[0] * rows * i:self.hr_shape[0] * rows * (i+1), self.hr_shape[1]  * cols * j:self.hr_shape[1] * cols * (j+1)] = img_windows[i*window_cols + j]

            # img_window = np.concatenate((hr_frame, est_frame[0]), axis=1)
            # img_window = np.concatenate((img_window, bicubic_frame), axis=1)
            # img_window = np.concatenate((img_window, nearest_frame), axis=1)

            t2 = datetime.datetime.now()
            delta_t = t2 - t1
            t1 = datetime.datetime.now()

            fps = int(1 / delta_t.total_seconds())
            delta_inference = inference_time_2 - inference_time_1
            inference_time = delta_inference.total_seconds() * 1000

            gray_hr_frame = cv2.cvtColor(hr_frame, cv2.COLOR_BGR2GRAY)
            gray_nearest_frame = cv2.cvtColor(nearest_frame, cv2.COLOR_BGR2GRAY)
            gray_bicubic_frame = cv2.cvtColor(bicubic_frame, cv2.COLOR_BGR2GRAY)
            gray_est_frame = cv2.cvtColor(est_frame[0].astype('Float32'), cv2.COLOR_BGR2GRAY)

            if t == 100:
                print('\n[INFO] Running at: %d[fps] \t FRVSR inference time: %d[ms] \t Total inference time: %d[ms]' % (fps, inference_time, delta_t.total_seconds() * 1000))
                print('[INFO] Nearest PSNR: %f \t Bicubic PSNR: %f \t FRVSR PSNR: %f' % (psnr(nearest_frame, hr_frame), psnr(bicubic_frame, hr_frame), psnr(hr_frame, est_frame[0])))
                print('[INFO] Nearest SSIM: %f \t Bicubic SSIM: %f \t FRVSR SSIM: %f' % (ssim(gray_nearest_frame, gray_hr_frame, full=True)[0],
                                                                                         ssim(gray_bicubic_frame, gray_hr_frame, full=True)[0],
                                                                                         ssim(gray_hr_frame, gray_est_frame, full=True)[0]))
                t = 0
            t += 1

            text_window = 'FPS:%d' % fps

            window_title = 'Original / Frame-Recurrent Video Super Resolution / Bicubic / Nearest'
            cv2.putText(img_window, text_window, (0, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 1, 0), 1, cv2.LINE_4)

            cv2.imshow(window_title, img_window)


            key = cv2.waitKey(1)
            if key == ord('p'):
                while True:
                    cv2.imshow('Low Resolution Frame', lr_frame[0])
                    cv2.imshow(window_title, img_window)
                    if cv2.waitKey(1) == ord('p'):
                        break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print('\n[INFO] Video stopped.')
