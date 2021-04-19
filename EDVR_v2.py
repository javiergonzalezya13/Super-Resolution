import numpy as np
from keras.layers import *
from keras import backend as K
from subpixel import *
from keras.models import Model
from keras.optimizers import Adam
import os
import sys
import cv2
from utils import *
# Paper's loss function
def charbonnier_penalty(y_true, y_pred):
    return K.sum(K.sqrt(1e-6 + K.square(y_pred - y_true)))

class EDVR:
        
        
    def __init__(self, configs, output_dir, inp_shape=(256, 256, 3), nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None, predeblur=False, HR_in=False):
        self.H, self.W, self.C = inp_shape
        self.is_predeblur = True if predeblur else False
        self.center = nframes // 2 if center is None else center
        self.nf = nf
        self.nframes = nframes
        self.groups = groups
        self.front_RBs = front_RBs
        self.back_RBs = back_RBs
        self.HR_in = HR_in
        self.configs = configs
        self.output_dir = output_dir
        self.lr_shape = inp_shape
        self.hr_shape = (inp_shape[0] * 4, inp_shape[1] * 4, inp_shape[2])
        
    def __ResidualBlock_noBN(self, x):
        identity = x
        out = Conv2D(self.nf, kernel_size=3, padding='same', activation='relu')(x)
        out = Conv2D(self.nf, kernel_size=3, padding='same')(x)
        return add([identity, out])
    
    def __conv_block(self, x, stride=1):
        x = Conv2D(self.nf, 3, strides=stride, padding='same')(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x

    def __get_center_layer(self, x):
        center_layer = Lambda(lambda x: x[:, self.center, :, :, :])(x)
        return center_layer
    
    def __Predeblur_ResNet_Pyramid(self, x):
        L1_fea = self.__conv_block(x)
        if self.HR_in:
            for i in range(2):
                L1_fea = self.__conv_block(L1_fea, 2)
        L2_fea = self.__conv_block(L1_fea, 2)
        L3_fea = self.__conv_block(L2_fea, 2)
        L3_fea = self.__ResidualBlock_noBN(L3_fea)
        L3_fea = UpSampling2D(interpolation='bilinear')(L3_fea)
        L2_fea = add([self.__ResidualBlock_noBN(L2_fea), L3_fea])
        L2_fea = self.__ResidualBlock_noBN(L2_fea)
        L2_fea = UpSampling2D(interpolation='bilinear')(L2_fea)
        L1_fea = add([self.__ResidualBlock_noBN(L1_fea), L2_fea])
        out = self.__ResidualBlock_noBN(L1_fea)
        for i in range(2):
            out = self.__ResidualBlock_noBN(out)
        return out

    def __PCD_Align(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,H,W,C] features
        '''
        # L3
        L3_offset = Concatenate()([nbr_fea_l[2], ref_fea_l[2]])
        for _ in range(2):
            L3_offset = Conv2D(self.nf, 3, padding='same')(L3_offset)
            L3_offset = LeakyReLU(alpha=.1)(L3_offset)

        # Deformable Conv Layer Should be here and take (nbr_fea_l[2], L3_offset) as input
        L3_fea = Conv2D(self.nf, 3, padding='same')(L3_offset)
        #     L3_fea = DeformableConvLayer(nf, 3, strides=1, 
        #                                  padding='same', dilation_rate=1, 
        #                                  num_deformable_group=groups)(nbr_fea_l[2], L3_offset)

        # L2
        L2_offset = Concatenate()([nbr_fea_l[1], ref_fea_l[1]])
        
        L2_offset = self.__conv_block(L2_offset)
        L3_offset = UpSampling2D(interpolation='bilinear')(L3_offset)
        L3_offset = Lambda(lambda x: x*2)(L3_offset)
        concat_offset = Concatenate()([L2_offset, L3_offset])
        
        L2_offset = self.__conv_block(concat_offset)
        L2_offset = self.__conv_block(L2_offset)

        # Deformable Conv Layer Should be here and take (nbr_fea_l[1], L2_offset) as input
        L2_fea = Conv2D(self.nf, 3, padding='same')(L2_offset)
        #     L2_fea = DeformableConvLayer(nf, 3, strides=1, 
        #                                  padding='same', dilation_rate=1, 
        #                                  num_deformable_group=groups)(nbr_fea_l[1], L2_offset)
        L3_fea = UpSampling2D(interpolation='bilinear')(L3_fea)
        concat_fea = Concatenate()([L2_fea, L3_fea])
        L2_fea = self.__conv_block(concat_fea)

        # L1
        L1_offset = Concatenate()([nbr_fea_l[0], ref_fea_l[0]])
        L1_offset = self.__conv_block(L1_offset)
        L2_offset = UpSampling2D(interpolation='bilinear')(L2_offset)
        L2_offset = Lambda(lambda x: x*2)(L2_offset)
        concat_offset = Concatenate()([L1_offset, L2_offset])
        L1_offset = self.__conv_block(concat_offset)
        L1_offset = self.__conv_block(L1_offset)

        # Deformable Conv Layer Should be here and take (nbr_fea_l[0], L1_offset) as input
        L1_fea = Conv2D(self.nf, 3, padding='same')(L1_offset)
        #     L1_fea = DeformableConvLayer(nf, 3, strides=1, 
        #                                  padding='same', dilation_rate=1, 
        #                                  num_deformable_group=groups)(nbr_fea_l[0], L1_offset)
        L2_fea = UpSampling2D(interpolation='bilinear')(L2_fea)
        concat_fea = Concatenate()([L1_fea, L2_fea])
        L1_fea = self.__conv_block(concat_fea)

        # Cascading
        offset = Concatenate()([L1_fea, ref_fea_l[0]])
        for _ in range(2):
            offset = self.__conv_block(offset)
        # Deformable Conv Layer Should be here and take (L1_fea, offset) as input
        L1_fea = Conv2D(self.nf, 3, padding='same')(offset)
        #     L1_fea = DeformableConvLayer(nf, 3, strides=1, 
        #                                  padding='same', dilation_rate=1, 
        #                                  num_deformable_group=groups)(L1_fea, offset)
        L1_fea = LeakyReLU(alpha=.1)(L1_fea)
        return L1_fea

    def __TSA_Fusion(self, aligned_fea):
        ''' Temporal Spatial Attention fusion module
        Temporal: correlation;
        Spatial: 3 pyramid levels.
        '''
        B, N, H, W, C = K.int_shape(aligned_fea)
        #### temporal attention

        emb_ref = Conv2D(self.nf, 3, padding='same', name='TSA_ref')(self.__get_center_layer(aligned_fea))
        reshaped_fea = Lambda(lambda x: K.reshape(x, (-1, H, W, C)))(aligned_fea)
        reshaped_emb = Conv2D(self.nf, 3, padding='same')(reshaped_fea)
        emb = Lambda(lambda x: K.reshape(x, (-1, N, H, W, C)))(reshaped_emb)
        cor_l = []
        for i in range(N):
            emb_nbr = Lambda(lambda x: x[:, i, :, :, :])(emb)
            m_emb = Multiply()([emb_nbr, emb_ref])
            cor_tmp = Lambda(lambda x: K.sum(x, axis=-1))(m_emb)
            cor_tmp = Lambda(lambda x: K.expand_dims(x, -1))(cor_tmp)
            cor_l.append(cor_tmp)
        cor_prob = Lambda(lambda x: K.sigmoid(x))(Concatenate()(cor_l))
        cor_prob = Lambda(lambda x: K.expand_dims(x, axis=-1))(cor_prob)
        cor_prob = Concatenate()(C*[cor_prob])
        cor_prob = Lambda(lambda x: K.reshape(x, (-1, H, W, C*N)))(cor_prob)
        aligned_fea = Lambda(lambda x: K.reshape(x, (-1, H, W, C*N)))(aligned_fea)
        aligned_fea = Multiply(name='aligned_features')([aligned_fea, cor_prob])

        #### fusion
        fea = Conv2D(self.nf, 1, name='TSA_fusion')(aligned_fea)
        fea = LeakyReLU(alpha=.1)(fea)
        #### spatial attention
        att = Conv2D(self.nf, 1, name='Spatial_attention')(aligned_fea)
        att = LeakyReLU(alpha=.1)(att)
        att_max = Lambda(lambda x: K.spatial_2d_padding(x))(att)
        att_max = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(att_max)
        att_avg = Lambda(lambda x: K.spatial_2d_padding(x))(att)
        att_avg = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(att_avg)
        cat_max_avg = Concatenate()([att_max, att_avg])
        cat_max_avg = Conv2D(self.nf, 1, name='last_SA')(cat_max_avg)
        att = LeakyReLU(alpha=.1)(cat_max_avg)
        # pyramid levels
        att_L = Conv2D(self.nf, 1, name='Pyramid_levels')(att)
        att_L = LeakyReLU(alpha=.1)(att_L)
        att_max = Lambda(lambda x: K.spatial_2d_padding(x))(att_L)
        att_max = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(att_max)
        att_avg = Lambda(lambda x: K.spatial_2d_padding(x))(att_L)
        att_avg = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(att_avg)
        cat_max_avg = Concatenate()([att_max, att_avg])
        att_L = self.__conv_block(cat_max_avg)
        att_L = UpSampling2D(interpolation='bilinear')(att_L)
        att = self.__conv_block(att)
        att = add([att, att_L])
        att = Conv2D(self.nf, 1)(att)
        att = LeakyReLU(alpha=.1)(att)
        att = UpSampling2D(interpolation='bilinear')(att)
        att = Conv2D(self.nf, 3, padding='same')(att)
        att_add = Conv2D(self.nf, 1)(att)
        att_add = LeakyReLU(alpha=.1)(att_add)
        att_add = Conv2D(self.nf, 1)(att_add)
        att = Lambda(lambda x: K.sigmoid(x), name='pyramid_level_attributes')(att)
        fea = Lambda(lambda x: x[0]*x[1]*2 + x[2], name='pyramid_level_features')([fea, att, att_add])
        return fea
    
    def build(self):
        input_x = Input((self.nframes, self.H, self.W, self.C))
        x_center = self.__get_center_layer(input_x)
        x_reshaped = Lambda(lambda x: K.reshape(x, (-1, self.H, self.W, self.C)))(input_x)
        # L1
        if self.is_predeblur:
            L1_fea = self.__Predeblur_ResNet_Pyramid(x_reshaped)
            L1_fea = Conv2D(self.nf, 1, name='post_predeblur_conv')(L1_fea)
            if self.HR_in:
                self.H, self.W = self.H // 4, self.W // 4
        else:
            L1_fea = self.__conv_block(x_reshaped)
            if self.HR_in:
                for i in range(2):
                    L1_fea = self.__conv_block(L1_fea, 2)
                self.H, self.W = self.H // 4, self.W // 4
        for _ in range(self.front_RBs):
            L1_fea = self.__ResidualBlock_noBN(L1_fea)
        # L2
        L2_fea = self.__conv_block(L1_fea, 2)
        L2_fea = self.__conv_block(L2_fea)
        # L3
        L3_fea = self.__conv_block(L2_fea, 2)
        L3_fea = self.__conv_block(L3_fea)
        
        L1_fea = Lambda(lambda x: K.reshape(x, (-1, self.nframes, self.H, self.W, self.nf)))(L1_fea)
        L2_fea = Lambda(lambda x: K.reshape(x, (-1, self.nframes, self.H//2, self.W//2, self.nf)))(L2_fea)
        L3_fea = Lambda(lambda x: K.reshape(x, (-1, self.nframes, self.H//4, self.W//4, self.nf)))(L3_fea)

        #### pcd align
        # ref feature list
        ref_fea_l = [self.__get_center_layer(L1_fea), self.__get_center_layer(L2_fea), 
                     self.__get_center_layer(L3_fea)]

        aligned_fea = []
        for i in range(self.nframes):
            nbr_fea_l = [Lambda(lambda x: x[:, i, :, :, :])(L1_fea), 
                         Lambda(lambda x: x[:, i, :, :, :])(L2_fea),
                         Lambda(lambda x: x[:, i, :, :, :])(L3_fea)]
            aligned_fea.append(self.__PCD_Align(nbr_fea_l, ref_fea_l))

        aligned_fea = Lambda(lambda x: K.stack(x, axis=1), name='PCD_aligned_features')(aligned_fea)

        fea = self.__TSA_Fusion(aligned_fea)

        for _ in range(self.back_RBs):
            fea = self.__ResidualBlock_noBN(fea)

        out = Subpixel(self.nf, 3, 2, padding='same', name='subpixel1')(fea)
        out = LeakyReLU(alpha=.1)(out)
        out = Subpixel(64, 3, 2, padding='same', name='subpixel2')(out)
        out = LeakyReLU(alpha=.1)(out)
        out = Conv2D(64, 3, padding='same', name='HR_conv')(out) # HR conv
        out = LeakyReLU(alpha=.1)(out)
        out = Conv2D(3, 3, padding='same', name='last_conv')(out) # Conv last
        if self.HR_in:
            base = x_center
        else:
            base = UpSampling2D(size=(4, 4), interpolation='bilinear')(x_center)
        out = add([out, base], name='output')
        self.edvr = Model(input_x, out, name='EDVR')
        optimizer = Adam(lr=4e-4, beta_1=.9, beta_2=0.999)
        self.edvr.compile(optimizer=optimizer, loss=charbonnier_penalty)

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