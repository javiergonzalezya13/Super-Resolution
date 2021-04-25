import math
import os

import cv2
import keras.backend as K
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.models import Model
from skimage.measure import compare_ssim
from keras.layers import Flatten

# Check configs parameters
def check_yaml(configs):
    if not 'train' in configs:
        configs['stage']['train'] = False
    if not 'eval' in configs['stage']:
        configs['stage']['eval'] = False
    if not 'run' in configs:
        configs['stage']['run'] = False
    if not 'rec' in configs:
        configs['stage']['rec'] = False

# Get videos list from configs
def get_videos(configs):
    videos = []
    configs['data']['videos'] = configs['data']['videos'].replace(' ', '').split(',')
    for data_dir in configs['data']['videos']:
        for f in os.listdir(data_dir):
            if os.path.isfile(os.path.join(data_dir, f)):
                videos.append(os.path.join(data_dir, f))
    return videos

# Get c consecutive frames from the video
def get_frames(video, configs):
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    valid_frames = False
    while not valid_frames:
        frame_n = np.random.randint(0, total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
        hr_c_frames = []
        for _ in range(configs['data']['c_frames']):
            ret, hr_frame = cap.read()
            # Video completed
            if not ret:
                break
            hr_frame = cv2.resize(hr_frame,
                                  (configs['data']['high_res'], configs['data']['high_res']),
                                  interpolation=cv2.INTER_CUBIC)
            hr_c_frames.append(hr_frame)
        if ret:
            valid_frames = True
    cap.release()
    return hr_c_frames

# Normalize to [0, 1]
def normalize(input_data):
    return input_data.astype(np.float32) / 255.0

# Denormalize to [0, 255]
def denormalize(input_data):
    input_data = input_data * 255.0
    return input_data.astype(np.uint8)

# Peak Signal-to-Noise Ratio
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 1.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))

# Structural Similarity Index
def ssim(img1, img2):
    img1 = cv2.cvtColor(img1.astype('Float32'), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2.astype('Float32'), cv2.COLOR_BGR2GRAY)
    return compare_ssim(img1, img2, full=True)[0]

# Feature map metric
class Vgg(object):
    def __init__(self, hr_shape):
        self.hr_shape = hr_shape
        
    def compare(self, img1, img2):
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.hr_shape)
        vgg19.trainable = False
        for l in vgg19.layers:
            l.trainable = False
        self.model54 = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        self.model54.trainable = False

        img1 = np.array([img1])
        img2 = np.array([img2])
        m1 = self.model54.predict(img1)[0]
        m2 = self.model54.predict(img2)[0]
        mean = K.eval(K.mean(K.square(m2-m1))).item()

        K.clear_session()

        return mean
