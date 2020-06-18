import math
import os

import cv2
import numpy as np


def check_yaml(configs):
    if not 'train' in configs:
        configs['stage']['train'] = False
    if not 'eval' in configs['stage']:
        configs['stage']['eval'] = False
    if not 'run' in configs:
        configs['stage']['run'] = False
    if not 'rec' in configs:
        configs['stage']['rec'] = False

    

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

def normalize(input_data):
    return input_data.astype(np.float32) / 255.0

def denormalize(input_data):
    input_data = input_data * 255.0
    return input_data.astype(np.uint8)

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
