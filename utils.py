import math

import numpy as np
import os
import cv2
def generate_frames(configs, output_dir):
    videos = [f for f in os.listdir(configs['data']['videos'])
              if os.path.isfile(os.path.join(configs['data']['videos'], f))]
    n_scenes = 0
    for n_video, video in enumerate(videos, start=0):
        print('[INFO] Loading video %d/%d ...' % (n_video + 1, len(videos)))
        cap = cv2.VideoCapture(os.path.join(configs['data']['videos'], video))
        i = 0
        while cap.isOpened:
            HR_c_frames = []
            LR_c_frames = []
            for _ in range(configs['data']['c_frames']):
                ret, HR_frame = cap.read()

                # Video completed
                if not ret:
                    break

                HR_frame = cv2.resize(HR_frame,
                                      (configs['data']['high_res'], configs['data']['high_res']),
                                      interpolation=cv2.INTER_CUBIC)
                HR_c_frames.append(HR_frame)

            # Video completed
            if not ret:
                break
            np.save(os.path.join(output_dir, 'HR_train_%d_%d' % (n_video, i)), HR_c_frames)
            i += 1
            n_scenes += 1

            if n_scenes % 50 == 0:
                print('[INFO] Scenes loaded: %d' % n_scenes)
        print('[INFO] Total scenes loaded: %d' % n_scenes)
    cap.release()
    return 0

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
