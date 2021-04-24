'''
Main code to use CNNs
'''
import argparse
import datetime
import os
import sys

import keras.backend as K
import tensorflow as tf
import yaml

# from EDVR import EDVR
# from EDVR_v2 import EDVR as EDVR_v2
from FRVSR import FrameRecurrentVideoSR
from TecoGAN import TecoGAN
from yoloV3_predict import YoloV3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    ap = argparse.ArgumentParser(description='Video Super-Resolution')

    ap.add_argument('--yaml_file', help='YAML file.')

    args = ap.parse_args()
    return args

if __name__ == '__main__':
    # Get yaml file parameters
    args = parse_args()
    with open(args.yaml_file) as file:
        configs = yaml.safe_load(file)

    # Shape parameters and upscale
    LR_SHAPE = (configs['data']['low_res'],
                configs['data']['low_res'],
                configs['data']['channels'])
    HR_SHAPE = (configs['data']['high_res'],
                configs['data']['high_res'],
                configs['data']['channels'])
    UPSCALE = configs['data']['upscale']

    if (LR_SHAPE[0] * UPSCALE != HR_SHAPE[0]) or (LR_SHAPE[1] * UPSCALE != HR_SHAPE[1]):
        print('[ERROR] Incompatible resolutions with upscale x%d.' % (UPSCALE))
        sys.exit(0)

    # Activate GPU or CPU mode
    if configs['gpu'] and K.tensorflow_backend._get_available_gpus():
        mode = '/gpu:0'
    else:
        if configs['gpu']:
            print('[INFO] No GPU available. Running on CPU ...')
        mode = '/cpu:0'

    # Create directory
    NOW = datetime.datetime.now()
    OUTPUT_DIR = '%d-%d-%d %d:%d:%d' % (NOW.year, NOW.month, NOW.day,
                                        NOW.hour, NOW.minute, NOW.second)

    OUTPUT_DIR = os.path.join(configs['root_dir'], 'MODEL', OUTPUT_DIR)

    with tf.device(mode):
        # Check neuronal network to use
        if configs['stage']['train'] or configs['stage']['eval'] or configs['stage']['run']:
            if configs['cnn']['model'] == 'frvsr':
                MODEL = FrameRecurrentVideoSR(LR_SHAPE, HR_SHAPE, OUTPUT_DIR, configs)
            elif configs['cnn']['model'] == 'tecogan':
                MODEL = TecoGAN(LR_SHAPE, HR_SHAPE, OUTPUT_DIR, configs)
            # elif configs['cnn']['model'] == 'edvr':
            #     MODEL = EDVR(LR_SHAPE, HR_SHAPE, OUTPUT_DIR, configs)
            # elif configs['cnn']['model'] == 'edvr_v2':
            #     MODEL = EDVR_v2(configs, OUTPUT_DIR, inp_shape=LR_SHAPE, nframes=5)
            else:
                print('[ERROR] Not valid model selected.')
                sys.exit(0)
            MODEL.build()

        # Train model
        if configs['stage']['train']:
            MODEL.train()

        # Evaluate model
        if configs['stage']['eval']:
            MODEL.eval()

        # Run single video or camera
        if configs['stage']['run']:
            MODEL.run()

        # Recognition with YOLO V3
        # K.clear_session()
        # if configs['stage']['rec']:
        #     YOLO_MODEL = YoloV3(configs)
        #     YOLO_MODEL.predict()
