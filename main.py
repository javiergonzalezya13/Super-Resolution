'''
Main code to use CNNs
'''
import argparse
import datetime
import os
import sys

import yaml

from FRVSR import *
from TecoGAN import *
from utils import generate_frames

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    ap = argparse.ArgumentParser(description='Frame-Recurrent Video Super Resolution')

    ap.add_argument('--yaml_file', help='YAML file.')

    args = ap.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    with open(args.yaml_file) as file:
        configs = yaml.safe_load(file)


    LR_shape = (configs['data']['low_res'],
                configs['data']['low_res'],
                configs['data']['channels'])
    HR_shape = (configs['data']['high_res'],
                configs['data']['high_res'],
                configs['data']['channels'])
    upscale = configs['data']['upscale']

    NOW = datetime.datetime.now()
    OUTPUT_DIR = '%d-%d-%d %d:%d:%d' % (NOW.year, NOW.month, NOW.day,
                                        NOW.hour, NOW.minute, NOW.second)

    if not configs['data']['videos']:
        print('[ERROR] No data available.')
        sys.exit(0)

    OUTPUT_DIR = os.path.join(configs['root_dir'], 'MODEL', OUTPUT_DIR)

    if (LR_shape[0] * upscale != HR_shape[0]) or (LR_shape[1] * upscale != HR_shape[1]):
        print('[ERROR] Incompatible resolutions with upscale x%d.' % (upscale))
        sys.exit(0)

    # MODEL
    if configs['stage']['train'] or configs['stage']['eval'] or configs['stage']['run']:
        if configs['model'] == 'frvsr':
            model = FrameRecurrentVideoSR(LR_shape, HR_shape, OUTPUT_DIR, configs)
        elif configs['model'] == 'tecogan':
            model = TecoGAN(LR_shape, HR_shape, OUTPUT_DIR, configs)
        else:
            print('[ERROR] Not valid model selected.')
            sys.exit(0)
        model.build()

    # TRAINING
    if configs['stage']['train']:
        model.train()

    # EVALUATION
    if configs['stage']['eval']:
        model.eval()

    # RUN
    if configs['stage']['run']:
        # model.run(configs)
        model.run()
