import argparse
import datetime
import os
import sys
from utils import generate_frames
import yaml
from TecoGAN import *
from FRVSR import *
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

    now = datetime.datetime.now()
    output_dir = '%d-%d-%d %d:%d:%d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    data_output_dir = os.path.join(configs['root_dir'], 'DATASET', output_dir)
    output_dir = os.path.join(configs['root_dir'], 'MODEL', output_dir)

    if (LR_shape[0] * upscale != HR_shape[0]) or (LR_shape[1] * upscale != HR_shape[1]):
        print('[ERROR] Incompatible resolutions with upscale x%d.' % (upscale))
        sys.exit(0)

    # DATASET
    if not configs['data']['load_data'] and configs['data']['create_data']:
        os.makedirs(data_output_dir)
        print('[INFO] Creating dataset...')
        generate_frames(configs, data_output_dir)
    elif not configs['data']['create_data'] and configs['data']['load_data']:
        print('[INFO] Loading dataset...')
        if isinstance(configs['data']['data_dir'], list):
            for idx, data_dir in enumerate(configs['data']['data_dir']):
                configs['data']['data_dir'][idx] = os.path.join(configs['root_dir'], data_dir)
        else:
            configs['data']['data_dir'] = os.path.join(configs['root_dir'], configs['data']['data_dir'])
    else:
        print('[INFO] No dataset available.')

    # MODEL
    if configs['stage']['train'] or configs['stage']['eval'] or configs['stage']['run']:
        if configs['model'] == 'frvsr':
            model = FrameRecurrentVideoSR(LR_shape, HR_shape, output_dir, configs)
        elif configs['model'] == 'tecogan':
            model = TecoGAN(LR_shape, HR_shape, output_dir, configs)
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
        model.run_testing()