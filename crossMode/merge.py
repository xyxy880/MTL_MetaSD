import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import time
from crossMode import merge_train as train
from utils import *
from argparse import ArgumentParser
# import test
import glob
import scipy.io
from argparse import ArgumentParser

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

conf = tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction = 0.96

HEIGHT = 201
WIDTH = 201
CHANNEL = 3
BATCH_SIZE = 25
EPOCH = 20000000
LEARNING_RATE = 1e-4
CHECK_POINT_DIR = 'Merge_SR'
SCALE = 3

'''
0:edsr 训练集中有wrf、era5
1：deepsd 训练集中有wrf、era5
2:EDSR (重新训练) 训练era5
3:deepsd (重新训练) 训练era5
4:edsr (重新训练) 训练wrf
5:deepsd (重新训练) 训练wrf

'''
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--trial', type=int,dest='trial', help='Trial Number',metavar='trial', default=5)
    parser.add_argument('--gpu',dest='gpu_num', help='GPU Number',metavar='GPU_NUM', default='2')
    parser.add_argument('--step', type=int,dest='global_step', help='Global Step',metavar='GLOBAL_STEP', default=0)

    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_num

    NUM_OF_DATA = 19000  #div2k 43600 77088    WRF_merge 1919000
    TF_RECORD_PATH=['/hdd/tianchuan/Meteorological_data/wrf_era5/wrf/merge/merge.tfrecord']

    Trainer=train.Train(trial=options.trial,step=options.global_step,size=[HEIGHT,WIDTH,CHANNEL], batch_size=BATCH_SIZE,
                        learning_rate=LEARNING_RATE, max_epoch=EPOCH,tfrecord_path=TF_RECORD_PATH,checkpoint_dir=CHECK_POINT_DIR,
                        scale=SCALE,num_of_data=NUM_OF_DATA, conf=conf)
    Trainer.run()

if __name__ == '__main__':
    main()