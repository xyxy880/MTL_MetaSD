import time
import merge_train as train
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

HEIGHT = 48
WIDTH = 120
CHANNEL = 3
BATCH_SIZE = 10
EPOCH = 20000000
LEARNING_RATE = 1e-4
CHECK_POINT_DIR = 'Merge_SR'
SCALE = 4

'''
0:MTL
1:edsr
2：mzsr
3：deepsd
'''

'''
24个任务：
4：DeepSD
5:edsr
'''
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--trial', type=int,dest='trial', help='Trial Number',metavar='trial', default=5)
    parser.add_argument('--gpu',dest='gpu_num', help='GPU Number',metavar='GPU_NUM', default='1')
    parser.add_argument('--step', type=int,dest='global_step', help='Global Step',metavar='GLOBAL_STEP', default=1000000)

    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_num

    NUM_OF_DATA = 241800  #div2k 43600 77088  美国8个任务280704  24个任务241800
    TF_RECORD_PATH=['/hdd/tianchuan/Meteorological_data/Data_x4/24tasks.tfrecord']
    #
    # NUM_OF_DATA = 34954
    # TF_RECORD_PATH=['/hdd/tianchuan/Meteorological_data/train_SR_x3.tfrecord']

    # NUM_OF_DATA = 280704
    # TF_RECORD_PATH = ['/hdd/tianchuan/Meteorological_data/era5hourUSA/8tasks.tfrecord']

    Trainer=train.Train(trial=options.trial,step=options.global_step,size=[HEIGHT,WIDTH,CHANNEL], batch_size=BATCH_SIZE,
                        learning_rate=LEARNING_RATE, max_epoch=EPOCH,tfrecord_path=TF_RECORD_PATH,checkpoint_dir=CHECK_POINT_DIR,
                        scale=SCALE,num_of_data=NUM_OF_DATA, conf=conf)
    Trainer.run()

if __name__ == '__main__':
    main()