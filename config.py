from argparse import ArgumentParser

parser=ArgumentParser()

# Global
parser.add_argument('--gpu', type=str, dest='gpu', default='1')

# mwp  sp  sst  swh
data = ['mwp']

# For Meta-test
parser.add_argument('--inputpath', type=str, dest='inputpath', default='')
parser.add_argument('--gtpath', type=str, dest='gtpath', default='/hdd/tianchuan/Meteorological_data/era5hourUSA/mwp/Test')
parser.add_argument('--kernelpath', type=str, dest='kernelpath', default='')
parser.add_argument('--savepath', type=str, dest='savepath', default='results/')
parser.add_argument('--model', type=int, dest='model', choices=[0,1,2,3], default=2)
parser.add_argument('--num', type=int, dest='num_of_adaptation', choices=[1,10], default=1)


# For Meta-Training
parser.add_argument('--trial', type=int, dest='trial', default=8)
parser.add_argument('--step', type=int, dest='step', default=0)
parser.add_argument('--train', dest='is_train', default=True, action='store_true')

'''
0:MTL
1:MZSR
2:EDSR
3:MTL+pre
4:EDSR+pre
5:MZSR+pre
6:deepsd+pre

'''

'''
24个任务：
7:MTL
8:MZSR

'''

args= parser.parse_args()

#Transfer Learning From Pre-trained model.
IS_TRANSFER = False
pre_MTL = '/hdd/tianchuan/Climate_work/MZSR-USA/Large-Scale_Training/SR/Model0/model-1000000'
pre_MZSR = '/hdd/tianchuan/Climate_work/MZSR-USA/Large-Scale_Training/SR/Model1/model-1000000'
pre_EDSR = '/hdd/tianchuan/Climate_work/MZSR-USA/Large-Scale_Training/SR/Model2/model-1000000'
pre_deepsd = '/hdd/tianchuan/Climate_work/MZSR-USA/Large-Scale_Training/SR/Model3/model-1000000'

TRANS_MODEL = pre_deepsd

# Dataset Options  美国96,120  张红湖 48,120
HEIGHT=48
WIDTH=120
CHANNEL=3

# SCALE_LIST=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
SCALE_LIST=[4.0]

META_ITER=200000
META_BATCH_SIZE=24 #美国8个任务 张红湖24
META_LR=1e-4

TASK_ITER=3
TASK_BATCH_SIZE=3
TASK_LR=1e-2

# Loading tfrecord and saving paths
TFRECORD_PATH='train_SR_MZSR.tfrecord'
CHECKPOINT_DIR='SR2'
