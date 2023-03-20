from argparse import ArgumentParser

parser=ArgumentParser()

# Global
parser.add_argument('--gpu', type=str, dest='gpu', default='4')
data = ['mwp']

# For Meta-test
parser.add_argument('--inputpath', type=str, dest='inputpath', default='')
parser.add_argument('--gtpath', type=str, dest='gtpath', default='/hdd/tianchuan/Meteorological_data/era5hourUSA/mwp/Test')
parser.add_argument('--kernelpath', type=str, dest='kernelpath', default='')
parser.add_argument('--savepath', type=str, dest='savepath', default='results/')
parser.add_argument('--model', type=int, dest='model', choices=[0,1,2,3], default=2)
parser.add_argument('--num', type=int, dest='num_of_adaptation', choices=[1,10], default=1)


# For Meta-Training
parser.add_argument('--trial', type=int, dest='trial', default=4)
parser.add_argument('--step', type=int, dest='step', default=0)
parser.add_argument('--train', dest='is_train', default=True, action='store_true')

'''
0:MTL+pre  dtr：wrf  dte：era5
1:MZSR+pre   dtr：wrf  dte：era5

2:MTL+pre   dtr：era5  dte：wrf
3：MTL+pre TASK_LR=1e-3  TASK_ITER=3
4:MZSR+pre   dtr：era5  dte：wrf   

'''

args= parser.parse_args()

#Transfer Learning From Pre-trained model.
IS_TRANSFER = True
TRANS_MODEL = '/hdd/tianchuan/Climate_work/MZSR/Large-Scale_Training/SR/Model1/model-800000' #mzsr
# TRANS_MODEL = '/hdd/tianchuan/Climate_work/MZSR/Large-Scale_Training/SR/Model4/model-800000' #EDSR
# TRANS_MODEL ='/hdd/tianchuan/Climate_work/MZSR-v2/Large-Scale_Training/SR/Model5/model-800000' #MTL
# TRANS_MODEL ='/hdd/tianchuan/Climate_work/MZSR-v3/Large-Scale_Training/SR/Model57/model-796860' #DeepSD


# Dataset Options
HEIGHT=201
WIDTH=201
CHANNEL=3

# SCALE_LIST=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
SCALE_LIST=[3.0]

META_ITER=500000
META_BATCH_SIZE=5 #5个任务
META_LR=1e-4

TASK_ITER=5
TASK_BATCH_SIZE=4
TASK_LR=1e-2

# Loading tfrecord and saving paths
TFRECORD_PATH='train_SR_MZSR.tfrecord'
CHECKPOINT_DIR='crossMode_SR'
