import train
import test
from utils import *
from argparse import ArgumentParser
import tensorflow as tf
import glob
import scipy.io
from argparse import ArgumentParser
import time

os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

conf = tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction = 0.96

'''
USA:96,240
EU:136,224
China:136,240
s2s_USA:192,480
cfsr_USA:48,120
EC:60 148
'''

# EU
# HEIGHT = 136
# WIDTH = 224

# China
# HEIGHT = 136
# WIDTH = 240

#s2s_USA
# HEIGHT = 192
# WIDTH = 480

#cfsr_USA
# HEIGHT = 48
# WIDTH = 120

#ec
# HEIGHT = 60
# WIDTH = 148

#era5_2020
#era5_2000
#GFS025
HEIGHT = 96
WIDTH = 240

#GFS050
# HEIGHT = 48
# WIDTH = 120

#new_China
# HEIGHT = 100
# WIDTH = 120


# new_Europe
# HEIGHT = 68
# WIDTH = 132

# China_yinzi
# HEIGHT = 96
# WIDTH = 240

BATCH_SIZE =20
EPOCH = 300
LEARNING_RATE = 1e-4
CHANNEL = 3
CHECK_POINT_DIR = 'SR'
SCALE = 4
IS_TRANSFER = True
# TRANS_MODEL = '/hdd/tianchuan/Climate_work/MZSR-USA/SR2/Model3/model-100000'
# TRANS_MODEL = '/hdd/tianchuan/Climate_work/MZSR-USA/Large-Scale_Training/Merge_SR/Model1/model-1000000'
# TRANS_MODEL = '/hdd/tianchuan/Climate_w.ork/MZSR-USA/Large-Scale_Training/Merge_SR/Model3/model-1000000'
# TRANS_MODEL = '/hdd/tianchuan/Climate_work/MZSR-USA/SR2/Model5/model-100000'

#消融实验
# TRANS_MODEL ='/hdd/tianchuan/Climate_work/MZSR-USA/Large-Scale_Training/Merge_SR/Model6/model-1000000'
# TRANS_MODEL ='/hdd/tianchuan/Climate_work/MZSR-USA/Large-Scale_Training/Merge_SR/Model6/model-2000000'
TRANS_MODEL ='/hdd/tianchuan/Climate_work/MZSR-USA/Large-Scale_Training/Merge_SR/Model6/model-3000000'

'24'
# TRANS_MODEL = '/hdd/tianchuan/Climate_work/MZSR-USA/SR2/Model7/model-100000' #MTL
# TRANS_MODEL = '/hdd/tianchuan/Climate_work/MZSR-USA/Large-Scale_Training/Merge_SR/Model5/model-3000000' #EDSR
# TRANS_MODEL = '/hdd/tianchuan/Climate_work/MZSR-USA/SR2/Model8/model-100000' #MZSR
# TRANS_MODEL = '/hdd/tianchuan/Climate_work/MZSR-USA/Large-Scale_Training/Merge_SR/Model4/model-1000000' #deepsd

#phire
#deepsd
#edsr
#mtl



model_name='EDSR'
# TASK = ['mwp','sp','sst','swh']
# TASK = ['d2m', 'msl', 'mwd', 'ssrc', 'tp', 't2m', 'u10', 'v10']  #
# TASK = ['slp','sp','st','t2m','u10','v10']
# TASK = ['t2m','d2m','sp','tp','u10','v10']
# TASK = ['msl','ssrc','sst']
# data = ['d2m', 'msl', 'mwd', 'mwp', 'tp', 't2m', 'u10', 'v10', 'sp', 'sst', 'swh']  #
# data = ['mwd', 'mwp',  'sst', 'swh']  #

# mode='era5_Europe' # GFS025
# data = 'd2m'
def build_parser(num,mode,data,flag):
    parser = ArgumentParser()
    parser.add_argument('--train', dest='is_train', default=False, action='store_true')
    parser.add_argument('--trial', type=int,dest='trial', help='Trial Number',metavar='trial', default=num)#zhh
    parser.add_argument('--gpu',dest='gpu_num', help='GPU Number',metavar='GPU_NUM', default='0')
    parser.add_argument('--step', type=int,dest='global_step', help='Global Step',metavar='GLOBAL_STEP', default=0)

    # TEST
    # parser.add_argument('--gtpath', type=str, dest='gtpath',
    #                     default='/hdd/tianchuan/Meteorological_data/era5hourUSA/{}/Test'.format(data))

    if flag==True:
        parser.add_argument('--gtpath', type=str, dest='gtpath',
                            default='/hdd/zhanghonghu/{}/Test/{}/HR/'.format(mode,data))
        parser.add_argument('--imgpath', type=str, dest='imgpath',
                            default='/hdd/zhanghonghu/{}/Test/{}/LR/'.format(mode,data))
    else:
        #ec、
        parser.add_argument('--gtpath', type=str, dest='gtpath',
                            default='/hdd/tianchuan/Meteorological_data/Data_x4/fine-test/{}/Test/{}/HR/'.format(mode,data))
        parser.add_argument('--imgpath', type=str, dest='imgpath',
                            default='/hdd/tianchuan/Meteorological_data/Data_x4/fine-test/{}/Test/{}/LR/'.format(mode,data))

    parser.add_argument('--kernelpath', type=str, dest='kernelpath', default='')
    # parser.add_argument('--savepath', type=str, dest='savepath', default='results/{}/{}'.format(model_name,data))
    parser.add_argument('--savepath', type=str, dest='savepath', default='/hdd/tianchuan/Desktop/MODEL/MZSR/')

    parser.add_argument('--model', type=int, dest='model', choices=[0, 1, 2, 3], default=2)
    parser.add_argument('--num', type=int, dest='num_of_adaptation', choices=[1, 10], default=1)

    return parser

def main():
    zhh1 = ['era5_Europe', 'era5_China', 's2s_USA', 'cfsr_USA', 'ec', 'era5_2020', 'GFS050', 'data_5new', 'China_yinzi',
            'kua_10']
    mtl = [900, 928, 956, 968, 1000, 1024, 1150, 1200, 1235, 1305]
    edsr = [907, 935, 959, 975, 1006, 1030, 1156, 1205, 1250, 1315]
    deepsd = [914, 942, 962, 982, 1012, 1036, 1162, 1210, 1275, 1325]
    duibi = [921, 949, 965, 989, 1018, 1042, 1168, 1215, 1290, 1335]
    xx = duibi

    # mode = 'era5_Europe'
    # datas = ['d2m', 'sp', 't2m', 'tp', 'u10', 'v10', 'msl']  #
    # HEIGHT = 136
    # WIDTH = 224
    # num = xx[0]
    # flag = True

    # mode = 'era5_China'
    # datas = ['d2m', 'sp', 't2m', 'tp', 'u10', 'v10', 'msl']
    # HEIGHT = 136
    # WIDTH = 240
    # num=xx[1]
    # flag=True

    # mode = 's2s_USA'
    # datas = ['d2m','t2m','sst']
    # HEIGHT = 192
    # WIDTH = 480
    # num = xx[2]
    # flag=True

    mode = 'cfsr_USA'
    datas = ['d2m', 'pwat', 'sp', 't2m', 'tp', 'u10', 'v10']
    HEIGHT = 48
    WIDTH = 120
    num = xx[3]
    flag=True

    # mode='ec'
    # datas=['slp','sp','st','t2m','u10','v10']
    # HEIGHT = 60
    # WIDTH = 148
    # num = xx[4]
    # flag=False

    # mode='era5_2020'
    # datas=['d2m','sp','t2m','tp','u10','v10']
    # HEIGHT = 96
    # WIDTH = 240
    # num = xx[5]
    # flag=False

    # mode='GFS050'
    # datas=['d2m','sp','t2m','tp','u10','v10']
    # HEIGHT = 48
    # WIDTH = 120
    # num = xx[6]
    # flag=False

    #
    # mode = 'data_5new'
    # datas = ['ci', 'lh', 'ms2', 'rm2', 'vv']
    # HEIGHT = 100
    # WIDTH = 120
    # num = xx[7]
    # flag=True

    # datas = ['u100','v100','cp','cvh','lsp','lcc','msl','msr','mtpr','skt','rsn','slhf','sshf','tcrw','p63.162']
    # mode='China_yinzi'
    # HEIGHT = 96
    # WIDTH = 240
    # num = xx[8]
    # flag=True

    # datas = [str(i) for i in range(10)]
    # mode='kua_10'
    # HEIGHT = 100
    # WIDTH = 120
    # num = xx[9]
    # flag=True


    # mode='era5_Europe'
    # datas = ['d2m', 'sp', 't2m', 'tp', 'u10', 'v10', 'msl']  #

    # mode = 'era5_Europe'
    # datas = ['mwd', 'mwp',  'swh','sst']

    # mode = 'era5_China'
    # datas = ['d2m', 'sp', 't2m', 'tp', 'u10', 'v10', 'msl']

    # mode = 'era5_China'
    # datas = ['mwd', 'mwp',  'swh','sst']

    # mode = 's2s_USA'
    # datas = ['d2m','t2m','sst']

    # mode = 'cfsr_USA'
    # datas = ['d2m', 'pwat', 'sp', 't2m', 'tp', 'u10', 'v10']

    # mode='ec'
    # datas=['slp','sp','st','t2m','u10','v10']


    # mode='era5_2020'
    # datas=['d2m','sp','t2m','tp','u10','v10']

    # mode='era5_2000'
    # datas=['d2m','sp','t2m','tp','u10','v10']

    # mode='GFS025'
    # datas=['d2m','sp','t2m','tp','u10','v10']

    # mode='GFS050'
    # datas=['d2m','sp','t2m','tp','u10','v10']

    # mode='data_Europe'
    # datas=['ci','lh','ms2','rm2','vv']

    # datas = ['u100','v100','cp','cvh','lsp','lcc','msl','msr','mtpr','skt','rsn','slhf','sshf','tcrw','p63.162']
    # mode='China_yinzi'

    # datas = [str(i) for i in range(10)]
    # mode='kua_10'

    result=''
    for data in datas:
        tf.reset_default_graph()#zhh
        num=num+1    #zhh
        parser = build_parser(num,mode,data,flag)#zhh
        options = parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_num
        if flag==True:
            TF_RECORD_PATH  = ['/hdd/tianchuan/Meteorological_data/Data_x4/fine-test/{}/Finetune/{}.tfrecord'.format(mode,data)]
        else:
            #ec、
            TF_RECORD_PATH = ['/hdd/tianchuan/Meteorological_data/Data_x4/fine-test/{}/Finetune/{}/{}.tfrecord'.format(mode,data,data)]
        NUM_OF_DATA = 20
        if options.is_train == True:
            Trainer=train.Train(trial = options.trial,step=options.global_step,size=[HEIGHT,WIDTH,CHANNEL], batch_size=BATCH_SIZE,
                            learning_rate=LEARNING_RATE, max_epoch=EPOCH,tfrecord_path=TF_RECORD_PATH,checkpoint_dir=CHECK_POINT_DIR,
                            scale=SCALE,num_of_data=NUM_OF_DATA, conf=conf,IS_TRANSFER=IS_TRANSFER,TRANS_MODEL=TRANS_MODEL)
            Trainer.run()
        else:
            t1 = time.time()
            final_step = NUM_OF_DATA//BATCH_SIZE*EPOCH
            model_path = '/hdd/tianchuan/Climate_work/MZSR-USA/Retrain/SR/Model{}/model-{}'.format(options.trial,final_step)
            print('model_path',model_path)
            gt_path = sorted(glob.glob(os.path.join(options.gtpath, '*.png')))
            img_path = sorted(glob.glob(os.path.join(options.imgpath, '*.png')))
            scale = 4.0
            try:
                kernel = scipy.io.loadmat(options.kernelpath)['kernel']
            except:
                kernel = 'cubic'
            Tester = test.Test(model_path, options.savepath, kernel, scale, conf, options.model, options.num_of_adaptation)
            P = []
            S = []
            R = []
            for i in range(len(gt_path)):
                if i not in [0, 23, 43, 65, 79]:
                    continue
                gt = imread(gt_path[i])
                img = imread(img_path[i])
                _, pp,ss,rr = Tester(img, gt, gt_path[i],mode,data,'duibi',i)
                if pp[0]<100:
                    P.append(pp)
                    S.append(ss)
                    R.append(rr)
            t2 = time.time()
            print('num:',num)#zhh
            print('time:',t2-t1)
            avg_PSNR = np.mean(P, 0)
            avg_SSIM = np.mean(S, 0)
            avg_RMSE=np.mean(R,0)
            print('[*] Average PSNR ** Initial: %.4f' % (avg_PSNR[0]))
            print('[*] Average SSIM ** Initial: %.4f' % (avg_SSIM[0]))
            print('[*] Average RMSE ** Initial: %.4f' % (avg_RMSE[0]))
            p4='%.4f'%(avg_PSNR[0])
            s4='%.4f'%(avg_SSIM[0])
            r4='%.4f'%(avg_RMSE[0])
            result += p4 + ' ' + s4 + ' '+r4+' '
    print(num)
    print(result)
if __name__ == '__main__':
    main()


