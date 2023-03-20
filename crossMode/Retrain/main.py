import train
from utils import *
from argparse import ArgumentParser
import test
import glob
import scipy.io
from argparse import ArgumentParser
import time
from imresize import imresize


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

conf = tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction = 0.96


HEIGHT = 12
WIDTH = 18
# BATCH_SIZE =20
# EPOCH = 10
BATCH_SIZE =1
EPOCH = 200
LEARNING_RATE = 1e-4
CHANNEL = 3
CHECK_POINT_DIR = 'SR'
SCALE = 3
IS_TRANSFER = True

# TRANS_MODEL ='/hdd/tianchuan/Climate_work/MZSR-USA/crossMode/crossMode_SR/Model0/model-100000' #MTL
# TRANS_MODEL ='/hdd/tianchuan/Climate_work/MZSR-USA/crossMode/Merge_SR/Model0/model-500000' #EDSR
# TRANS_MODEL ='/hdd/tianchuan/Climate_work/MZSR-USA/crossMode/Merge_SR/Model1/model-200000' #DeepSD
# TRANS_MODEL ='/hdd/tianchuan/Climate_work/MZSR-USA/crossMode/crossMode_SR/Model1/model-100000' #MZSR


# 先era5再wrf
# TRANS_MODEL ='/hdd/tianchuan/Climate_work/MZSR-USA/crossMode/crossMode_SR/Model4/model-100000' #MZSR
TRANS_MODEL ='/hdd/tianchuan/Climate_work/MZSR-USA/crossMode/crossMode_SR/Model2/model-100000' #MTL


model_name='EDSR'
# WRF = ['Wind','T2','SST','TSK','LH',]
era5 = ['wind','t2m','sst','skt','slhf',]
# EC = ['slp', 'st', 't2m', 'wind']

data = 'wind'
'''
WRF:
    0-4:MTL+pre
    5--9:EDSR
    10-14:DeepSD
    15-19:MZSR
    20-24:MTL-fine
ERA5:
    25-29:edsr
    30-34:MTL dtr:era5

35：edsr merge 测试wrf

EC：
    37：MTL在era5上训练的模型
    38-42：edsr测试
    43-46:deepsd
    47-50:mzsr
    
    51-55:MTL(先era5，再wrf)

'''
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--train', dest='is_train', default=True, action='store_true')
    parser.add_argument('--trial', type=int,dest='trial', help='Trial Number',metavar='trial', default=140)
    parser.add_argument('--gpu',dest='gpu_num', help='GPU Number',metavar='GPU_NUM', default='7')
    parser.add_argument('--step', type=int,dest='global_step', help='Global Step',metavar='GLOBAL_STEP', default=0)

    # TEST
    parser.add_argument('--gtpath', type=str, dest='gtpath',
                        default='/hdd/zhanghonghu/code/torch1/data/bhw/{}/'.format(data))
    # parser.add_argument('--imgpath', type=str, dest='imgpath',
    #                     default='/hdd/tianchuan/Meteorological_data/wrf_era5/wrf/{}/LR_test'.format(data))

    parser.add_argument('--kernelpath', type=str, dest='kernelpath', default='')
    # parser.add_argument('--savepath', type=str, dest='savepath', default='results/{}/{}'.format(model_name,data))
    parser.add_argument('--savepath', type=str, dest='savepath', default='results/mzsr_crossMode')

    parser.add_argument('--model', type=int, dest='model', choices=[0, 1, 2, 3], default=2)
    parser.add_argument('--num', type=int, dest='num_of_adaptation', choices=[1, 10], default=1)

    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_num
    # TF_RECORD_PATH = ['/hdd/tianchuan/Meteorological_data/wrf_era5/wrf/{}/finetune.tfrecord'.format(data)]
    # TF_RECORD_PATH = ['/hdd/tianchuan/Meteorological_data/wrf_era5/era5/{}/finetune.tfrecord'.format(data)]
    TF_RECORD_PATH = ['/hdd/tianchuan/Meteorological_data/wrf_era5/ec/{}_finetune.tfrecord'.format(data)]

    # NUM_OF_DATA = 200
    NUM_OF_DATA = 3


    if options.is_train == True:
        # print(TF_RECORD_PATH)
        Trainer=train.Train(trial = options.trial,step=options.global_step,size=[HEIGHT,WIDTH,CHANNEL], batch_size=BATCH_SIZE,
                        learning_rate=LEARNING_RATE, max_epoch=EPOCH,tfrecord_path=TF_RECORD_PATH,checkpoint_dir=CHECK_POINT_DIR,
                        scale=SCALE,num_of_data=NUM_OF_DATA, conf=conf,IS_TRANSFER=IS_TRANSFER,TRANS_MODEL=TRANS_MODEL)
        Trainer.run()
    # else:
        t1 = time.time()
        final_step = NUM_OF_DATA//BATCH_SIZE*EPOCH
        model_path = '/hdd/tianchuan/Climate_work/MZSR-USA/crossMode/Retrain/SR/Model{}/model-{}'.format(options.trial,final_step)
        # model_path=TRANS_MODEL
        print('model_path',model_path)

        gt_path = sorted(glob.glob(os.path.join(options.gtpath, '*.png')))
        # img_path = sorted(glob.glob(os.path.join(options.imgpath, '*.png')))

        scale = 3.0
        try:
            kernel = scipy.io.loadmat(options.kernelpath)['kernel']
        except:
            kernel = 'cubic'
        Tester = test.Test(model_path, options.savepath, kernel, scale, conf, options.model, options.num_of_adaptation)

        P = []
        S = []
        for i in range(len(gt_path)):
            gt = imread(gt_path[i])
            # img = imread(img_path[i])
            img = imresize(gt, scale=1. / scale, kernel='cubic')

            img_norm = np.zeros(img.shape)
            onedata = cv2.normalize(img, dst=img_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            onedata = np.uint8(onedata)
            imageio.imsave('test_{}.png'.format(data), onedata)
            img = imread('test_{}.png'.format(data))

            _, pp,ss = Tester(img, gt, gt_path[i])
            P.append(pp)
            S.append(ss)
        t2 = time.time()
        print('time:',t2-t1)

        avg_PSNR = np.mean(P, 0)
        avg_SSIM = np.mean(S, 0)
        print('[*] Average PSNR ** Initial: %.4f' % (avg_PSNR[0]))
        print('[*] Average SSIM ** Initial: %.4f' % (avg_SSIM[0]))

if __name__ == '__main__':
    main()


