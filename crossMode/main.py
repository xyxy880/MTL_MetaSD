import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from crossMode import dataGenerator
from crossMode import train
import test
from utils import *
from crossMode.config import *
import glob
import scipy.io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

conf = tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction = 0.96


def main():
    if args.is_train == True:
        data_generator = dataGenerator.dataGenerator(output_shape=[HEIGHT, WIDTH, CHANNEL],
                                                     meta_batch_size=META_BATCH_SIZE,
                                                     task_batch_size=TASK_BATCH_SIZE, tfrecord_path=TFRECORD_PATH)

        Trainer = train.Train(trial=args.trial, step=args.step, size=[HEIGHT, WIDTH, CHANNEL],
                              scale_list=SCALE_LIST, meta_batch_size=META_BATCH_SIZE, meta_lr=META_LR,
                              meta_iter=META_ITER, task_batch_size=TASK_BATCH_SIZE,
                              task_lr=TASK_LR, task_iter=TASK_ITER, data_generator=data_generator,
                              checkpoint_dir=CHECKPOINT_DIR, conf=conf)

        Trainer()
    else:
        model_path = '/hdd/tianchuan/Climate_work/MZSR-USA/SR2/Model3/model-100000'

        # img_path = sorted(glob.glob(os.path.join(args.inputpath, '*.png')))
        gt_path = sorted(glob.glob(os.path.join(args.gtpath, '*.png')))

        scale = 8.0

        try:
            kernel = scipy.io.loadmat(args.kernelpath)['kernel']
        except:
            kernel = 'cubic'

        Tester = test.Test(model_path, args.savepath, kernel, scale, conf, args.model, args.num_of_adaptation)
        P = []
        S = []
        for i in range(len(gt_path)):
            gt = imread(gt_path[i])
            img = imresize(gt, scale=1. / scale, kernel='cubic')
            img_norm = np.zeros(img.shape)
            onedata = cv2.normalize(img, dst=img_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            onedata = np.uint8(onedata)
            imageio.imsave('test_lr.png', onedata)
            img = imread('test_lr.png')

            _, pp, ss = Tester(img, gt, gt_path[i])
            P.append(pp)
            S.append(ss)
        avg_PSNR = np.mean(P, 0)
        avg_SSIM = np.mean(S, 0)

        print('Average PSNR ** Initial: %.4f' % (avg_PSNR[0]))
        print('Average SSIM ** Initial: %.4f' % (avg_SSIM[0]))


if __name__ == '__main__':
    main()
