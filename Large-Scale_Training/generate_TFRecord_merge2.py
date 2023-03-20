import random
import imageio
import os
import glob
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
import cv2
from imresize import imresize


def imread(path):
    img1 = imageio.imread(path)
    x, y, c = img1.shape
    if c == 3:
        return img1
    elif c == 4:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))


def write_to_tfrecord(writer, label, image):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
    }))
    writer.write(example.SerializeToString())
    return
def random_horizontal_flip(lr, hr):
    if random.random() < 0.5:
        lr = lr[:, ::-1, :].copy()
        hr = hr[:, ::-1, :].copy()
    return lr, hr


def random_vertical_flip(lr, hr):
    if random.random() < 0.5:
        lr = lr[::-1, :, :].copy()
        hr = hr[::-1, :, :].copy()
    return lr, hr

def random_rotate_90(hr):
    if random.random() < 0.5:
        hr = np.rot90(hr, axes=(1, 0)).copy()
    return hr

def generate_TFRecord(label_path1, label_path2, tfrecord_file, patch_h, patch_w, stride):
    vars = ['d2m', 'sp', 'tp', 't2m', 'u10', 'v10']
    label_list=[]
    img_list=[]
    for var in vars:
        a = ((glob.glob('/hdd/zhanghonghu/Data_x4/HR/era5_2000/{}/*'.format(var))))
        b = ((glob.glob('/hdd/zhanghonghu/Data_x4/HR/era5_2020/{}/*'.format(var))))
        c = ((glob.glob('/hdd/zhanghonghu/Data_x4/HR/GFS025/{}/*'.format(var))))
        d = ((glob.glob('/hdd/zhanghonghu/Data_x4/HR/GFS050/{}/*'.format(var))))
        label_list =label_list+ a+b+c+d
    for var in vars:
        a = ((glob.glob('/hdd/zhanghonghu/Data_x4/LR/era5_2000/{}/*'.format(var))))
        b = ((glob.glob('/hdd/zhanghonghu/Data_x4/LR/era5_2020/{}/*'.format(var))))
        c = ((glob.glob('/hdd/zhanghonghu/Data_x4/LR/GFS025/{}/*'.format(var))))
        d = ((glob.glob('/hdd/zhanghonghu/Data_x4/LR/GFS050/{}/*'.format(var))))
        img_list =label_list+ a+b+c+d

    offset = 0

    fileNum = len(label_list)

    labels = []
    patches=[]

    for n in range(fileNum):
        print('[*] Image number: %d/%d' % ((n + 1), fileNum))
        img = imread(img_list[n])
        label = imread(label_list[n])
        scale = 4
        assert os.path.basename(img_list[n]) == os.path.basename(label_list[n])

        # img = imresize(label, scale=1. / 8.0, kernel='cubic')
        # img_norm = np.zeros(img.shape)
        # onedata = cv2.normalize(img, dst=img_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # onedata = np.uint8(onedata)
        # imageio.imsave('lr.png', onedata)
        # lr = imread('lr.png')

        x, y, ch = label.shape
        for i in range(0, 96 - 48 + 1, 48):
            for j in range(0, 240 - 120 + 1, 120):
                patch_d = img[i // scale:i // scale + patch_h // scale, j // scale:j // scale + patch_w // scale]
                patch_l = label[i:i + patch_h, j:j + patch_w]

                lr, hr = random_horizontal_flip(patch_d, patch_l)
                lr, hr = random_vertical_flip(lr, hr)

                if hr.shape == (48,120,3) and lr.shape==(12,30,3):
                # print('YES')
                    patches.append(lr.tobytes())
                    labels.append(hr.tobytes())
                else:print('NO')
        # lr, hr = random_horizontal_flip(img, label)
        # lr, hr = random_vertical_flip(lr, hr)
        #
        # if hr.shape == (48, 120, 3) and lr.shape == (12, 30, 3):
        #     # print('YES')
        #     patches.append(lr.tobytes())
        #     labels.append(hr.tobytes())
        # else:
        #     print('NO')
    np.random.seed(36)
    np.random.shuffle(patches)
    np.random.seed(36)
    np.random.shuffle(labels)
    print('Num of patches:', len(patches))
    # print('Shape: [%d, %d, %d]' % (patch_h, patch_w, ch))

    writer = tf.io.TFRecordWriter(tfrecord_file)
    for i in range(len(patches)):
        write_to_tfrecord(writer, labels[i], patches[i])

    writer.close()

if __name__ == '__main__':
    vars = ['d2m','ssrc','msl','mwd','tp','t2m','u10','v10'] #'d2m','msl','mwd','tp','t2m','u10','v10'
    # for var in vars:
    #     print(var)
    var ='d2m'
    # '/hdd/zhanghonghu/Data_x4/HR/GFS050/v10'
    parser = ArgumentParser()
    parser.add_argument('--labelpath', dest='labelpath', help='Path to HR images (./DIV2K_train_HR)',
                        default='/hdd/tianchuan/Meteorological_data/era5hourUSA/{}/1980'.format(var))
    parser.add_argument('--labelpath2', dest='labelpath2', help='Path to HR images (./DIV2K_train_HR)',
                        default='/hdd/tianchuan/Meteorological_data/era5hourUSA/{}/1981'.format(var))
    parser.add_argument('--tfrecord', dest='tfrecord', help='Save path for tfrecord file',
                        default='/hdd/tianchuan/Meteorological_data/Data_x4/24tasks')
    options = parser.parse_args()

    labelpath1 = os.path.join(options.labelpath, '*.png')
    labelpath2 = os.path.join(options.labelpath2, '*.png')

    tfrecord_file = options.tfrecord + '.tfrecord'

    generate_TFRecord(labelpath1, labelpath2, tfrecord_file, 48, 120, 120)
    print('Done')
