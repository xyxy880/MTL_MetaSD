import random
import imageio
import os
import glob
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
import cv2
from imresize import imresize

def augmentation(x, mode):
    if mode == 0:
        y = x

    elif mode == 1:
        y = np.flipud(x)

    elif mode == 2:
        y = np.rot90(x, 1)

    elif mode == 3:
        y = np.rot90(x, 1)
        y = np.flipud(y)

    elif mode == 4:
        y = np.rot90(x, 2)

    elif mode == 5:
        y = np.rot90(x, 2)
        y = np.flipud(y)

    elif mode == 6:
        y = np.rot90(x, 3)

    elif mode == 7:
        y = np.rot90(x, 3)
        y = np.flipud(y)

    return y


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

def generate_TFRecord(label_path1, tfrecord_file, patch_h, patch_w, stride):
    label_list = np.sort(np.asarray(glob.glob(label_path1)))

    offset = 0

    fileNum = len(label_list)

    labels = []
    patches=[]

    for n in range(fileNum):
        print('[*] Image number: %d/%d' % ((n + 1), fileNum))
        label = imread(label_list[n])

        img = imresize(label, scale=1. / 3.0, kernel='cubic')
        img_norm = np.zeros(img.shape)
        onedata = cv2.normalize(img, dst=img_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        onedata = np.uint8(onedata)
        imageio.imsave('lr.png', onedata)
        lr = imread('lr.png')

        x, y, ch = img.shape

        lr, hr = random_horizontal_flip(lr, label)
        lr, hr = random_vertical_flip(lr, hr)
        # lr,hr=lr,label
        if hr.shape == (21,21,3) and lr.shape==(7,7,3):
            # print('YES')
            patches.append(lr.tobytes())
            labels.append(hr.tobytes())
        else:print('NO')
    np.random.seed(36)
    np.random.shuffle(patches)
    np.random.seed(36)
    np.random.shuffle(labels)
    print('Num of patches:', len(patches))
    print('Shape: [%d, %d, %d]' % (patch_h, patch_w, ch))

    writer = tf.io.TFRecordWriter(tfrecord_file)
    for i in range(len(patches)):
        write_to_tfrecord(writer, labels[i], patches[i])

    writer.close()


if __name__ == '__main__':
    # data = ['mwp','sp','sst','swh'] #
    data = ['wind','t2m','sst','slhf','skt']
    # data = ['mwd','mwp','swh'] #


    for var in data:
        print(var)
        parser = ArgumentParser()
        parser.add_argument('--labelpath', dest='labelpath', help='Path to HR images (./DIV2K_train_HR)',
                            default='/hdd/tianchuan/Meteorological_data/wrf_era5/era5/{}/HR_finetune/'.format(var))
        parser.add_argument('--tfrecord', dest='tfrecord', help='Save path for tfrecord file',
                            default='/hdd/tianchuan/Meteorological_data/wrf_era5/era5/{}/finetune'.format(var,var))
        options = parser.parse_args()

        labelpath1 = os.path.join(options.labelpath, '*.png')
        # labelpath1 = os.path.join('/hdd/tianchuan/Meteorological_data/era5_test_wrf/wind/finetune/', '*.png')
        #
        tfrecord_file = options.tfrecord + '.tfrecord'
        # tfrecord_file ='/hdd/tianchuan/Meteorological_data/era5_test_wrf/wind/finetune' + '.tfrecord'

        generate_TFRecord(labelpath1, tfrecord_file, 96, 240, 120)
        print('Done')
