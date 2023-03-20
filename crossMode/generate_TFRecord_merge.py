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
    label_list=np.sort(np.asarray(glob.glob(label_path1)))
    # img_list=np.sort(np.asarray(glob.glob(label_path2)))
    offset = 0
    fileNum = len(label_list)

    labels = []
    patches=[]
    scale=3

    for n in range(fileNum):
        print('[*] Image number: %d/%d' % ((n + 1), fileNum))
        # img = imread(img_list[n])
        label = imread(label_list[n])

        img = imresize(label, scale=1. / 3.0, kernel='cubic')
        img_norm = np.zeros(img.shape)
        onedata = cv2.normalize(img, dst=img_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        onedata = np.uint8(onedata)
        imageio.imsave('lr.png', onedata)
        img = imread('lr.png')

        lr, hr = random_horizontal_flip(img, label)
        lr, hr = random_vertical_flip(lr, hr)
        if hr.shape == (21, 21, 3) and lr.shape == (7, 7, 3):
            patches.append(lr.tobytes())
            labels.append(hr.tobytes())
        else:
            print('NO')


        # if( os.path.basename(img_list[n]) == os.path.basename(label_list[n]) ):
        #     lr, hr = random_horizontal_flip(img, label)
        #     lr, hr = random_vertical_flip(lr, hr)
        #     if hr.shape == (201, 201, 3) and lr.shape == (67, 67, 3):
        #         patches.append(lr.tobytes())
        #         labels.append(hr.tobytes())
        #     else:
        #         print('NO')

            # x, y, ch = label.shape
            # if(x==201):
            #     for i in range(0 + offset, x - patch_h + 1, stride):
            #         for j in range(0 + offset, y - patch_w + 1, stride):
            #             patch_l = label[i:i + patch_h, j:j + patch_w]
            #             patch_d = img[i // scale:i // scale + patch_h // scale, j // scale:j // scale + patch_w // scale]
            #             lr, hr = random_horizontal_flip(patch_d, patch_l)
            #             lr, hr = random_vertical_flip(lr, hr)
            #             if hr.shape == (21, 21, 3) and lr.shape == (7, 7, 3):
            #                 patches.append(lr.tobytes())
            #                 labels.append(hr.tobytes())
            #             else:print('NO')
            # else:
            #     lr, hr = random_horizontal_flip(img, label)
            #     lr, hr = random_vertical_flip(lr, hr)
            #     if hr.shape == (21, 21, 3) and lr.shape == (7, 7, 3):
            #         patches.append(lr.tobytes())
            #         labels.append(hr.tobytes())
            #     else:
            #         print('NO')
        # else:print(os.path.basename(img_list[n]),os.path.basename(label_list[n]))

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
    # vars = ['Wind','TSK','T2','SST','LH'] #'d2m','msl','mwd','tp','t2m','u10','v10'
    # vars = ['wind', 't2m', 'sst', 'skt', 'slhf']
    vars = ['Wind', 'T2', 'SST', 'TSK', 'LH']

    # for var in vars:
    #     print(var)
    # var ='d2m'
    parser = ArgumentParser()
    parser.add_argument('--labelpath', dest='labelpath', help='Path to HR images (./DIV2K_train_HR)',
                        default='/hdd/tianchuan/Meteorological_data/wrf_era5/era5/merge/HR/')
    parser.add_argument('--labelpath2', dest='labelpath2', help='Path to HR images (./DIV2K_train_HR)',
                        default='/hdd/tianchuan/Meteorological_data/wrf_era5/era5/merge/LR/')
    parser.add_argument('--tfrecord', dest='tfrecord', help='Save path for tfrecord file',
                        default='/hdd/tianchuan/Meteorological_data/wrf_era5/era5/merge/merge')
    options = parser.parse_args()

    labelpath1 = os.path.join(options.labelpath, '*.png')
    labelpath2 = os.path.join(options.labelpath2, '*.png')

    tfrecord_file = options.tfrecord + '.tfrecord'

    generate_TFRecord(labelpath1, labelpath2, tfrecord_file, 21, 21, 20)
    print('Done')
