import imageio
import os
import glob
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
import random
import cv2
def imread(path):
    img1=imageio.imread(path)
    x,y,c=img1.shape
    if c==3: return img1
    elif c==4:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))
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


def random_rotate_90(lr, hr):
    if random.random() < 0.5:
        lr = np.rot90(lr, axes=(1, 0)).copy()
        hr = np.rot90(hr, axes=(1, 0)).copy()
    return lr, hr
def random_crop(lr, hr, size, scale):
    lr_left = random.randint(0, lr.shape[1] - size)
    lr_right = lr_left + size
    lr_top = random.randint(0, lr.shape[0] - size)
    lr_bottom = lr_top + size
    hr_left = lr_left * scale
    hr_right = lr_right * scale
    hr_top = lr_top * scale
    hr_bottom = lr_bottom * scale
    lr = lr[lr_top:lr_bottom, lr_left:lr_right]
    hr = hr[hr_top:hr_bottom, hr_left:hr_right]
    return lr, hr
def modcrop(imgs, modulo):
    sz=imgs.shape
    sz=np.asarray(sz)
    if len(sz)==2:
        sz = sz - sz% modulo
        out = imgs[0:sz[0], 0:sz[1]]
    elif len(sz)==3:
        szt = sz[0:2]
        szt = szt - szt % modulo
        out = imgs[0:szt[0], 0:szt[1],:]

    return out

def write_to_tfrecord(writer, label, image):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
    }))
    writer.write(example.SerializeToString())
    return

def generate_TFRecord(data_path,label_path,tfrecord_file,patch_h,patch_w,stride):
    label_list=np.sort(np.asarray(glob.glob(label_path)))
    img_list = np.sort(np.asarray(glob.glob(os.path.join(data_path, 'LR_finetune' + '/*.png'))))

    offset=0

    fileNum=len(label_list)

    patches=[]
    labels=[]

    for n in range(fileNum):
        print('[*] Image number: %d/%d' % ((n+1), fileNum))
        img=imread(img_list[n])
        label=imread(label_list[n])

        assert os.path.basename(img_list[n]) == os.path.basename(label_list[n])

        x, y, ch = label.shape
        lr, hr = random_crop(img, label, 42, 3)
        lr, hr = random_horizontal_flip(lr, hr)
        lr, hr = random_vertical_flip(lr, hr)
        lr, hr = random_rotate_90(lr, hr)
        patches.append(lr.tobytes())
        labels.append(hr.tobytes())


    np.random.seed(36)
    np.random.shuffle(patches)
    np.random.seed(36)
    np.random.shuffle(labels)
    # print('Num of patches:', len(patches))
    # print('Shape: [%d, %d, %d]' % (patch_h, patch_w, ch))

    writer = tf.io.TFRecordWriter(tfrecord_file)
    for i in range(len(patches)):
        write_to_tfrecord(writer, labels[i], patches[i])

    writer.close()

if __name__=='__main__':
    # TASK = ['Wind', 'OLR', 'PBLH', 'Q2', 'slp', 'T2', 'TH2', 'TSK','GRDFLX','HFX','QFX','SST','LH']
    # TASK = ['T2','Wind','GRDFLX', 'HFX', 'QFX', 'SST', 'LH']
    TASK =['5Tasks']
    for data in TASK:
        print(data)
        parser = ArgumentParser()
        parser.add_argument('--scale', dest='scale', help='Scaling Factor for Super-Resolution', type=int, default=3)
        parser.add_argument('--labelpath', dest='labelpath',default='/hdd/tianchuan/Meteorological_data/DataSet_RGB/{}/HR_finetune/'.format(data))
        parser.add_argument('--datapath', dest='datapath', default='/hdd/tianchuan/Meteorological_data/DataSet_RGB/{}'.format(data))
        parser.add_argument('--tfrecord', dest='tfrecord', default='/hdd/tianchuan/Meteorological_data/DataSet_RGB/{}/'.format(data))
        options = parser.parse_args()

        if not os.path.exists(options.tfrecord):
            os.makedirs(options.tfrecord)

        scale = options.scale
        labelpath = os.path.join(options.labelpath, '*.png')
        datapath = options.datapath
        tfrecord_file = options.tfrecord + '{}_finetune.tfrecord'.format(data)

        # generate_TFRecord(datapath, labelpath, tfrecord_file,48*scale,48*scale,180)
        generate_TFRecord(datapath, labelpath, tfrecord_file, 42 * scale, 42 * scale, 90)

        print('Done')