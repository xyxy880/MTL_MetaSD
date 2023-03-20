import imageio
import os
import glob
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

def imread(path):
    img = imageio.imread(path)
    return img

def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))

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
    img_list = np.sort(np.asarray(glob.glob(os.path.join(data_path, 'LR_train' + '/*.png'))))

    offset=0

    fileNum=len(label_list)

    patches=[]
    labels=[]

    for n in range(fileNum):
        print('[*] Image number: %d/%d' % ((n+1), fileNum))
        img=imread(img_list[n])
        label=imread(label_list[n])

        assert os.path.basename(img_list[n]) == os.path.basename(label_list[n])

        img=modcrop(img,scale)
        label=modcrop(label,scale)

        x, y, ch = label.shape
        for i in range(0+offset,x-patch_h+1,stride):
            for j in range(0+offset,y-patch_w+1,stride):
                patch_d = img[i // scale:i // scale + patch_h // scale, j // scale:j // scale + patch_w // scale]
                patch_l = label[i:i + patch_h, j:j + patch_w]

            # if np.log(gradients(patch_l.astype(np.float64)/255.)+1e-10) >= -6.0:
                patches.append(patch_d.tobytes())
                labels.append(patch_l.tobytes())


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

if __name__=='__main__':
    TASK = ['Wind', 'OLR', 'PBLH', 'Q2', 'slp', 'T2', 'TH2', 'TSK']
    for data in TASK:
        parser = ArgumentParser()
        parser.add_argument('--scale', dest='scale', help='Scaling Factor for Super-Resolution', type=int, default=3)
        parser.add_argument('--labelpath', dest='labelpath',default='/hdd/tianchuan/Meteorological_data/DataSet/Retrain/{}/HR_train/'.format(data))
        parser.add_argument('--datapath', dest='datapath', default='/hdd/tianchuan/Meteorological_data/DataSet/Retrain/{}'.format(data))
        parser.add_argument('--tfrecord', dest='tfrecord', default='/hdd/tianchuan/Meteorological_data/DataSet/Retrain/{}500/'.format(data))
        options = parser.parse_args()

        if not os.path.exists(options.tfrecord):
            os.makedirs(options.tfrecord)

        scale = options.scale
        labelpath = os.path.join(options.labelpath, '*.png')
        datapath = options.datapath
        tfrecord_file = options.tfrecord + '{}.tfrecord'.format(data)

        # generate_TFRecord(datapath, labelpath, tfrecord_file,48*scale,48*scale,180)
        generate_TFRecord(datapath, labelpath, tfrecord_file, 42 * scale, 42 * scale, 90)

        print('Done')