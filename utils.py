import imageio
import os
import numpy as np
import tensorflow as tf
import re
import math
from imresize import imresize

from time import strftime, localtime

import cv2

# def imread(path):
#     img=imageio.imread(path).astype(np.float32)
#     img=img/255.
#     return img

def imread(path):
    img1=imageio.imread(path).astype(np.float32)
    x,y,c=img1.shape
    if c== 4:
        img = cv2.imread(path).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img/255.
    else:return img1/255.

def load(saver, sess, checkpoint_dir, folder):
    print('==================== Reading Checkpoints ====================')
    checkpoint = os.path.join(checkpoint_dir, folder)

    ckpt= tf.train.get_checkpoint_state(checkpoint)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint, ckpt_name))

        step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))

        print("=================== Success to load {} ====================".format(ckpt_name))
        return True, step
    else:
        print("=================== Fail to find a Checkpoint ====================")
        return False, 0


def save(saver, sess, checkpoint_dir, trial, step):
    model_name='model'
    checkpoint = os.path.join(checkpoint_dir, 'Model%d'% trial)

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    saver.save(sess, os.path.join(checkpoint, model_name), global_step=step)

def count_param(scope=None):
    N=np.sum([np.prod(v.get_shape().as_list()) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)])
    print('Model Params: %d K' % (N/1000))

def psnr(img1, img2):
    img1=np.float32(img1)
    img2=np.float32(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    if np.max(img1) <= 1.0:
        PIXEL_MAX= 1.0
    else:
        PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def print_time():
    print('Time: ', strftime('%b-%d %H:%M:%S', localtime()))

''' color conversion '''
def rgb2y(x):
    if x.dtype==np.uint8:
        x=np.float64(x)
        y=65.481/255.*x[:,:,0]+128.553/255.*x[:,:,1]+24.966/255.*x[:,:,2]+16
        y=np.round(y).astype(np.uint8)
    else:
        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16 /255

    return y
# ----------
# SSIM
# ----------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def modcrop(imgs, modulo):
    sz=imgs.shape
    sz=np.asarray(sz)

    if len(sz)==2:
        sz = sz - sz% modulo
        out = imgs[0:int(sz[0]), 0:int(sz[1])]
    elif len(sz)==3:
        szt = sz[0:2]
        szt = szt - szt % modulo
        out = imgs[0:int(szt[0]), 0:int(szt[1]),:]

    return out

def back_projection(y_sr, y_lr, down_kernel, up_kernel, sf=None, ds_method='direct'):
    y_sr += imresize(y_lr - imresize(y_sr, scale=1.0/sf, output_shape=y_lr.shape, kernel=down_kernel, ds_method=ds_method),
                     scale=sf,
                     output_shape=y_sr.shape,
                     kernel=up_kernel)
    return np.clip(y_sr, 0, 1)
