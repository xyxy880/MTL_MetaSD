import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import numpy as np


def save_jet_figure(image, filename,i):
    ''' save jet colormap figure using matplotlib

    Args:
    ---

        image: ndarray of (width, height)
            input image in gray scale.

        filename: str
            save filename

    Returns:
    ---

        None

    Example:
    ---

        from skimage.io import imread
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        img = imread('./test.png')
        img = img[:, :, 0]
        save_jet_figure(img, './result.png')

        plt.show()

    '''
    # show image
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    im = ax.imshow(image, 'seismic', vmax=80, vmin=-80)

    ax.axis('off')


    filename = (filename.split('.'))[0]
    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.04)
    cbar = plt.colorbar(im,
                        cax=cax,
                        extend='both',
                        extendrect=True,
                        # ticks=[-40, -20,0,20,40]
                        )
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=12,
                        width=0,
                        length=0,
                        pad=1, )

    # save image
    fig.savefig(filename,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True,
                dpi=300)
    plt.show()


def rgb2y(x):
    if x.dtype == np.uint8:
        x = np.float64(x)
        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16
        y = np.round(y).astype(np.uint8)
    else:
        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16 / 255
    return y


if __name__ == '__main__':
    mode = 'era5_2020'
    datas = ['d2m', 'sp', 'tp', 't2m', 'u10', 'v10']
    mode='ec'
    datas=['slp','sp','st','t2m','u10','v10']
    # [0, 23, 43, 65, 79]
    num = 0

    # 创建一个空列表来存储生成的子图
    subplots = []

    # 遍历数据和模型
    for v in datas:
        for m in ['mtl','EDSR', 'DeepSD', 'Bicubic']:

            # 读取SR和HR图片并转换为numpy数组
            SR = cv2.imread('/hdd/tianchuan/zhh_bias/SR/{}_{}_{}_{}.png'.format(mode, v, m, num))
            HR = cv2.imread('/hdd/tianchuan/zhh_bias/HR/{}_{}_{}_{}.png'.format(mode, v, m, num))
            HR = np.array(HR)
            SR = np.array(SR)

            # 计算差异并转换为灰度值
            diff = HR.astype('float32') - SR.astype('float32')
            image = rgb2y(diff)


            # 将差异图片添加到列表中

            subplots.append(image)

    # 创建一个4x4的GridSpec对象，并指定水平和垂直间距为0.05
    fig = plt.figure()

    gs = fig.add_gridspec(6,5, hspace=0.05, wspace=0.05)

    # 在右侧创建一个新的子图用于放置颜色条，并指定宽度为5%和间距为0.04
    cax = fig.add_subplot(gs[:, -1])

    # 遍历列表中的差异图片，并在对应位置创建子图
    for i in range(len(subplots)):
        # 计算行号和列号（从零开始）
        row = i //4
        col = i % 4

        # 在对应位置创建子图，并指定颜色映射为seismic，最大值为80，最小值为-80
        ax = fig.add_subplot(gs[row, col])

        im = ax.imshow(subplots[i], 'seismic', vmax=80, vmin=-80)

        # 去掉坐标轴标签和刻度线
        ax.axis('off')

    # 调整整张图片与边框的距离，并指定透明度为True

    fig.savefig('/hdd/tianchuan/bias.png', bbox_inches='tight', pad_inches=0, transparent=True)

    # 在新创建的子图上添加一个颜色条，并指定扩展方式为both，去掉外框和刻度线，并设置shrink参数为0.5

    cbar = plt.colorbar(im, cax=cax, extend='both', extendrect=True, shrink=0.05)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(width=0, length=0)

    plt.show()