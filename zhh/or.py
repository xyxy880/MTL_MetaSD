import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import numpy as np
def save_jet_figure(image, filename):
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
    im = ax.imshow(image, 'seismic',vmax=10,vmin=-10)
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

def rgb2y(x):
    if x.dtype == np.uint8:
        x = np.float64(x)
        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16
        y = np.round(y).astype(np.uint8)
    else:
        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16 / 255
    return y


if __name__ == '__main__':

    mode = 'GFS050'
    datas = ['d2m', 'sp', 'tp', 't2m', 'u10', 'v10']
    num =0

    # 创建一个空列表来存储生成的子图
    subplots = []

    # 遍历数据和模型
    row=len(datas)
    for v in datas:
        for m in ['kelijing','Bicubic','DeepSD','EDSR','Phire','mtl']:
            SR = cv2.imread('/hdd/tianchuan/zhh_bias/SR/{}_{}_{}_{}.png'.format(mode, v, m, num))
            HR = cv2.imread('/hdd/tianchuan/zhh_bias/HR/{}_{}_{}_{}.png'.format(mode, v, m, num))
            HR = np.array(HR)
            SR = np.array(SR)

            # 计算差异并转换为灰度值
            diff = HR.astype('float32') - SR.astype('float32')
            image = rgb2y(diff)
            # 将差异图片添加到列表中
            subplots.append(image)

    fig,axs=plt.subplots(row, 6, figsize=(10, 10), sharex=True, sharey=True)
    fig.tight_layout(h_pad=-60,w_pad=-0.5)
    for i in range(row):
        for j in range(6):
            if j==0:
                axs[i][j].set_ylabel(i)
            if i==5:
                axs[i][j].set_xlabel(j)
            im=axs[i][j].imshow(subplots[i*6+j], 'seismic', vmax=10, vmin=-10)
            axs[i][j].axis('off')
            cax = axs[0][0]

    fc=fig.colorbar(im,ax=axs,pad=0.05)
    ax2=fc.ax
    fig.savefig('/hdd/tianchuan/zhh_bias/bias{}.png'.format(mode),
                bbox_inches='tight',
                pad_inches=0,
                transparent=True,
                dpi=300)
    plt.show()

