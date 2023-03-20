import matplotlib.image as image
import cv2
from tqdm import tqdm
import os
from pykrige import OrdinaryKriging
import numpy as np
import math
import sys, time
import os
import matplotlib
import numpy as np
from netCDF4 import Dataset
import glob
import matplotlib.image as img
# Interpolation kernel
def u(s,a):
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

#Paddnig
def padding(img,H,W,C):
    zimg = np.zeros((H+4,W+4,C))
    zimg[2:H+2,2:W+2,:C] = img
    #Pad the first/last two col and row
    zimg[2:H+2,0:2,:C]=img[:,0:1,:C]
    zimg[H+2:H+4,2:W+2,:]=img[H-1:H,:,:]
    zimg[2:H+2,W+2:W+4,:]=img[:,W-1:W,:]
    zimg[0:2,2:W+2,:C]=img[0:1,:,:C]
    #Pad the missing eight points
    zimg[0:2,0:2,:C]=img[0,0,:C]
    zimg[H+2:H+4,0:2,:C]=img[H-1,0,:C]
    zimg[H+2:H+4,W+2:W+4,:C]=img[H-1,W-1,:C]
    zimg[0:2,W+2:W+4,:C]=img[0,W-1,:C]
    return zimg

# https://github.com/yunabe/codelab/blob/master/misc/terminal_progressbar/progress.py
def get_progressbar_str(progress):
    END = 170
    MAX_LEN = 30
    BAR_LEN = int(MAX_LEN * progress)
    return ('Progress:[' + '=' * BAR_LEN +
            ('>' if BAR_LEN < MAX_LEN else '') +
            ' ' * (MAX_LEN - BAR_LEN) +
            '] %.1f%%' % (progress * 100.))

# Bicubic operation
def bicubic(img, ratio, a):
    #Get image size
    H,W,C = img.shape
    print('C',C)

    img = padding(img,H,W,C)
    #Create new image
    dH = math.floor(H*ratio)
    dW = math.floor(W*ratio)
    dst = np.zeros((dH, dW, 3))

    h = 1/ratio

    print('Start bicubic interpolation')
    print('It will take a little while...')
    inc = 0
    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                x, y = i * h + 2 , j * h + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                mat_l = np.matrix([[u(x1,a),u(x2,a),u(x3,a),u(x4,a)]])
                mat_m = np.matrix([[img[int(y-y1),int(x-x1),c],img[int(y-y2),int(x-x1),c],img[int(y+y3),int(x-x1),c],img[int(y+y4),int(x-x1),c]],
                                   [img[int(y-y1),int(x-x2),c],img[int(y-y2),int(x-x2),c],img[int(y+y3),int(x-x2),c],img[int(y+y4),int(x-x2),c]],
                                   [img[int(y-y1),int(x+x3),c],img[int(y-y2),int(x+x3),c],img[int(y+y3),int(x+x3),c],img[int(y+y4),int(x+x3),c]],
                                   [img[int(y-y1),int(x+x4),c],img[int(y-y2),int(x+x4),c],img[int(y+y3),int(x+x4),c],img[int(y+y4),int(x+x4),c]]])
                mat_r = np.matrix([[u(y1,a)],[u(y2,a)],[u(y3,a)],[u(y4,a)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m),mat_r)

                # Print progress
                inc = inc + 1
                sys.stderr.write('\r\033[K' + get_progressbar_str(inc/(C*dH*dW)))
                sys.stderr.flush()
    sys.stderr.write('\n')
    sys.stderr.flush()
    return dst
if __name__ == '__main__':

    for year in range(2021, 2022):
        a = Dataset('/hdd/mujialing/Data/era5hourFullUsa/{}.nc'.format(year))
        # print(a.variables)
        data = np.array(a.variables['t2m'][0, 0:96, 0:240])

        # Scale factor
        ratio = 1 / 8.0
        # Coefficient
        Coefficient = -1 / 8
        data1= data.reshape(96,240,1)
        dst = bicubic(data1, ratio, Coefficient)

        q=dst[0:12,0:30,0]
        lon = np.linspace(-130, -70.25, 30)
        lat = np.linspace(25.25, 49, 12)

        # lon = np.array((a.variables['longitude'][0:240]))
        # lat = np.array((a.variables['latitude'][0:96]))
        lon_list, lat_list, value_list = [], [], []
        for i in range(12):
            for j in range(30):
                value_list.append(q[i,j])
                lon_list.append(lon[j])
                lat_list.append(lat[i])
        n = np.array(value_list)
        lon = np.array(lon_list)
        lat = np.array(lat_list)
        # 克里金插值,zgrid是插值结果
        un_lon = np.linspace(-130, -70.25, 240)
        un_lat = np.linspace(25.25, 49, 96)  # 区域的经纬度范围，从67*67插值到201*201
        OK = OrdinaryKriging(lon, lat, n, variogram_model='exponential', nlags=6)
        zgrid, ss = OK.execute('grid', un_lon, un_lat)
        # #保存图片
        # img_name = '{}{}{}_{}_{}.png'.format(date[0], date[1], date[2],var, k - 6)
        matplotlib.image.imsave('/hdd/tianchuan/kriging_t2m.png', zgrid)


