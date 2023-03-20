import os

from skimage.metrics import peak_signal_noise_ratio as psnr_rgb
from skimage.metrics import structural_similarity as cal_ssim
from skimage.metrics import mean_squared_error as mse
import cv2
import numpy as np
import math
import matplotlib.image as image

# img1=cv2.imread('./slp0.png')
# img2=cv2.imread('./slp5.png')
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# print(psnr_rgb(img1,img2))
# print(cal_ssim(img1,img2,multichannel=True))




# zhh1=['era5_Europe','era5_China','s2s_USA','cfsr_USA','ec','era5_2020','GFS050','data_5new','China_yinzi','kua_10']
# mtl=[900,928,956,968,1000,1024,1150,1200,1235,1305]
# edsr=[907,935,959,975,1006,1030,1156,1205,1250,1315]
# deepsd=[914,942,962,982,1012,1036,1162,1210,1275,1325]
# duibi=[921,949,965,989,1018,1042,1168,1215,1290,1335]







# dir2s=['bhw','quanqiu']
# dir3s=['slp','sp','st','t2m','u10','v10','wind']
#
# dir2s=['ec']
# dir3s=['slp','sp','st','t2m','u10','v10']

# dir2s=['era5_1980','era5_2000','era5_2020','GFS025']
# dir2s=['GFS050']
# dir3s=['d2m','sp','t2m','tp','u10','v10']

# dir2s = ['data_5new']
# dir3s = ['ci', 'lh', 'ms2', 'rm2', 'vv']

# dir3s = ['u100', 'v100', 'cp', 'cvh', 'lsp', 'lcc', 'msl', 'msr', 'mtpr', 'skt', 'rsn', 'slhf', 'sshf', 'tcrw','p63.162']
# dir2s = ['China_yinzi']

# dir3s = [str(i) for i in range(10)]
# dir2s = ['kua_10']

# 21.6218 0.6636 21.3905 28.6008 0.7875 10.0272 30.7327 0.8617 7.4912 22.5965 0.7314 19.0469 28.5070 0.8235 9.7634


# dir2s=['cfsr_USA']
# dir3s=['d2m','pwat','sp','t2m','tp','u10','v10']
# dir3s=['d2m','sp','t2m','tp','u10','v10']

# dir2s=['era5_2020']
# dir3s=['d2m','sp','t2m','tp','u10','v10']
# #
# dir2s=['era5_2019']
# dir3s=['d2m','t2m','sst']

# dir2s=['era5_Europe']
# dir3s=['d2m','sp','t2m','tp','u10','v10','msl']
# dir3s=['mwd', 'mwp',  'swh','sst']

# dir2s=['era5_China']
# dir3s=['d2m','sp','t2m','tp','u10','v10','msl']
# dir3s=['mwd', 'mwp',  'swh','sst']

# dir2s = ['s2s_USA']
# dir3s= ['d2m','t2m','sst']












#
#
# mode = 's2s_USA'
# datas = ['d2m','t2m','sst']
# HEIGHT = 192
# WIDTH = 480
# lon1,lon2 = 70,130
# lat1,lat2 = 25,49
#
# mode = 'cfsr_USA'
# datas = ['d2m', 'pwat', 'sp', 't2m', 'tp', 'u10', 'v10']
# HEIGHT = 48
# WIDTH = 120
# lon1,lon2 = 70,130
# lat1,lat2 = 25,49
#
mode='ec'
datas=['slp','sp','st','t2m','u10','v10']
HEIGHT = 60
WIDTH = 148
lon1,lon2 = 70,130
lat1,lat2 = 25,49
#



# mode = 'data_5new'
# datas = ['ci', 'lh', 'ms2', 'rm2', 'vv']
# HEIGHT = 100
# WIDTH = 120
# lon1,lon2 = 73.66,135.05
# lat1,lat2 = 3.86,53.55
#
# datas = ['u100','v100','cp','cvh','lsp','lcc','msl','msr','mtpr','skt','rsn','slhf','sshf','tcrw','p63.162']
# mode='China_yinzi'  #美国的跨因子
# HEIGHT = 96
# WIDTH = 240
# lon1,lon2 = -130,-70
# lat1,lat2 = 25,49
#
# datas = [str(i) for i in range(10)]
# mode='kua_10'
# HEIGHT = 100
# WIDTH = 120
# lon1,lon2 = 73.66,135.05
# lat1,lat2 = 3.86,53.55


''' color conversion '''

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
def rgb2y(x):
    if x.dtype==np.uint8:
        x=np.float64(x)
        y=65.481/255.*x[:,:,0]+128.553/255.*x[:,:,1]+24.966/255.*x[:,:,2]+16
        y=np.round(y).astype(np.uint8)
    else:
        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16 /255

    return y
def cal_psnr(img1, img2):
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

if __name__=='__main__':
    result=''
    result_y=''
    h=HEIGHT
    w=WIDTH

    for dir2 in ['s']:
        dir2=mode
        for dir3 in datas:
            if dir3 == '5':
                result += str(0) + ' ' + str(0) + ' ' + str(0) + ' '
                continue
            # cur='./test/LR/' + dir2 + '/' + dir3
            # cur='/hdd/tianchuan/Meteorological_data/Data_x4/fine-test/{}/Test/{}/LR'.format(dir2,dir3)
            # y='/hdd/tianchuan/Meteorological_data/Data_x4/fine-test/{}/Test/{}/HR'.format(dir2,dir3)
            # cur = '/hdd/zhanghonghu/tian_x4/test/LR/{}/{}'.format(dir2, dir3)
            y = '/hdd/zhanghonghu/tian_x4/test/HR/{}/{}'.format(dir2, dir3)
            # cur = '/hdd/zhanghonghu/{}/Test/{}/LR'.format(dir2, dir3)
            # y = '/hdd/zhanghonghu/{}/Test/{}/HR'.format(dir2, dir3)
            paths=os.listdir(y)
            ssim=0
            psnr=0
            rmse=0
            PSNR_y=0
            tt=0
            for idx,path in enumerate(paths):
                img1=cv2.imread(y+'/'+path)
                # img1=cv2.resize(img1,None,fx=1/4,fy=1/4,interpolation=cv2.INTER_CUBIC)
                # img1=cv2.resize(img1,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
                data=cv2.imread(y+'/'+path,0)
                ratio = 1 / 4.0
                # Coefficient
                Coefficient = -1 / 4
                data1 = data.reshape(h, w, 1)
                dst = bicubic(data1, ratio, Coefficient)
                q = dst[0:h//4, 0:w//4, 0]
                lr=q
                lon = np.linspace(lon1, lon2, w//4)
                lat = np.linspace(lat1, lat2, h//4)

                # lon = np.array((a.variables['longitude'][0:240]))
                # lat = np.array((a.variables['latitude'][0:96]))
                lon_list, lat_list, value_list = [], [], []
                for i in range(h//4):
                    for j in range(w//4):
                        value_list.append(q[i, j])
                        lon_list.append(lon[j])
                        lat_list.append(lat[i])
                n = np.array(value_list)
                lon = np.array(lon_list)
                lat = np.array(lat_list)
                # 克里金插值,zgrid是插值结果
                un_lon = np.linspace(lon1, lon2, w)
                un_lat = np.linspace(lat1, lat2, h)  # 区域的经纬度范围，从67*67插值到201*201
                OK = OrdinaryKriging(lon, lat, n, variogram_model='exponential', nlags=6)
                zgrid, ss = OK.execute('grid', un_lon, un_lat)
                # #保存图片
                # img_name = '{}{}{}_{}_{}.png'.format(date[0], date[1], date[2],var, k - 6)

                # matplotlib.image.imsave('/hdd/tianchuan/kriging_t2m.png', zgrid)





                img1=zgrid












                img2=cv2.imread(y+'/'+path)
                # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                # psnr += psnr_rgb(img1, img2)

                # zhh保存图片
                if idx in [0, 23, 43, 65, 79]:
                    matplotlib.image.imsave('/hdd/tianchuan/zhh_bias/LR/{}_{}_{}_{}'.format(dir2,dir3, 'kelijing', idx) + '.png', lr)
                    matplotlib.image.imsave('/hdd/tianchuan/zhh_bias/SR/{}_{}_{}_{}'.format(dir2,dir3, 'kelijing', idx) + '.png', img1)
                    cv2.imwrite('/hdd/tianchuan/zhh_bias/HR/{}_{}_{}_{}'.format(dir2,dir3, 'kelijing', idx) + '.png', img2)

                matplotlib.image.imsave('/hdd/tianchuan/zhh_bias/1.png',img1)
                SR=cv2.imread('/hdd/tianchuan/zhh_bias/1.png')


                SR = np.round(np.clip((SR) * 255, 0., 255.)).astype(np.uint8)
                HR = np.round(np.clip((img2) * 255, 0., 255.)).astype(np.uint8)
                if psnr_rgb(SR,HR)>1000:
                    continue
                psnr += psnr_rgb(SR,HR)

                # PSNR_y += cal_psnr(rgb2y(SR),rgb2y(HR))
                ssim += cal_ssim(SR,HR, multichannel=True)
                rmse += mse(SR,HR)**0.5
                tt+=1
            p4='%.4f'%(psnr/(tt))
            s4='%.4f'%(ssim/(tt))
            r4='%.4f'%(rmse/(tt))
            p_y = '%.4f' % (PSNR_y / (tt))

            print(dir2+' '+dir3+' avg',tt,' psnr:',p4)
            print(dir2+' '+dir3+' avg',tt,' ssim:',s4)
            result+=str(p4)+' '+str(s4)+' '+str(r4)+' '
            result_y+=str(p_y)+' '+str(s4)+' '




    print(result)
    # print(result_y)