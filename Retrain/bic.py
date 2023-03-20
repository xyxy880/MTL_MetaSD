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

dir2s = ['data_5new']
dir3s = ['ci', 'lh', 'ms2', 'rm2', 'vv']

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

dir2s=['era5_China']
dir3s=['tp']
# dir3s=['d2m','sp','t2m','tp','u10','v10','msl']
# dir3s=['mwd', 'mwp',  'swh','sst']

# dir2s = ['s2s_USA']
# dir3s= ['d2m','t2m','sst']

''' color conversion '''
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
result=''
result_y=''

for dir2 in dir2s:
    for dir3 in dir3s:
        if dir3 == '5':
            result += str(0) + ' ' + str(0) + ' ' + str(0) + ' '
            continue
        # cur='./test/LR/' + dir2 + '/' + dir3
        # cur='/hdd/tianchuan/Meteorological_data/Data_x4/fine-test/{}/Test/{}/LR'.format(dir2,dir3)
        # y='/hdd/tianchuan/Meteorological_data/Data_x4/fine-test/{}/Test/{}/HR'.format(dir2,dir3)
        # cur = '/hdd/zhanghonghu/tian_x4/test/LR/{}/{}'.format(dir2, dir3)
        # y = '/hdd/zhanghonghu/tian_x4/test/HR/{}/{}'.format(dir2, dir3)
        cur = '/hdd/zhanghonghu/{}/Test/{}/LR'.format(dir2, dir3)
        y = '/hdd/zhanghonghu/{}/Test/{}/HR'.format(dir2, dir3)
        paths=os.listdir(cur)
        ssim=0
        psnr=0
        rmse=0
        PSNR_y=0
        tt=0
        for idx,path in enumerate(paths):
            img1=cv2.imread(cur+'/'+path)
            lr=img1
            # img1=cv2.resize(img1,None,fx=1/4,fy=1/4,interpolation=cv2.INTER_CUBIC)
            img1=cv2.resize(img1,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)



            img2=cv2.imread(y+'/'+path)
            # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            # psnr += psnr_rgb(img1, img2)

            # zhh保存图片
            if idx in [0, 23, 43, 65, 79]:
                cv2.imwrite('/hdd/tianchuan/zhh_bias/LR/{}_{}_{}_{}'.format(dir2,dir3, 'Bicubic', idx) + '.png', lr)
                cv2.imwrite('/hdd/tianchuan/zhh_bias/SR/{}_{}_{}_{}'.format(dir2,dir3, 'Bicubic', idx) + '.png', img1)
                cv2.imwrite('/hdd/tianchuan/zhh_bias/HR/{}_{}_{}_{}'.format(dir2,dir3, 'Bicubic', idx) + '.png', img2)


            HR = np.round(np.clip((img2) * 255, 0., 255.)).astype(np.uint8)
            SR = np.round(np.clip((img1) * 255, 0., 255.)).astype(np.uint8)
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