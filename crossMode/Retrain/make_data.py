##深度学习过程中，需要制作训练集和验证集、测试集。

import os, random, shutil


def moveFile(fileDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径

    pathDir.sort()
    # filenumber = len(pathDir)
    # rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    # picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    # sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    # print(sample)
    # for name in sample:
    for name in (pathDir[:]):
        # shutil.move(fileDir + name, tarDir + name)
        shutil.copy(fileDir + name, fineDir + name)
    # for name in (pathDir[720:]):
        # shutil.move(fileDir + name, tarDir + name)
        # shutil.copy(fileDir + name, testDir + name)
    return


if __name__ == '__main__':
    # data = ['mwp','sp','sst','swh'] #
    # data = ['d2m','msl','mwd','ssrc','tp','t2m','u10','v10','mwp','sp','sst','swh'] #
    data = ['wind','t2m','sst','slhf','skt']
    for i in data:
        # fileDir = '/hdd/tianchuan/Meteorological_data/era5hourChinaAll/{}/1980/'.format(i)  # 源图片文件夹路径
        # fineDir = '/hdd/tianchuan/Meteorological_data/era5hourChinaAll/{}/Finetune/'.format(i)  # 移动到新的文件夹路径
        # testDir = '/hdd/tianchuan/Meteorological_data/era5hourChinaAll/{}/Test/'.format(i)  # 移动到新的文件夹路径
        fileDir = '/hdd/tianchuan/Meteorological_data/wrf_era5/era5/{}/HR_train/'.format(i)  # 源图片文件夹路径
        fineDir = '/hdd/tianchuan/Meteorological_data/wrf_era5/era5/merge/HR/'  # 移动到新的文件夹路径
        # testDir = '/hdd/tianchuan/Meteorological_data/era5hourChinaAll/{}/Test/'.format(i)  # 移动到新的文件夹路径

        isExists_HR_TRAIN = os.path.exists(fineDir)
        if not (isExists_HR_TRAIN):
            os.makedirs(fineDir)
        # isExists_HR_TRAIN = os.path.exists(testDir)
        # if not (isExists_HR_TRAIN):
        #     os.makedirs(testDir)
        moveFile(fileDir)