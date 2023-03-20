##深度学习过程中，需要制作训练集和验证集、测试集。

import os, random, shutil


def moveFile(fileHRDir,fileLRDir):
    pathHRDir = os.listdir(fileHRDir)  # 取图片的原始路径
    pathLRDir = os.listdir(fileLRDir)  # 取图片的原始路径

    pathHRDir.sort()
    pathLRDir.sort()
    # filenumber = len(pathDir)
    # rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    # picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    # sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    # print(sample)
    # for name in sample:
    for name in (pathHRDir[:20]):
        # shutil.move(fileDir + name, tarDir + name)
        shutil.copy(fileHRDir + name, fineHRDir + name)
    for name in (pathLRDir[:20]):
        # shutil.move(fileDir + name, tarDir + name)
        shutil.copy(fileLRDir + name, fineLRDir + name)
    for name in (pathHRDir[20:]):
        # shutil.move(fileDir + name, tarDir + name)
        shutil.copy(fileHRDir + name, testHRDir + name)
    for name in (pathLRDir[20:]):
        # shutil.move(fileDir + name, tarDir + name)
        shutil.copy(fileLRDir + name, testLRDir + name)

    return


if __name__ == '__main__':
    # data = ['mwp','sp','sst','swh'] #
    data = ['d2m', 'sp', 'tp', 't2m', 'u10','v10']
    # data = ['msl', 'ssrc', 'sst']


    for i in data:
        fileLRDir = '/hdd/zhanghonghu/fine_test/LR/GFS050/{}/'.format(i)  # 源图片文件夹路径
        fileHRDir = '/hdd/zhanghonghu/fine_test/HR/GFS050/{}/'.format(i)  # 源图片文件夹路径

        fineHRDir = '/hdd/tianchuan/Meteorological_data/Data_x4/fine-test/GFS050/Finetune/{}/HR/'.format(i)  # 移动到新的文件夹路径
        fineLRDir = '/hdd/tianchuan/Meteorological_data/Data_x4/fine-test/GFS050/Finetune/{}/LR/'.format(i)  # 移动到新的文件夹路径

        testHRDir = '/hdd/tianchuan/Meteorological_data/Data_x4/fine-test/GFS050/Test/{}/HR/'.format(i)  # 移动到新的文件夹路径
        testLRDir = '/hdd/tianchuan/Meteorological_data/Data_x4/fine-test/GFS050/Test/{}/LR/'.format(i)  # 移动到新的文件夹路径

        isExists_HR_fine = os.path.exists(fineHRDir)
        if not (isExists_HR_fine):
            os.makedirs(fineHRDir)
        isExists_LR_fine = os.path.exists(fineLRDir)
        if not (isExists_LR_fine):
            os.makedirs(fineLRDir)

        isExists_HR_test = os.path.exists(testHRDir)
        if not (isExists_HR_test):
            os.makedirs(testHRDir)
        isExists_LR_test = os.path.exists(testLRDir)
        if not (isExists_LR_test):
            os.makedirs(testLRDir)

        moveFile(fileHRDir,fileLRDir)