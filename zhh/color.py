import matplotlib.pyplot as plt
import cv2

# 读取一张图片
image = cv2.imread('/hdd/tianchuan/zhh_bias/biasGFS050.png') # 你需要指定图片的路径
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转换颜色模式为RGB

# 给图片添加颜色条
plt.imshow(image, cmap='seismic', vmin=-80, vmax=80) # 设置颜色条的色调和范围
cbar = plt.colorbar(fraction=0.1) # fraction参数控制颜色条的宽度
cbar.set_label('Colorbar') # 设置颜色条的标签

# 设置颜色条的字体为Times New Roman 尺寸为16
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_family('Times New Roman')
    l.set_size(12)

# 显示图片和颜色条
plt.show()