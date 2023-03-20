import tensorflow as tf
from PIL import Image
CUDA_VISIBLE_DEVICES=2

#写入将要保存图片路径，需要自己手动新建文件夹
swd = '/hdd/tianchuan/Climate_work/MZSR-v3/Retrain/'
#TFRecord文件路径，只能打开某一个具体的tfrecord,有多个那就改一下咯。
data_path = '/hdd/tianchuan/solar_LR-MR.tfrecord'
# 获取文件名列表
data_files = tf.gfile.Glob(data_path)
# 文件名列表生成器
filename_queue = tf.train.string_input_producer(data_files,shuffle=True)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #返回文件名和文件

features = tf.parse_single_example(serialized_example,
                                   features={'index': tf.FixedLenFeature([], tf.int64),
                                           'data_LR': tf.FixedLenFeature([], tf.string),
                                              'h_LR': tf.FixedLenFeature([], tf.int64),
                                              'w_LR': tf.FixedLenFeature([], tf.int64),
                                           'data_HR': tf.FixedLenFeature([], tf.string),
                                              'h_HR': tf.FixedLenFeature([], tf.int64),
                                              'w_HR': tf.FixedLenFeature([], tf.int64),
                                           'c': tf.FixedLenFeature([], tf.int64)})  #取出包含image和label的feature对象

#tf.decode_raw可以将字符串解析成图像对应的像素数组
image = tf.cast(tf.decode_raw(features['data_LR'], tf.uint8), tf.float32)
label = tf.cast(tf.decode_raw(features['data_HR'], tf.uint8), tf.float32)
height = features['h_LR']
width = features['w_LR']
c = features['c']

image = tf.reshape(image, [height,width,c])

with tf.Session() as sess: #开始一个会话
    print('111111')
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # #启动多线程
    # coord=tf.train.Coordinator()
    #
    # threads= tf.train.start_queue_runners(coord=coord)

    for i in range(2):
        single,l = sess.run([image,label])#在会话中取出image和label

        img=Image.fromarray(single, 'RGB')
        img.save('2.jpg')

    # coord.request_stop()
    #
    # coord.join(threads)


