from utils import *
from imresize import imresize
from gkernel import generate_kernel

class dataGenerator(object):
    def __init__(self, output_shape, meta_batch_size, task_batch_size, tfrecord_path):
        self.buffer_size = 1000  # tf.data.TFRecordDataset buffer size

        self.TASK_BATCH_SIZE = task_batch_size
        self.HEIGHT, self.WIDTH, self.CHANNEL = output_shape

        self.META_BATCH_SIZE = meta_batch_size
        # self.tfrecord_path = tfrecord_path
        # self.label_train = self.load_tfrecord()

        tfdir = ['/hdd/tianchuan/Meteorological_data/Data_x4/era5_2000/d2m.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/era5_2000/sp.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/era5_2000/t2m.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/era5_2000/tp.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/era5_2000/u10.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/era5_2000/v10.tfrecord',

                 '/hdd/tianchuan/Meteorological_data/Data_x4/era5_2020/d2m.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/era5_2020/sp.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/era5_2020/t2m.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/era5_2020/tp.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/era5_2020/u10.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/era5_2020/v10.tfrecord',

                 '/hdd/tianchuan/Meteorological_data/Data_x4/GFS025/d2m.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/GFS025/sp.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/GFS025/t2m.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/GFS025/tp.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/GFS025/u10.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/GFS025/v10.tfrecord',

                 '/hdd/tianchuan/Meteorological_data/Data_x4/GFS050/d2m.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/GFS050/sp.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/GFS050/t2m.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/GFS050/tp.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/GFS050/u10.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/Data_x4/GFS050/v10.tfrecord',
                 ]
        self.label_era5_2000_d2m ,self.input_era5_2000_d2m= self.load_tfrecord(tfdir[0])
        self.label_era5_2000_sp ,self.input_era5_2000_sp= self.load_tfrecord(tfdir[1])
        self.label_era5_2000_t2m ,self.input_era5_2000_t2m= self.load_tfrecord(tfdir[2])
        self.label_era5_2000_tp ,self.input_era5_2000_tp= self.load_tfrecord(tfdir[3])
        self.label_era5_2000_u10 ,self.input_era5_2000_u10= self.load_tfrecord(tfdir[4])
        self.label_era5_2000_v10 ,self.input_era5_2000_v10= self.load_tfrecord(tfdir[5])

        self.label_era5_2020_d2m, self.input_era5_2020_d2m = self.load_tfrecord(tfdir[6])
        self.label_era5_2020_sp, self.input_era5_2020_sp = self.load_tfrecord(tfdir[7])
        self.label_era5_2020_t2m, self.input_era5_2020_t2m = self.load_tfrecord(tfdir[8])
        self.label_era5_2020_tp, self.input_era5_2020_tp = self.load_tfrecord(tfdir[9])
        self.label_era5_2020_u10, self.input_era5_2020_u10 = self.load_tfrecord(tfdir[10])
        self.label_era5_2020_v10, self.input_era5_2020_v10 = self.load_tfrecord(tfdir[11])

        self.label_GFS025_d2m, self.input_GFS025_d2m = self.load_tfrecord(tfdir[12])
        self.label_GFS025_sp, self.input_GFS025_sp = self.load_tfrecord(tfdir[13])
        self.label_GFS025_t2m, self.input_GFS025_t2m = self.load_tfrecord(tfdir[14])
        self.label_GFS025_tp, self.input_GFS025_tp = self.load_tfrecord(tfdir[15])
        self.label_GFS025_u10, self.input_GFS025_u10 = self.load_tfrecord(tfdir[16])
        self.label_GFS025_v10, self.input_GFS025_v10 = self.load_tfrecord(tfdir[17])

        self.label_GFS050_d2m, self.input_GFS050_d2m = self.load_tfrecord(tfdir[18])
        self.label_GFS050_sp, self.input_GFS050_sp = self.load_tfrecord(tfdir[19])
        self.label_GFS050_t2m, self.input_GFS050_t2m = self.load_tfrecord(tfdir[20])
        self.label_GFS050_tp, self.input_GFS050_tp = self.load_tfrecord(tfdir[21])
        self.label_GFS050_u10, self.input_GFS050_u10 = self.load_tfrecord(tfdir[22])
        self.label_GFS050_v10, self.input_GFS050_v10 = self.load_tfrecord(tfdir[23])

    def make_data_tensor(self, sess, scale_list, noise_std=0.0):
        label_era5_2000_d2m,input_era5_2000_d2m = sess.run([self.label_era5_2000_d2m ,self.input_era5_2000_d2m])
        label_era5_2000_sp, input_era5_2000_sp= sess.run([self.label_era5_2000_sp ,self.input_era5_2000_sp])
        label_era5_2000_t2m, input_era5_2000_t2m = sess.run([self.label_era5_2000_t2m, self.input_era5_2000_t2m])
        label_era5_2000_tp, input_era5_2000_tp = sess.run([self.label_era5_2000_tp, self.input_era5_2000_tp])
        label_era5_2000_u10, input_era5_2000_u10 = sess.run([self.label_era5_2000_u10, self.input_era5_2000_u10])
        label_era5_2000_v10, input_era5_2000_v10 = sess.run([self.label_era5_2000_v10, self.input_era5_2000_v10])

        label_era5_2020_d2m, input_era5_2020_d2m = sess.run([self.label_era5_2020_d2m, self.input_era5_2020_d2m])
        label_era5_2020_sp, input_era5_2020_sp = sess.run([self.label_era5_2020_sp, self.input_era5_2020_sp])
        label_era5_2020_t2m, input_era5_2020_t2m = sess.run([self.label_era5_2020_t2m, self.input_era5_2020_t2m])
        label_era5_2020_tp, input_era5_2020_tp = sess.run([self.label_era5_2020_tp, self.input_era5_2020_tp])
        label_era5_2020_u10, input_era5_2020_u10 = sess.run([self.label_era5_2020_u10, self.input_era5_2020_u10])
        label_era5_2020_v10, input_era5_2020_v10 = sess.run([self.label_era5_2020_v10, self.input_era5_2020_v10])

        label_GFS025_d2m, input_GFS025_d2m = sess.run([self.label_GFS025_d2m, self.input_GFS025_d2m])
        label_GFS025_sp, input_GFS025_sp = sess.run([self.label_GFS025_sp, self.input_GFS025_sp])
        label_GFS025_t2m, input_GFS025_t2m = sess.run([self.label_GFS025_t2m, self.input_GFS025_t2m])
        label_GFS025_tp, input_GFS025_tp = sess.run([self.label_GFS025_tp, self.input_GFS025_tp])
        label_GFS025_u10, input_GFS025_u10 = sess.run([self.label_GFS025_u10, self.input_GFS025_u10])
        label_GFS025_v10, input_GFS025_v10 = sess.run([self.label_GFS025_v10, self.input_GFS025_v10])

        label_GFS050_d2m, input_GFS050_d2m = sess.run([self.label_GFS050_d2m, self.input_GFS050_d2m])
        label_GFS050_sp, input_GFS050_sp = sess.run([self.label_GFS050_sp, self.input_GFS050_sp])
        label_GFS050_t2m, input_GFS050_t2m = sess.run([self.label_GFS050_t2m, self.input_GFS050_t2m])
        label_GFS050_tp, input_GFS050_tp = sess.run([self.label_GFS050_tp, self.input_GFS050_tp])
        label_GFS050_u10, input_GFS050_u10 = sess.run([self.label_GFS050_u10, self.input_GFS050_u10])
        label_GFS050_v10, input_GFS050_v10 = sess.run([self.label_GFS050_v10, self.input_GFS050_v10])


        input_meta = []
        label_meta = []

        for t in range(self.META_BATCH_SIZE):
            input_task = []
            label_task = []

            scale = np.random.choice(scale_list, 1)[0]
            # Kernel = generate_kernel(k1=scale * 2.5, ksize=15)
            if t == 0: label_train_ , input_train= label_era5_2000_d2m,input_era5_2000_d2m
            if t == 1: label_train_ , input_train= label_era5_2000_sp, input_era5_2000_sp
            if t == 2: label_train_ , input_train= label_era5_2000_t2m, input_era5_2000_t2m
            if t == 3: label_train_ , input_train= label_era5_2000_tp, input_era5_2000_tp
            if t == 4: label_train_ , input_train= label_era5_2000_u10, input_era5_2000_u10
            if t == 5: label_train_ , input_train= label_era5_2000_v10, input_era5_2000_v10

            if t == 6: label_train_, input_train = label_era5_2020_d2m, input_era5_2020_d2m
            if t == 7: label_train_, input_train = label_era5_2020_sp, input_era5_2020_sp
            if t == 8: label_train_, input_train = label_era5_2020_t2m, input_era5_2020_t2m
            if t == 9: label_train_, input_train = label_era5_2020_tp, input_era5_2020_tp
            if t == 10: label_train_, input_train = label_era5_2020_u10, input_era5_2020_u10
            if t == 11: label_train_, input_train = label_era5_2020_v10, input_era5_2020_v10

            if t == 12: label_train_, input_train = label_GFS025_d2m, input_GFS025_d2m
            if t == 13: label_train_, input_train = label_GFS025_sp, input_GFS025_sp
            if t == 14: label_train_, input_train = label_GFS025_t2m, input_GFS025_t2m
            if t == 15: label_train_, input_train = label_GFS025_tp, input_GFS025_tp
            if t == 16: label_train_, input_train = label_GFS025_u10, input_GFS025_u10
            if t == 17: label_train_, input_train = label_GFS025_v10, input_GFS025_v10

            if t == 18: label_train_, input_train = label_GFS050_d2m, input_GFS050_d2m
            if t == 19: label_train_, input_train = label_GFS050_sp, input_GFS050_sp
            if t == 20: label_train_, input_train = label_GFS050_t2m, input_GFS050_t2m
            if t == 21: label_train_, input_train = label_GFS050_tp, input_GFS050_tp
            if t == 22: label_train_, input_train = label_GFS050_u10, input_GFS050_u10
            if t == 23: label_train_, input_train = label_GFS050_v10, input_GFS050_v10

            for idx in range(self.TASK_BATCH_SIZE * 2):
                img_HR = label_train_[t * self.TASK_BATCH_SIZE * 2 + idx]
                img_ILR = input_train[t * self.TASK_BATCH_SIZE * 2 + idx]

                img_ILR = imresize(img_ILR, scale=scale, output_shape=img_HR.shape, kernel='cubic')

                input_task.append(img_ILR)
                label_task.append(img_HR)

            input_meta.append(np.asarray(input_task))
            label_meta.append(np.asarray(label_task))

        input_meta = np.asarray(input_meta)
        label_meta = np.asarray(label_meta)

        inputa = input_meta[:, :self.TASK_BATCH_SIZE, :, :]
        labela = label_meta[:, :self.TASK_BATCH_SIZE, :, :]
        inputb = input_meta[:, self.TASK_BATCH_SIZE:, :, :]
        labelb = label_meta[:, self.TASK_BATCH_SIZE:, :, :]

        return inputa, labela, inputb, labelb

    '''Load TFRECORD'''

    def _parse_function(self, example_proto):
        keys_to_features = {'label': tf.FixedLenFeature([], tf.string),
                            'image': tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)

        img = parsed_features['image']
        img = tf.divide(tf.cast(tf.decode_raw(img, tf.uint8), tf.float32), 255.)
        img = tf.reshape(img, [self.HEIGHT // 4, self.WIDTH // 4, self.CHANNEL])

        label = parsed_features['label']
        label = tf.divide(tf.cast(tf.decode_raw(label, tf.uint8), tf.float32), 255.)
        label = tf.reshape(label, [self.HEIGHT, self.WIDTH, self.CHANNEL])

        return label, img

    def load_tfrecord(self, dir):
        # print(dir)
        dataset = tf.data.TFRecordDataset(dir)
        dataset = dataset.map(self._parse_function)

        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.TASK_BATCH_SIZE * self.META_BATCH_SIZE * 2)
        iterator = dataset.make_one_shot_iterator()

        label_train, input_train = iterator.get_next()

        return label_train, input_train