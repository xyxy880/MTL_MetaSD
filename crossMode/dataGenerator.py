from utils import *
from imresize import imresize
from gkernel import generate_kernel


class dataGenerator(object):
    def __init__(self, output_shape, meta_batch_size, task_batch_size, tfrecord_path):
        self.buffer_size = 1000  # tf.data.TFRecordDataset buffer size

        self.TASK_BATCH_SIZE = task_batch_size
        # self.HEIGHT, self.WIDTH, self.CHANNEL = output_shape

        self.META_BATCH_SIZE = meta_batch_size

        tfdir = ['/hdd/tianchuan/Meteorological_data/wrf_era5/wrf/T2/T2_train.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/wrf_era5/era5/t2m/t2m_train.tfrecord',

                 '/hdd/tianchuan/Meteorological_data/wrf_era5/wrf/Wind/Wind_train.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/wrf_era5/era5/wind/wind_train.tfrecord',

                 '/hdd/tianchuan/Meteorological_data/wrf_era5/wrf/SST/SST_train.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/wrf_era5/era5/sst/sst_train.tfrecord',

                 '/hdd/tianchuan/Meteorological_data/wrf_era5/wrf/TSK/TSK_train.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/wrf_era5/era5/skt/skt_train.tfrecord',

                 '/hdd/tianchuan/Meteorological_data/wrf_era5/wrf/LH/LH_train.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/wrf_era5/era5/slhf/slhf_train.tfrecord',
                 ]
        self.label_train_T2 ,self.input_train_T2= self.load_tfrecord1(tfdir[0])
        self.label_train_t2m ,self.input_train_t2m= self.load_tfrecord2(tfdir[1])

        self.label_train_wind, self.input_train_wind = self.load_tfrecord1(tfdir[2])
        self.label_train_uv10, self.input_train_uv10 = self.load_tfrecord2(tfdir[3])

        self.label_train_sst, self.input_train_sst = self.load_tfrecord1(tfdir[4])
        self.label_train_SST, self.input_train_SST = self.load_tfrecord2(tfdir[5])

        self.label_train_TSK, self.input_train_TSK = self.load_tfrecord1(tfdir[6])
        self.label_train_skt, self.input_train_skt = self.load_tfrecord2(tfdir[7])

        self.label_train_LH, self.input_train_LH = self.load_tfrecord1(tfdir[8])
        self.label_train_slhf, self.input_train_slhf = self.load_tfrecord2(tfdir[9])


    def make_data_tensor(self, sess, scale_list, noise_std=0.0):
        label_train_t2m_, input_train_t2m= sess.run([self.label_train_t2m ,self.input_train_t2m])
        label_train_T2_, input_train_T2= sess.run([self.label_train_T2 ,self.input_train_T2])

        label_train_wind_, input_train_wind = sess.run([self.label_train_wind, self.input_train_wind])
        label_train_uv10_, input_train_uv10 = sess.run([self.label_train_uv10, self.input_train_uv10])

        label_train_sst_, input_train_sst = sess.run([self.label_train_sst, self.input_train_sst])
        label_train_SST_, input_train_SST = sess.run([self.label_train_SST, self.input_train_SST])

        label_train_TSK_, input_train_TSK = sess.run([self.label_train_TSK, self.input_train_TSK])
        label_train_skt_, input_train_skt = sess.run([self.label_train_skt, self.input_train_skt])

        label_train_LH_, input_train_LH = sess.run([self.label_train_LH, self.input_train_LH])
        label_train_slhf_, input_train_slhf = sess.run([self.label_train_slhf, self.input_train_slhf])

        input_meta_wrf = []
        label_meta_wrf = []
        input_meta_era5 = []
        label_meta_era5 = []

        for t in range(self.META_BATCH_SIZE):
            input_task_wrf = []
            label_task_wrf = []
            input_task_era5 = []
            label_task_era5 = []
            scale = np.random.choice(scale_list, 1)[0]
            if t == 0:
                label_wrf_ , input_wrf= label_train_T2_, input_train_T2
                label_era5_, input_era5 = label_train_t2m_, input_train_t2m
            if t == 1:
                label_wrf_, input_wrf = label_train_wind_, input_train_wind
                label_era5_, input_era5 = label_train_uv10_, input_train_uv10
            if t == 2:
                label_wrf_, input_wrf = label_train_sst_, input_train_sst
                label_era5_, input_era5 = label_train_SST_, input_train_SST
            if t == 3:
                label_wrf_, input_wrf = label_train_TSK_, input_train_TSK
                label_era5_, input_era5 = label_train_skt_, input_train_skt
            if t == 4:
                label_wrf_, input_wrf = label_train_LH_, input_train_LH
                label_era5_, input_era5 = label_train_slhf_, input_train_slhf

            for idx in range(self.TASK_BATCH_SIZE):
                wrf_HR = label_wrf_[t * self.TASK_BATCH_SIZE+ idx]
                wrf_ILR = input_wrf[t * self.TASK_BATCH_SIZE+ idx]
                wrf_ILR = imresize(wrf_ILR, scale=scale, output_shape=wrf_HR.shape, kernel='cubic')

                era5_HR = label_era5_[t * self.TASK_BATCH_SIZE + idx]
                era5_ILR = input_era5[t * self.TASK_BATCH_SIZE + idx]
                era5_ILR = imresize(era5_ILR, scale=scale, output_shape=era5_HR.shape, kernel='cubic')

                input_task_wrf.append(wrf_ILR)
                label_task_wrf.append(wrf_HR)
                input_task_era5.append(era5_ILR)
                label_task_era5.append(era5_HR)

            input_meta_wrf.append(np.asarray(input_task_wrf))
            label_meta_wrf.append(np.asarray(label_task_wrf))
            input_meta_era5.append(np.asarray(input_task_era5))
            label_meta_era5.append(np.asarray(label_task_era5))

        input_meta_wrf = np.asarray(input_meta_wrf)
        label_meta_wrf = np.asarray(label_meta_wrf)
        input_meta_era5 = np.asarray(input_meta_era5)
        label_meta_era5 = np.asarray(label_meta_era5)

        inputa = input_meta_wrf[:, :, :, :]
        labela = label_meta_wrf[:, :, :, :]

        inputb = input_meta_era5[:, :, :, :]
        labelb = label_meta_era5[:, :, :, :]

        # return inputa, labela, inputb, labelb

        return inputb, labelb,inputa, labela

    '''Load TFRECORD'''

    def _parse_function1(self, example_proto):
        keys_to_features = {'label': tf.FixedLenFeature([], tf.string),
                            'image': tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)

        img = parsed_features['image']
        img = tf.divide(tf.cast(tf.decode_raw(img, tf.uint8), tf.float32), 255.)
        img = tf.reshape(img, [201 // 3, 201 // 3, 3])

        label = parsed_features['label']
        label = tf.divide(tf.cast(tf.decode_raw(label, tf.uint8), tf.float32), 255.)
        label = tf.reshape(label, [201, 201, 3])

        return label, img

    def load_tfrecord1(self, dir):
        # print(dir)
        dataset = tf.data.TFRecordDataset(dir)
        dataset = dataset.map(self._parse_function1)

        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.TASK_BATCH_SIZE * self.META_BATCH_SIZE)
        iterator = dataset.make_one_shot_iterator()

        label_train, input_train = iterator.get_next()

        return label_train, input_train

    '''Load TFRECORD'''

    def _parse_function2(self, example_proto):
        keys_to_features = {'label': tf.FixedLenFeature([], tf.string),
                            'image': tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)

        img = parsed_features['image']
        img = tf.divide(tf.cast(tf.decode_raw(img, tf.uint8), tf.float32), 255.)
        img = tf.reshape(img, [21 // 3, 21 // 3, 3])

        label = parsed_features['label']
        label = tf.divide(tf.cast(tf.decode_raw(label, tf.uint8), tf.float32), 255.)
        label = tf.reshape(label, [21, 21, 3])

        return label, img

    def load_tfrecord2(self, dir):
        # print(dir)
        dataset = tf.data.TFRecordDataset(dir)
        dataset = dataset.map(self._parse_function2)

        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.TASK_BATCH_SIZE * self.META_BATCH_SIZE)
        iterator = dataset.make_one_shot_iterator()

        label_train, input_train = iterator.get_next()

        return label_train, input_train