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

        tfdir = ['/hdd/tianchuan/Meteorological_data/era5hourUSA/d2m_train.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/era5hourUSA/t2m_train.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/era5hourUSA/msl_train.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/era5hourUSA/u10_train.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/era5hourUSA/v10_train.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/era5hourUSA/mwd_train.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/era5hourUSA/ssrc_train.tfrecord',
                 '/hdd/tianchuan/Meteorological_data/era5hourUSA/tp_train.tfrecord',
                 ]
        self.label_train_d2m ,self.input_train_d2m= self.load_tfrecord(tfdir[0])
        self.label_train_t2m ,self.input_train_t2m= self.load_tfrecord(tfdir[1])
        self.label_train_msl ,self.input_train_msl= self.load_tfrecord(tfdir[2])
        self.label_train_u10 ,self.input_train_u10= self.load_tfrecord(tfdir[3])
        self.label_train_v10 ,self.input_train_v10= self.load_tfrecord(tfdir[4])
        self.label_train_mwd ,self.input_train_mwd= self.load_tfrecord(tfdir[5])
        self.label_train_ssrc ,self.input_train_ssrc= self.load_tfrecord(tfdir[6])
        self.label_train_tp ,self.input_train_tp= self.load_tfrecord(tfdir[7])

    def make_data_tensor(self, sess, scale_list, noise_std=0.0):
        label_train_d2m_,input_train_d2m = sess.run([self.label_train_d2m ,self.input_train_d2m])
        label_train_t2m_, input_train_t2m= sess.run([self.label_train_t2m ,self.input_train_t2m])
        label_train_msl_,input_train_msl = sess.run([self.label_train_msl ,self.input_train_msl])
        label_train_u10_,input_train_u10 = sess.run([self.label_train_u10 ,self.input_train_u10])
        label_train_v10_ ,input_train_v10= sess.run([self.label_train_v10 ,self.input_train_v10])
        label_train_mwd_ ,input_train_mwd= sess.run([self.label_train_mwd ,self.input_train_mwd])
        label_train_ssr_ ,input_train_ssrc= sess.run([self.label_train_ssrc ,self.input_train_ssrc])
        label_train_tp_ ,input_train_tp = sess.run([self.label_train_tp ,self.input_train_tp])

        input_meta = []
        label_meta = []

        for t in range(self.META_BATCH_SIZE):
            input_task = []
            label_task = []

            scale = np.random.choice(scale_list, 1)[0]
            # Kernel = generate_kernel(k1=scale * 2.5, ksize=15)

            if t == 0: label_train_ , input_train= label_train_d2m_,input_train_d2m
            if t == 1: label_train_ , input_train= label_train_t2m_,input_train_t2m
            if t == 2: label_train_ , input_train= label_train_msl_,input_train_msl
            if t == 3: label_train_ , input_train= label_train_u10_,input_train_u10
            if t == 4: label_train_ , input_train= label_train_v10_,input_train_v10
            if t == 5: label_train_ , input_train= label_train_mwd_,input_train_mwd
            if t == 6: label_train_ , input_train= label_train_ssr_,input_train_ssrc
            if t == 7: label_train_ , input_train= label_train_tp_,input_train_tp

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
        img = tf.reshape(img, [self.HEIGHT // 8, self.WIDTH // 8, self.CHANNEL])

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