from utils import *
from imresize import imresize
from gkernel import generate_kernel

def _parse_function( example_proto):
    keys_to_features = {'label': tf.FixedLenFeature([], tf.string),
                        'image': tf.FixedLenFeature([], tf.string)}

    parsed_features = tf.parse_single_example(example_proto, keys_to_features)

    img = parsed_features['image']
    img = tf.divide(tf.cast(tf.decode_raw(img, tf.uint8), tf.float32), 255.)
    img = tf.reshape(img, [126 // 3, 126 // 3, 3])
    from PIL import Image

    img = Image.fromarray(img, 'RGB')  # 这里Image是之前提到的
    img.save('1.png')
    imageio.imsave('LR.png', img)

    label = parsed_features['label']
    label = tf.divide(tf.cast(tf.decode_raw(label, tf.uint8), tf.float32), 255.)
    label = tf.reshape(label, [126, 126, 3])

    return label, img

def load_tfrecord(dir):
    # print(dir)
    dataset = tf.data.TFRecordDataset(dir)
    dataset = dataset.map(_parse_function)

    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat()
    dataset = dataset.batch(20)
    iterator = dataset.make_one_shot_iterator()

    label_train, input_train = iterator.get_next()

    return label_train, input_train
if __name__ == '__main__':
    tfdir = '/hdd/tianchuan/Meteorological_data/DataSet_RGB/Wind/Wind.tfrecord'
    load_tfrecord(tfdir)