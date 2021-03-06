import tensorflow as tf
import numpy as np
import csv
import string
import wavenet.conf as conf



# default data path
_data_path = 'asset/data/'

#
# vocabulary table
#

# index to byte mapping
index2byte = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '<EMP>']

# byte to index mapping
byte2index = {}
for i, ch in enumerate(index2byte):
    byte2index[ch] = i

# vocabulary size
voca_size = len(index2byte)

# convert sentence to index list
def str2index(str_):

    # clean white space
    str_ = ' '.join(str_.split())
    # remove punctuation and make lower case
    translator = str.maketrans('', '', string.punctuation)
    str_ = str_.translate(translator).lower()

    res = []
    for ch in str_:
        try:
            res.append(byte2index[ch])
        except KeyError:
            # drop OOV
            pass
    return res

# convert index list to string
def index2str(index_list):
    # transform label index to character
    str_ = ''
    for ch in index_list:
        if ch > 0:
            str_ += index2byte[ch]
        elif ch == 0:  # <EOS>
            break
    return str_

# print list of index list
def print_index(indices):
    for index_list in indices:
        print(index2str(index_list))

# real-time wave to mfcc conversion function

def _load_mfcc(label, mfcc_file:bytes):

    # decode string to integer
    label_new = np.fromstring(label, np.int)

    mfcc_file_str = mfcc_file.decode()

    # load mfcc
    mfcc = np.load(mfcc_file_str, allow_pickle=False)

    seq_len = np.array(mfcc.shape).astype('int32')[1] # int32
    # speed perturbation augmenting
    mfcc = _augment_speech(mfcc).T

    mfcc = mfcc.astype('float32')
    label_new = label_new.astype('int32')
    return label_new, mfcc, [seq_len] # stupid batch


def _augment_speech(mfcc):

    # random frequency shift ( == speed perturbation effect on MFCC )
    r = np.random.randint(-2, 2)

    # shifting mfcc
    mfcc = np.roll(mfcc, r, axis=0)

    # zero padding
    if r > 0:
        mfcc[:r, :] = 0
    elif r < 0:
        mfcc[r:, :] = 0

    return mfcc


# Speech Corpus
class SpeechCorpus(object):
    def __init__(self, batch_size=16, set_name='train'):
        # load meta file
        label, mfcc_file = [], []
        with open(_data_path + 'preprocess/meta/%s.csv' % set_name) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                # mfcc file
                mfcc_file.append(_data_path + 'preprocess/mfcc/' + row[0] + '.npy')
                # label info ( convert to string object for variable-length support )
                label.append(np.asarray(row[1:], dtype=np.int).tostring())

        # to constant tensor
        label_t = tf.convert_to_tensor(label)
        mfcc_file_t = tf.convert_to_tensor(mfcc_file)

        # New pipeline
        datasource = tf.data.Dataset.from_tensor_slices((label_t, mfcc_file_t))
        dataset = datasource.shuffle(buffer_size=1024)
        dataset = dataset.map(lambda x, y: tf.py_func(func=_load_mfcc, inp=[x, y], Tout=[tf.int32, tf.float32, tf.int32]),
                              num_parallel_calls=64)
        dataset = dataset.prefetch(256)
        dataset = dataset.padded_batch(batch_size, padded_shapes=([None],[None, conf.FEATURE_DIM],1))
        dataset = dataset.map(self.to_sparse_representation)

        self.dataset = dataset
        self.iterator = dataset.make_initializable_iterator()
        self.next_batch = self.iterator.get_next()

    def to_sparse_representation(self, labels, x, seq_len):

        #labels = tf.transpose(labels, [2,0,1]) # Alpha size x batch_size x num chars, for ctc_loss
        #x = tf.transpose(x, [1, 0, 2]) # for ctc_loss
        indices = tf.where(tf.not_equal(labels, 0))
        sparse_label = tf.SparseTensor(indices=indices,
                               values=tf.gather_nd(tf.cast(labels,tf.int32), indices) - 1,  # for zero-based index
                               dense_shape=tf.cast(tf.shape(labels), tf.int64))
        return sparse_label, x, seq_len
