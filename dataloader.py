import csv
import string

import numpy as np
import tensorflow as tf
import conf
from sklearn.preprocessing import StandardScaler

# convert index list to string
def index2str(index_list):
    # transform label index to character
    str_ = []
    for ch in index_list:
        if ch <= conf.PHONEME_SIZE:
            str_.append(conf.PHONEMES[ch])
        # else:  # <EOS>
        #     break
    return str_

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
            res.append(conf.PHONEMES_DICT[ch])
        except KeyError:
            # drop OOV
            pass
    return res

class DataLoader(object):

    def __init__(self, batch_size=16, set_name='train'):
        # load meta file
        label, mfcc_files, mfcc_seq_len, mfccs, = [], [], [], []
        with open(conf.PREPROCESSED_DATA + 'preprocess/meta/%s.csv' % set_name) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(reader):
                # if i >= conf.BATCH_SIZE:
                #     break
                # mfcc file
                mfcc_filename = conf.PREPROCESSED_DATA + 'preprocess/mfcc/' + row[0] + '.npy'
                mfcc = np.load(mfcc_filename, allow_pickle=False).T
                mfcc_seq_len.append(mfcc.shape[0]) # sequence length
                mfccs.append(mfcc)
                mfcc_files.append(mfcc_filename)
                # label info ( convert to string object for variable-length support )
                label.append(np.asarray(row[1:], dtype=np.int).tostring())

        # Zero mean and unit variance
        # This step somehow causes 0 loss, so disabling until a better fix can be found
        # mfcc_massive = np.vstack(mfccs)
        # sc = StandardScaler()
        # mfcc_massive_normalized = sc.fit_transform(mfcc_massive)
        # mfccs = np.split(mfcc_massive_normalized, mfcc_seq_len)

        # to constant tensor
        label_t = tf.convert_to_tensor(label)
        # Generator used to feed data into Dataset. Using generator because of uneven shapes (from_tensor_slices could
        # not handle
        def gen():
            for i, mfcc in enumerate(mfccs):
                yield mfcc
        # New pipeline, loads mfcc content, labels, and mfcc file name
        datasource_mfcc = tf.data.Dataset.from_generator(gen, output_types=tf.float32, output_shapes=[None, 20])
        datasource_labels = tf.data.Dataset.from_tensor_slices(label_t)
        datasource_sound_file = tf.data.Dataset.from_tensor_slices(mfcc_files)
        datasource = tf.data.Dataset.zip((datasource_labels, datasource_mfcc, datasource_sound_file))
        #dataset = datasource.shuffle(buffer_size=1024)
        # Python processing function
        dataset = datasource.map(lambda x, y, z: tf.py_func(func=self._load_mfcc, inp=[x, y, z], Tout=[tf.int32, tf.string, tf.float32, tf.int32, tf.string]),
                              num_parallel_calls=64)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(1000)
        # Padding batch so the output has the same shape. E.g. MFCC have different sequence lengths,
        # This padding will dynamically pad to the longest length
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=([None],[], [None, conf.FEATURE_DIM],1, []),
                                       padding_values=(27, '', 0.0, 0, '')) # 27 is <EMP> Hard coded
        # Sparse matrix conversion
        dataset = dataset.map(self.to_sparse_representation, num_parallel_calls=24)
        dataset = dataset.prefetch(100)

        # Iterator used for initliazation
        self.dataset = dataset
        self.iterator = dataset.make_initializable_iterator()
        self.next_batch = self.iterator.get_next()

    def training_set(self):
        return self.next_batch

    def _load_mfcc(self,label, mfcc, mfcc_file):
        """
        Main processing function for converting mfcc to int, convert labels to int
        :param label:
        :param mfcc:
        :param mfcc_file:
        :return:
        """
        # decode string to integer
        seq_len = np.array(mfcc.shape).astype('int32')[0]  # int32
        mfcc = mfcc.astype('float32')
        label_new = np.fromstring(label, np.int)
        label_encoded = label_new.astype('int32')

        return label_encoded, label, mfcc, [seq_len], mfcc_file  # stupid batch

    def _augment_speech(self, mfcc):
        '''
        Cited in another paper that augmented mfcc are more resilient to noise
        :param mfcc:
        :return:
        '''
        r = np.random.randint(-2, 2)
        # shifting mfcc
        mfcc = np.roll(mfcc, r, axis=1)
        # zero padding
        if r > 0:
            mfcc[:, :r] = 0
        elif r < 0:
            mfcc[:, r:] = 0
        return mfcc

    def to_sparse_representation(self, labels, label_text, x, seq_len, mfcc_file_list):
        '''
        Convert a dense label matrix into a sparse matrix
        :param labels:
        :param label_text:
        :param x:
        :param seq_len:
        :param mfcc_file_list:
        :return:
        '''
        indices = tf.where(tf.not_equal(labels, 27)) # 27 being the <Epsilon> char used for CTC
        sparse_label = tf.SparseTensor(indices=indices,
                               values=tf.gather_nd(tf.cast(labels,tf.int32), indices),  # for zero-based index
                               dense_shape=tf.cast(tf.shape(labels), tf.int64))
        return sparse_label,label_text, x, seq_len, mfcc_file_list
