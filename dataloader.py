import csv
import string

import numpy as np
import tensorflow as tf
import conf
from sklearn.preprocessing import StandardScaler

# convert index list to string
def index2str(index_list):
    # transform label index to character
    str_ = ''
    for ch in index_list:
        if ch > 0:
            str_ += conf.ALPHABETS[ch]
        elif ch == 0:  # <EOS>
            break
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
            res.append(conf.ALPHABETS_DICT[ch])
        except KeyError:
            # drop OOV
            pass
    return res

class DataLoader(object):

    def __init__(self, batch_size=16, set_name='train'):
        # load meta file
        label, mfcc_files, mfcc_seq_len = [], [], []
        with open(conf.PREPROCESSED_DATA + 'preprocess/meta/%s.csv' % set_name) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                # mfcc file
                mfcc_filename = conf.PREPROCESSED_DATA + 'preprocess/mfcc/' + row[0] + '.npy'
                mfcc = np.load(mfcc_filename, allow_pickle=False).T
                mfcc_seq_len.append(mfcc.shape[0]) # sequence length
                mfcc_files.append(mfcc)
                # label info ( convert to string object for variable-length support )
                label.append(np.asarray(row[1:], dtype=np.int).tostring())





        # Zero mean and unit variance
        mfcc_massive = np.vstack(mfcc_files)
        sc = StandardScaler()
        mfcc_massive_normalized = sc.fit_transform(mfcc_massive)
        mfcc_files = np.split(mfcc_massive_normalized, mfcc_seq_len)
        # to constant tensor
        label_t = tf.convert_to_tensor(label)
        #mfcc_file_t = tf.convert_to_tensor(mfcc_files)
        def gen():
            for file in mfcc_files:
                yield file
        # New pipeline
        datasource_mfcc = tf.data.Dataset.from_generator(gen, output_types=tf.float32, output_shapes=[None, 20])
        datasource_labels = tf.data.Dataset.from_tensor_slices(label_t)
        datasource = tf.data.Dataset.zip((datasource_labels, datasource_mfcc))
        dataset = datasource.shuffle(buffer_size=1024)
        dataset = dataset.map(lambda x, y: tf.py_func(func=self._load_mfcc, inp=[x, y], Tout=[tf.int32, tf.string, tf.float32, tf.int32]),
                              num_parallel_calls=64)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(1000)
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=([None],[], [None, conf.FEATURE_DIM],1),
                                       padding_values=(27, '', 0.0, 0)) # 27 is <EMP> Hard coded
        dataset = dataset.map(self.to_sparse_representation)

        self.dataset = dataset
        self.iterator = dataset.make_initializable_iterator()
        self.next_batch = self.iterator.get_next()

    def training_set(self):
        return self.next_batch

    def _load_mfcc(self,label, mfcc_file: bytes):
        # decode string to integer
        label_new = np.fromstring(label, np.int)
        #mfcc_file_str = mfcc_file.decode()
        mfcc = mfcc_file #np.load(mfcc_file_str, allow_pickle=False)
        seq_len = np.array(mfcc.shape).astype('int32')[0]  # int32
        #mfcc = self._augment_speech(mfcc).T
        mfcc = mfcc.astype('float32')
        label_encoded = label_new.astype('int32')

        return label_encoded, label, mfcc, [seq_len]  # stupid batch

    def _augment_speech(self, mfcc):

        r = np.random.randint(-2, 2)
        # shifting mfcc
        mfcc = np.roll(mfcc, r, axis=1)
        # zero padding
        if r > 0:
            mfcc[:, :r] = 0
        elif r < 0:
            mfcc[:, r:] = 0
        return mfcc

    def to_sparse_representation(self, labels, label_text, x, seq_len):
        indices = tf.where(tf.not_equal(labels, 0))
        sparse_label = tf.SparseTensor(indices=indices,
                               values=tf.gather_nd(tf.cast(labels,tf.int32), indices),  # for zero-based index
                               dense_shape=tf.cast(tf.shape(labels), tf.int64))
        return sparse_label,label_text, x, seq_len
