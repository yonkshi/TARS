PREPROCESSED_DATA = 'asset/data/'
BATCH_SIZE = 64
FEATURE_DIM = 20 # Features in MFCC
MAX_FRAMES = 1000 # TODO I just made this number up

ALPHABETS =  [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '<EMP>']
ALPHABETS_DICT = {}
for i, ch in enumerate(ALPHABETS):
    ALPHABETS_DICT[ch] = i

ALPHA_SIZE = len(ALPHABETS)

CHAR_SIZE = 700
