PREPROCESSED_DATA = 'asset/data/'
BATCH_SIZE = 16
FEATURE_DIM = 20 # Features in MFCC

ALPHABETS =  [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '<EMP>']
ALPHABETS_DICT = {}
for i, ch in enumerate(ALPHABETS):
    ALPHABETS_DICT[ch] = i

ALPHA_SIZE = len(ALPHABETS)

CHAR_SIZE = 700

PHONEMES = ['sp',  'AE2',  'AA2',  'OW0',  'UW0',  'AY2',  'NG',  'EH0',  'OW2',  'N',  'P',  'AE1',  'OW1',  'AO1',  'ER0',  'OY1',  'UH2',  'IH0',  'UH0',  'K',  'AH1',  'EH1',  'EY2',  'OY2',  'T',  'AO2',  'CH',  'AO0',  'UH1',  'V',  'IY2',  'UW',  'OY0',  'UW2',  'M',  'IH1',  'ZH',  'AA0',  'AW0',  'G',  'AH2',  'ER2',  'IY1',  'AY1',  'JH',  'W',  'L',  'AY0',  'UW1',  'AE0',  'IY0',  'AH0',  'B',  'EH2',  'Y',  'HH',  'TH',  'AW1',  'DH',  'F',  'AW2',  'S',  'IH2',  'SH',  'EY1',  'EY0',  'Z',  'ER1',  'D',  'AA1',  'R',  '<eps>']
PHONEMES_DICT = {'sp': 0, 'AA2': 2, 'UW0': 4, 'OW0': 3, 'NG': 6, 'P': 10, 'N': 9, 'AE2': 1, 'OW1': 12, 'AO1': 13, 'ER0': 14, 'OY1': 15, 'IH0': 17, 'K': 19, 'AH1': 20, 'EH1': 21, 'EY2': 22, 'OY2': 23, 'T': 24, 'AO2': 25, 'CH': 26, 'UW1': 48, 'AO0': 27, 'UH1': 28, 'V': 29, 'IY2': 30, 'AY2': 5, 'OY0': 32, 'M': 34, 'IH1': 35, 'ZH': 36, 'Z': 66, 'AA0': 37, 'OW2': 8, 'G': 39, 'AH2': 40, 'EH2': 53, 'ER2': 41, 'IY1': 42, 'UW': 31, 'AY1': 43, 'JH': 44, 'W': 45, 'L': 46, 'AE1': 11, 'D': 68, 'AE0': 49, 'IY0': 50, 'AY0': 47, 'UH2': 16, 'HH': 55, 'UH0': 18, 'AW1': 57, 'AW2': 60, 'DH': 58, 'B': 52, 'F': 59, 'S': 61, 'IH2': 62, 'SH': 63, 'EH0': 7, 'AW0': 38, 'EY1': 64, 'EY0': 65, 'Y': 54, 'ER1': 67, 'UW2': 33, 'AH0': 51, '<eps>': 71, 'AA1': 69, 'R': 70, 'TH': 56}
PHONEME_SIZE = len(PHONEMES)
PHONEME_EPISILON = len(PHONEMES)-1
