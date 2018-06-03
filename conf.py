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

intToPhoneme = {0:'sp', 1: 'AE2', 2: 'AA2', 3: 'OW0', 4: 'UW0', 5: 'AY2', 6: 'NG', 7: 'EH0', 8: 'OW2', 9: 'N', 10: 'P', 11: 'AE1', 12: 'OW1', 13: 'AO1', 14: 'ER0', 15: 'OY1', 16: 'UH2', 17: 'IH0', 18: 'UH0', 19: 'K', 20: 'AH1', 21: 'EH1', 22: 'EY2', 23: 'OY2', 24: 'T', 25: 'AO2', 26: 'CH', 27: 'AO0', 28: 'UH1', 29: 'V', 30: 'IY2', 31: 'UW', 32: 'OY0', 33: 'UW2', 34: 'M', 35: 'IH1', 36: 'ZH', 37: 'AA0', 38: 'AW0', 39: 'G', 40: 'AH2', 41: 'ER2', 42: 'IY1', 43: 'AY1', 44: 'JH', 45: 'W', 46: 'L', 47: 'AY0', 48: 'UW1', 49: 'AE0', 50: 'IY0', 51: 'AH0', 52: 'B', 53: 'EH2', 54: 'Y', 55: 'HH', 56: 'TH', 57: 'AW1', 58: 'DH', 59: 'F', 60: 'AW2', 61: 'S', 62: 'IH2', 63: 'SH', 64: 'EY1', 65: 'EY0', 66: 'Z', 67: 'ER1', 68: 'D', 69: 'AA1', 70: 'R', 71: '<eps>'}
phonemeToInt = {'sp': 0, 'AA2': 2, 'UW0': 4, 'OW0': 3, 'NG': 6, 'P': 10, 'N': 9, 'AE2': 1, 'OW1': 12, 'AO1': 13, 'ER0': 14, 'OY1': 15, 'IH0': 17, 'K': 19, 'AH1': 20, 'EH1': 21, 'EY2': 22, 'OY2': 23, 'T': 24, 'AO2': 25, 'CH': 26, 'UW1': 48, 'AO0': 27, 'UH1': 28, 'V': 29, 'IY2': 30, 'AY2': 5, 'OY0': 32, 'M': 34, 'IH1': 35, 'ZH': 36, 'Z': 66, 'AA0': 37, 'OW2': 8, 'G': 39, 'AH2': 40, 'EH2': 53, 'ER2': 41, 'IY1': 42, 'UW': 31, 'AY1': 43, 'JH': 44, 'W': 45, 'L': 46, 'AE1': 11, 'D': 68, 'AE0': 49, 'IY0': 50, 'AY0': 47, 'UH2': 16, 'HH': 55, 'UH0': 18, 'AW1': 57, 'AW2': 60, 'DH': 58, 'B': 52, 'F': 59, 'S': 61, 'IH2': 62, 'SH': 63, 'EH0': 7, 'AW0': 38, 'EY1': 64, 'EY0': 65, 'Y': 54, 'ER1': 67, 'UW2': 33, 'AH0': 51, '<eps>': 71, 'AA1': 69, 'R': 70, 'TH': 56}

