import numpy as np
import dataloader as data
import glob,csv,librosa,functools,soundfile,multiprocessing.pool,os,os.path,sklearn.preprocessing,nltk,editdistance

# data path
_data_path = "data/real/"

# process VCTK corpus
def process_speaker_folder(speaker, audio_directory, transcription_directory,
                           target_directory):
    csv_elements = []

    print('Processing ' + speaker)

    if not os.path.exists(transcription_directory + '/' + speaker):
        raise RuntimeError

    os.mkdir(target_directory + '/' + speaker)

    audio_files = os.listdir(audio_directory + '/' + speaker)
    audio_files.sort()
    for audio_file in audio_files:
        transcript_filepath = (transcription_directory + '/' + speaker + '/'
                               + os.path.splitext(audio_file)[0] + '.txt')
        if not os.path.exists(transcript_filepath):
            raise RuntimeError
        transcript_file = open(transcript_filepath)
        labels = data.str2index(transcript_file.read())
        csv_elements.append([audio_file] + labels)
        transcript_file.close()

        target_filepath = (target_directory + '/' + speaker + '/' +
                           os.path.splitext(audio_file)[0] + '_mfcc.npy')
        if os.path.exists(target_filepath):
            continue
        audio_filepath = audio_directory + '/' + speaker + '/' + audio_file
        audio, _ = librosa.load(audio_filepath, sr=16000)
        mfcc = librosa.feature.mfcc(audio, sr=16000)
        np.save(target_filepath, mfcc)

        if not len(labels) <= mfcc.shape[1]:
            raise ValueError('Transcript longer than MFCC sequence in '
                             + transcript_filepath)

    return csv_elements


# Preprocess VCTK corpus (both synthetic and standard) -- new version
#
# Walk through the hierarchy of audio files (in audio_directory) and
# compute the associated MFCC.
# Then save these MFCC in a similar hierarchy in target_directory.
#
# At the same time, open the transcription associated to each audio file walking
# through (transcription_directory) and
# write the corresponding labels in the output comma-separated CSV file.
# The first column in the CSV file is the basename and the next columns are the
# labels (in number format from dataloader.str2index).
def process_vctk(csv_filename, audio_directory, transcription_directory,
                 target_directory):
    speakers = os.listdir(audio_directory)
    speakers.sort()
    pool = multiprocessing.pool.Pool()
    all_csv_elements = pool.map(
        functools.partial(process_speaker_folder,
                          audio_directory=audio_directory,
                          transcription_directory=transcription_directory,
                          target_directory=target_directory),
        speakers
    )

    csv_file = open(csv_filename, mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    for speaker_csv_elements in all_csv_elements:
        for sample_csv_elements in speaker_csv_elements:
            csv_writer.writerow(sample_csv_elements)
    csv_file.close()


def normalize_vctk(source_directory, target_directory):
    if not os.path.exists(source_directory):
        raise RuntimeError
    if not os.path.exists(target_directory):
        raise RuntimeError

    scaler = sklearn.preprocessing.StandardScaler()

    speakers = os.listdir(source_directory)
    speakers.sort()
    for speaker in speakers:
        mfcc_files = os.listdir(source_directory + '/' + speaker)
        mfcc_files.sort()
        for mfcc_file in mfcc_files:
            mfcc_filepath = source_directory + '/' + speaker + '/' + mfcc_file
            mfcc = np.load(mfcc_filepath)
            scaler.partial_fit(mfcc.T)

    for speaker in speakers:
        os.mkdir(target_directory + '/' + speaker)
        mfcc_files = os.listdir(source_directory + '/' + speaker)
        mfcc_files.sort()
        for mfcc_file in mfcc_files:
            mfcc_filepath = source_directory + '/' + speaker + '/' + mfcc_file
            mfcc = np.load(mfcc_filepath).T
            mfcc = scaler.transform(mfcc)
            np.save(target_directory + '/' + speaker + '/' + mfcc_file, mfcc.T)

    print(scaler.get_params())


# process LibriSpeech corpus
def process_libri(csv_file, category):

    parent_path = _data_path + 'LibriSpeech/' + category + '/'
    labels, wave_files = [], []

    # create csv writer
    writer = csv.writer(csv_file, delimiter=',')

    # read directory list by speaker
    speaker_list = glob.glob(parent_path + '*')
    for spk in speaker_list:

        # read directory list by chapter
        chapter_list = glob.glob(spk + '/*/')
        for chap in chapter_list:

            # read label text file list
            txt_list = glob.glob(chap + '/*.txt')
            for txt in txt_list:
                with open(txt, 'rt') as f:
                    records = f.readlines()
                    for record in records:
                        # parsing record
                        field = record.split('-')  # split by '-'
                        speaker = field[0]
                        chapter = field[1]
                        field = field[2].split()  # split field[2] by ' '
                        utterance = field[0]  # first column is utterance id

                        # wave file name
                        wave_file = parent_path + '%s/%s/%s-%s-%s.flac' % \
                                                  (speaker, chapter, speaker, chapter, utterance)
                        wave_files.append(wave_file)

                        # label index
                        labels.append(data.str2index(' '.join(field[1:])))  # last column is text label
                        #print(data.str2index(' '.join(field[1:])))

    # save results
    for i, (wave_file, label) in enumerate(zip(wave_files, labels)):
        fn = wave_file.split('/')[-1]
        target_filename = 'asset/data/preprocess/mfcc/' + fn + '.npy'
        # if os.path.exists( target_filename ):
        #    continue
        # print info
        print("LibriSpeech corpus preprocessing (%d / %d) - '%s']" % (i, len(wave_files), wave_file))

        # load flac file
        wave, sr = soundfile.read(wave_file)

        # get mfcc feature
        mfcc = librosa.feature.mfcc(wave, sr=16000)

        # save result ( exclude small mfcc data to prevent ctc loss )
        if len(label) < mfcc.shape[1]:
            # filename

            # save meta info
            writer.writerow([fn] + label)

            # save mfcc
            #np.save(target_filename, mfcc, allow_pickle=False)


def getPhonemeIntMaps():
    phonemeSet = set()
    a = nltk.corpus.cmudict.dict()

    keys = a.keys()
    for key in keys:
        for i in range(0,len(a[key])):
            for j in range(0, len(a[key][i])):
                phonemeSet.add(a[key][i][j])

    intToPhoneme = dict(enumerate(phonemeSet))
    phonemeToInt = {v: k for k, v in intToPhoneme.items()}
    return intToPhoneme,phonemeToInt

def getPhonemes(word):
    return nltk.corpus.cmudict.dict()[word][0]

def getWord(phonemes):
    corpus = nltk.corpus.cmudict.dict()
    closestWord = 'NULL'
    distance = 10000#Arbitrary large number

    keys = corpus.keys()
    for key in keys:
        for i in range(0,len(corpus[key])):
            newDist = editdistance.eval(phonemes,corpus[key][i])
            if newDist < distance:
                closestWord = key
                distance = newDist
                print(closestWord)
                print(distance)

    return closestWord

# Create directories
if not os.path.exists('asset/data/preprocess'):
    os.makedirs('asset/data/preprocess')
if not os.path.exists('asset/data/preprocess/meta'):
    os.makedirs('asset/data/preprocess/meta')
if not os.path.exists('asset/data/preprocess/mfcc'):
    os.makedirs('asset/data/preprocess/mfcc')

if __name__ == "__main__":
    #intToPhoneme, phonemeToInt = getPhonemeIntMaps()
    #print(getWord(getPhonemes('campaign')))

    # Run pre-processing for training
    csv_f = open('asset/data/preprocess/meta/train.csv', 'w')
    process_libri(csv_f, 'dev-clean')
    # process_vctk(csv_f) #uncomment and comment out libri to switch to VCTK
    csv_f.close()