# TARS
Top-notch Automatic speech Recognition System (TARS)

# Installation

Follow the instructions as the original wavenet implementation.

However, instead of installing 
`scikits.audiolab`, use `soundfile` instead, which supports Python 3.x

Also, when compiling Libsndfile from source / installing from brew, make sure to enable Flac support

MacOS: (Courtesy of [this person](https://github.com/facebookresearch/wav2letter/issues/3#issuecomment-361710074))

```
brew install libsndfile --with-lame --with-flac --with-libvorbis
brew link --overwrite libsndfile
```

Also, installed `ffmpeg`
![alt text](https://i.pinimg.com/originals/6a/42/ed/6a42ed5bdb29da3b2328f961deca15f8.jpg)


# Important files
preprocess.py - written by someone else,used to create lmfcc from the sound files and store those the labels in asset/data/preprocess folder
Conf.py - Parameters and alphabet mapping

train.py - Main file for training, run this to get the log files which you can plot in tensorboard
model.py - Creates the graph for the model
dataloder.py - The dataloading pipeline, loads data efficiently. Does the batching too!
test_ctc.py - Some experiments, can be ignored