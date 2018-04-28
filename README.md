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