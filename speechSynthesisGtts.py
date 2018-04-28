from gtts import gTTS

tts = gTTS(text='Clement, welcome to sweden', lang='en')
filename = 'temp.mp3'
tts.save(filename)