from gtts import gTTS
import os
import time

path = "../VCTK/VCTK-Corpus/txt/"

dirs = os.listdir(path)
start_time = time.time()
for diri in range(0,len(dirs)):
    print("Starting to generate sound files for folder "+dirs[diri])
    filenames = os.listdir(path+dirs[diri])
    sentences = []

    if not os.path.exists("output/"+dirs[diri]):
        os.makedirs("output/"+dirs[diri])

    for filename in filenames:
        f = open(path+dirs[diri]+"/"+filename,"r")
        sentences.append(f.readline())

    tts = []
    for i in range(0,len(sentences)):
        tts.append(gTTS(text=sentences[i], lang="en"))

    for i in range(0,len(sentences)):
        tts[i].save("output/"+dirs[diri]+"/"+filenames[i][:-3]+"mp3")

    print("[" + ('%.1f' % round(100 * (diri + 1) / len(dirs), 2)) + "%] directory " + dirs[diri] + " done")
    timeElapsed = (time.time() - start_time)
    timeLeft = ((timeElapsed) / (diri + 1)) * (len(dirs) - diri - 1)
    print("Time elapsed: %s seconds" % (int(timeElapsed)))
    print("Time left: %s seconds" % (int(timeLeft)))

