import google.cloud.texttospeech
import os
import os.path
import random


def navigate_and_generate(top_directory, output_directory):
    os.chdir(top_directory)

    pages = os.listdir(top_directory)
    pages.sort()
    for page in pages:
        if not os.path.exists(output_directory + '/' + page):
            os.mkdir(output_directory + '/' + page)

        txt_filenames = os.listdir(top_directory + '/' + page)
        txt_filenames.sort()
        for txt_filename in txt_filenames:
            print (page + '/' + txt_filename + ' ...'),

            mp3_file_path = (output_directory + '/' + page + '/'
                            + os.path.splitext(txt_filename)[0] + '.mp3')
            if os.path.exists(mp3_file_path):
                print ' skipped'
                continue

            txt_file = open(page + '/' + txt_filename, 'r')
            text = txt_file.read()
            txt_file.close()

            response = synthesize_text(
                text, random.choice(['A', 'B', 'C', 'D', 'E', 'F']))
            mp3_file = open(mp3_file_path, 'wb')
            mp3_file.write(response)
            mp3_file.close()

            print ' done'


def synthesize_text(text, wavenet_letter):
    client = google.cloud.texttospeech.TextToSpeechClient()

    input_text = google.cloud.texttospeech.types.SynthesisInput(text=text)
    voice = google.cloud.texttospeech.types.VoiceSelectionParams(
        language_code='en-US', name=('en-US-Wavenet-' + wavenet_letter))
    audio_config = google.cloud.texttospeech.types.AudioConfig(
        audio_encoding=google.cloud.texttospeech.enums.AudioEncoding.MP3)

    response = client.synthesize_speech(input_text, voice, audio_config)
    return response.audio_content
