import google.generativeai as genai
from openai import OpenAI
import pyaudio
import speech_recognition as sr
import time
import os

from faster_whisper import WhisperModel
whisper_size = 'base'
num_cores = os.cpu_count()
generation_Config = {"temperature": 0.7, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
wake_word = "hello"
listening_for_wake_word = True

whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores,
    num_workers=num_cores
)

OPENAI_KEY = ''
client = OpenAI(api_key=OPENAI_KEY)
GOOGLE_API_KEY = 'AIzaSyAXfPBcA_hnqh-4KOnbacXWNyxnFUR89gE'
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash-latest')

# response = model.generate_content(input('Ask Gemini: '))
# print(response.text)

convo = model.start_chat()

system_message = '''SYSTEM MESSAGE: You are being used to power a voice assistant and should respond as so. 
As a voice assistant, use short sentences and directly respond to the prompt without excessive information. 
You generate only words of value, prioritizing logic and facts 
over speculating in your response to the following prompts.'''

system_message = system_message.replace('\n', '')

convo.send_message(system_message)


r = sr.Recognizer()
source = sr.Microphone()


def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    stream_start = False

    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        response_format="pcm",
        input=text,
    ) as response:
        silent_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            elif max(chunk) > silent_threshold:
                player_stream.write(chunk)
                stream_start = True


def wav_to_text(audiopath):
    segments, _ = whisper_model.transcribe(audiopath)
    text = ''.join(segment.text for segment in segments)
    return text


def listen_for_wake_word(audio):
    global listening_for_wake_word
    wake_audio_path = 'wake_detect.wav'
    with open(wake_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    text_input = wav_to_text(wake_audio_path)

    if wake_word in text_input.lower().strip():
        print('Wake word detected. Please speak your prompt')
        listen_for_wake_word = False


def prompt_gpt(audio):
    global listening_for_wake_word

    try:
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())

        prompt_text = wav_to_text(prompt_audio_path)

        if len(prompt_text.strip()) == 0:
            print("Please speak again")
            listening_for_wake_word = True
        else:
            print('User: ' + prompt_text)
            convo.send_message(prompt_text)
            output = convo.last.text
            print("Gemini: " + output)
            speak(output)
            print('\nSay', wake_word, 'to wake me up. \n')
            listening_for_wake_word = True

    except Exception as e:
        print('prompt error: ', e)


def callback(recognizer, audio):
    global listening_for_wake_word

    if listening_for_wake_word:  
        listen_for_wake_word(audio)  #
    else:
        prompt_gpt(audio)



def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
        
    print('\nSay', wake_word, 'to wake me up.\n')

    r.listen_in_background(source, callback)
    while True:
        time.sleep(0.5)


if __name__ == '__main__':
    start_listening()
