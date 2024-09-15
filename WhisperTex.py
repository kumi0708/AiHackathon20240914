from io import BytesIO

import numpy as np
import soundfile as sf
import speech_recognition as sr
import whisper
import time
from openai_adapter import OpenAIAdapter
from voicevox_adapter import VoicevoxAdapter
from play_sound import PlaySound

adapter = OpenAIAdapter()
voicevox_adapter = VoicevoxAdapter()
play_sound = PlaySound("ヘッドホン")



if __name__ == "__main__":
    #model = whisper.load_model("large")
    model = whisper.load_model("small")

    recognizer = sr.Recognizer()
    while True:
        # 「マイクから音声を取得」参照
        with sr.Microphone(sample_rate=16_000) as source:
            print("なにか話してください")
            audio = recognizer.listen(source)

        print("音声処理中 ...")
        # 「音声データをWhisperの入力形式に変換」参照
        wav_bytes = audio.get_wav_data()
        wav_stream = BytesIO(wav_bytes)
        audio_array, sampling_rate = sf.read(wav_stream)
        audio_fp32 = audio_array.astype(np.float32)

        #result = model.transcribe(audio_fp32, fp16=False)
        result = model.transcribe(audio_fp32, fp16=False,language="ja")

        #result["text"]には音声認識の結果が入っているか調べる
        if result["text"] == "":
            print("音声認識失敗")
        else:
            print('\033[31m'+"input :"+result["text"]+'\033[0m')
            response_text = adapter.create_chat(result["text"])
            print('\033[32m'+"output:"+response_text+'\033[0m')

            #print(adapter.chat_log)
            data, rate = voicevox_adapter.get_voice(response_text)
            play_sound.play_sound(data, rate)
            time.sleep(3) # 3秒待つ


        