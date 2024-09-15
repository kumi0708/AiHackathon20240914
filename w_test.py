from io import BytesIO

import numpy as np
import soundfile as sf
import speech_recognition as sr
import whisper
import time

if __name__ == "__main__":
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
        if result["text"] == "":
            print("音声認識失敗")
        else:
            print(result["text"])
            #time.sleep(1) # 3秒待つ