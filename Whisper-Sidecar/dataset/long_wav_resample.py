# resamle long wav (>30s) to 16k 30s, and update the jsonl file


import jsonlines
import os

from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf



jsonl_file = "./dataset/librispeech3mix_test.jsonl"
temp_file = "./dataset/librispeech3mix_test_temp.jsonl"
# time strench long wav (>30s) to 16k 30s, and update the jsonl line in-place

with jsonlines.open(jsonl_file, 'r') as reader, jsonlines.open(temp_file, 'w') as writer:
    for obj in reader:
        wav_path = obj['audio']['path']
        duration = obj['duration']
        sentences = obj['sentences']
        # resample wav
        if duration > 30.1:
            print(wav_path, duration)
            wav = AudioSegment.from_file(wav_path)
            target_len = 30.0 * 1000
            speed_up_rate = len(wav) / target_len
            wav = wav.speedup(playback_speed=speed_up_rate)
            wav = wav[:target_len]
            wav.export(wav_path, format="wav")
            print(speed_up_rate)
            obj['duration'] = 30
            for sentence in sentences:
                sentence['start'] = sentence['start'] / speed_up_rate
                sentence['end'] = sentence['end'] / speed_up_rate
                if sentence['start'] > 30:
                    sentence['start'] = 30
                if sentence['end'] > 30:
                    sentence['end'] = 30
            obj['sentences'] = sentences
        elif duration > 30.0:
            wav, sr = sf.read(wav_path)
            wav = wav[:int(16000 * 30)]
            obj['duration'] = 30 if duration > 30 else duration
            for sentence in sentences:
                sentence['start'] = sentence['start'] if sentence['start'] < 30 else 30
                sentence['end'] = sentence['end'] if sentence['end'] < 30 else 30
            obj['sentences'] = sentences
            sf.write(wav_path, wav, sr)

        
        writer.write(obj)