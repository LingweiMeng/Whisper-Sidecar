import json
import os
import random
import sys
from typing import List

import librosa
import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import AutoTokenizer

class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 processor,
                 mono=True,
                 language=None,
                 timestamps=False,
                 sample_rate=16000,
                 min_duration=0.5,
                 max_duration=30,
                 augment_config_path=None,
                 num_spks=2,
                 is_eval_dataset=False,
                 for_target_asr_eval=False
                 ):
        """
        Args:
            data_list_path: 
                Path to the data list file or the header file of the binary list
            processor: 
                Whisper's preprocessing tool, obtained via WhisperProcessor.from_pretrained
            mono:
                Whether to convert audio to mono channel; this must be True
            language: 
                Language of the fine-tuning data
            timestamps: 
                Whether to use timestamps during fine-tuning
            sample_rate: 
                Audio sample rate, default is 16000
            min_duration: 
                Audio shorter than this duration will be truncated, in seconds; cannot be less than 0.5, default is 0.5s
            max_duration: 
                Audio longer than this duration will be truncated, in seconds; cannot exceed 30, default is 30s
            augment_config_path: 
                Path to the data augmentation configuration parameter file
        """
        super(CustomDataset, self).__init__()
        assert min_duration >= 0.5, f"min_duration cannot be less than 0.5, current value: {min_duration}"
        assert max_duration <= 30, f"max_duration cannot be greater than 30, current value: {max_duration}"
        self.data_list_path = data_list_path
        self.processor = processor
        self.data_list_path = data_list_path
        self.sample_rate = sample_rate
        self.mono = mono
        self.language = language
        self.timestamps = timestamps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.vocab = self.processor.tokenizer.get_vocab()
        self.startoftranscript = self.vocab['<|startoftranscript|>']
        self.endoftext = self.vocab['<|endoftext|>']
        if '<|nospeech|>' in self.vocab.keys():
            self.nospeech = self.vocab['<|nospeech|>']
            self.timestamp_begin = None
        else:
            # Compatible with old model version
            self.nospeech = self.vocab['<|nocaptions|>']
            self.timestamp_begin = self.vocab['<|notimestamps|>'] + 1

        self.split_symbol= "</s>"

        self.data_list: List[dict] = []
        # Load data list
        self._load_data_list()
        # Data augmentation configuration parameters
        self.augment_configs = None
        self.noises_path = None
        self.speed_rates = None
        if augment_config_path:
            with open(augment_config_path, 'r', encoding='utf-8') as f:
                self.augment_configs = json.load(f)
        
        self.num_spks = num_spks
        self.is_eval_dataset = is_eval_dataset
        self.for_target_asr_eval = for_target_asr_eval

    def _load_data_list(self):
        with open(self.data_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.data_list = []
        for line in tqdm(lines, desc='load data list'):
            if isinstance(line, str):
                line = json.loads(line)
            if not isinstance(line, dict): continue
            # Skip audio that exceeds the length limit
            if line["duration"] < self.min_duration:
                continue
            if self.max_duration != -1 and line["duration"] > self.max_duration:
                continue
            self.data_list.append(dict(line))

    # Obtain audio data, sampling rate, and text from the data list
    def _get_list_data(self, idx):
        data_list = self.data_list[idx]
        audio_file = data_list["audio"]['path']
        transcript = data_list["sentence"]
        transcript_timestamps = data_list["sentences"] if 'sentences' in data_list.keys() else None
        language = data_list["language"] if 'language' in data_list.keys() else None
        speakers = data_list["speakers"] if 'speakers' in data_list.keys() else None
        if 'start_time' not in data_list["audio"].keys():
            sample, sample_rate = soundfile.read(audio_file, dtype='float32')
        else:
            start_time, end_time = data_list["audio"]["start_time"], data_list["audio"]["end_time"]
            sample, sample_rate = self.slice_from_file(audio_file, start=start_time, end=end_time)
        sample = sample.T
        if self.mono:
            sample = librosa.to_mono(sample)
        if self.augment_configs:
            sample, sample_rate = self.augment(sample, sample_rate)
        if self.sample_rate != sample_rate:
            sample = self.resample(sample, orig_sr=sample_rate, target_sr=self.sample_rate)
        return sample, sample_rate, transcript, transcript_timestamps, language, speakers, audio_file

    def _load_timestamps_transcript(self, transcript: List[dict]):
        assert isinstance(transcript, list), f"The transcript type should list , instead of {type(transcript)}"
        labels = self.processor.tokenizer.prefix_tokens[:3]
        for t in transcript:
            # Encode the target text into the label index
            start = round(t['start'], 2)
            if self.timestamp_begin is None:
                start = self.vocab[f'<|{start:.2f}|>'] if f'<|{start:.2f}|>' in self.vocab.keys() else self.vocab[f'<|{start+0.01:.2f}|>']
            else:
                start = self.timestamp_begin + round(start * 100) // 2
            end = round(t['end'], 2)
            if self.timestamp_begin is None:
                end = self.vocab[f'<|{end:.2f}|>'] if f'<|{end:.2f}|>' in self.vocab.keys() else self.vocab[f'<|{end-0.01:.2f}|>']
            else:
                end = self.timestamp_begin + round(end * 100) // 2
            label = self.processor(text=t['text']).input_ids[4:-1]
            labels.extend([start])
            labels.extend(label)
            labels.extend([end])
        return labels + [self.endoftext]

    def __getitem__(self, idx):
        try:
            # Get audio data, sampling rate, and text from the data list
            sample, sample_rate, transcript, transcript_timestamps, language, speakers, audio_file = self._get_list_data(idx=idx)

            data = {"raw_audio": sample}
            self.processor.tokenizer.set_prefix_tokens(language=language if language is not None else self.language)
            if len(transcript) > 0:
                # Load text with a timestamp
                if self.timestamps and (random.random() < 0.5 and not self.is_eval_dataset):
                    data["labels"] = [self._load_timestamps_transcript([trans]) for trans in transcript_timestamps]
                    if len(data["labels"]) > self.num_spks:
                        raise ValueError(f"The number of speakers in audio {idx} is greater than {self.num_spks}")
                    elif len(data["labels"]) < self.num_spks:
                        data["labels"].extend([self.processor.tokenizer.prefix_tokens + [2411] + [self.endoftext] for _ in range(0, self.num_spks - len(data["labels"]))])
                else:
                    # split text for speakers
                    transcript = [t.strip() for t in transcript.split(self.split_symbol)]
                    if len(transcript) > self.num_spks:
                        raise ValueError(f"The number of speakers in audio {idx} is greater than {self.num_spks}")
                    if len(transcript) < self.num_spks:
                        transcript.extend([' .' for _ in range(self.num_spks - len(transcript))])
                    # Obtain log-Mel features and label IDs
                    data["labels"] = self.processor(text=transcript).input_ids

            else:
                # # If there is no text, use the <|nospeech|> tag 
                # data['labels'] = [self.startoftranscript, self.nospeech, self.endoftext]
                # data['labels'] = self.processor(text=' .').input_ids
                raise ValueError(f"Audio {idx} has no text")

            data['speakers'] = speakers
            # data['train_split'] = "train-360" if "train-360" in audio_file else "train-100"
            data['train_split'] = "all"
            data['is_eval_dataset'] = self.is_eval_dataset
            data['for_target_asr_eval'] = self.for_target_asr_eval

            return data
        except Exception as e:
            print(f'Error reading data, data index: {idx}, error message: {e}', file=sys.stderr)
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def slice_from_file(file, start, end):
        sndfile = soundfile.SoundFile(file)
        sample_rate = sndfile.samplerate
        duration = round(float(len(sndfile)) / sample_rate, 3)
        start = round(start, 3)
        end = round(end, 3)
        # from the end
        if start < 0.0: start += duration
        if end < 0.0: end += duration
        # Ensure data does not exceed boundaries
        if start < 0.0: start = 0.0
        if end > duration: end = duration
        if end < 0.0:
            raise ValueError("Slice end position (%f s) out of bounds" % end)
        if start > end:
            raise ValueError("The slice start position (%f s) is later than the slice end position (%f s)" % (start, end))
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        sample = sndfile.read(frames=end_frame - start_frame, dtype='float32')
        return sample, sample_rate

    # data augment
    def augment(self, sample, sample_rate):
        for config in self.augment_configs:
            if config['type'] == 'speed' and random.random() < config['prob']:
                if self.speed_rates is None:
                    min_speed_rate, max_speed_rate, num_rates = config['params']['min_speed_rate'], \
                        config['params']['max_speed_rate'], config['params']['num_rates']
                    self.speed_rates = np.linspace(min_speed_rate, max_speed_rate, num_rates, endpoint=True)
                rate = random.choice(self.speed_rates)
                sample = self.change_speed(sample, speed_rate=rate)
            if config['type'] == 'shift' and random.random() < config['prob']:
                min_shift_ms, max_shift_ms = config['params']['min_shift_ms'], config['params']['max_shift_ms']
                shift_ms = random.randint(min_shift_ms, max_shift_ms)
                sample = self.shift(sample, sample_rate, shift_ms=shift_ms)
            if config['type'] == 'volume' and random.random() < config['prob']:
                min_gain_dBFS, max_gain_dBFS = config['params']['min_gain_dBFS'], config['params']['max_gain_dBFS']
                gain = random.randint(min_gain_dBFS, max_gain_dBFS)
                sample = self.volume(sample, gain=gain)
            if config['type'] == 'resample' and random.random() < config['prob']:
                new_sample_rates = config['params']['new_sample_rates']
                new_sample_rate = np.random.choice(new_sample_rates)
                sample = self.resample(sample, orig_sr=sample_rate, target_sr=new_sample_rate)
                sample_rate = new_sample_rate
            if config['type'] == 'noise' and random.random() < config['prob']:
                min_snr_dB, max_snr_dB = config['params']['min_snr_dB'], config['params']['max_snr_dB']
                if self.noises_path is None:
                    self.noises_path = []
                    noise_dir = config['params']['noise_dir']
                    if os.path.exists(noise_dir):
                        for file in os.listdir(noise_dir):
                            self.noises_path.append(os.path.join(noise_dir, file))
                noise_path = random.choice(self.noises_path)
                snr_dB = random.randint(min_snr_dB, max_snr_dB)
                sample = self.add_noise(sample, sample_rate, noise_path=noise_path, snr_dB=snr_dB)
        return sample, sample_rate

    @staticmethod
    def change_speed(sample, speed_rate):
        if speed_rate == 1.0:
            return sample
        if speed_rate <= 0:
            raise ValueError("The speed rate should be greater than zero.")
        old_length = sample.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        sample = np.interp(new_indices, old_indices, sample).astype(np.float32)
        return sample

    @staticmethod
    def shift(sample, sample_rate, shift_ms):
        duration = sample.shape[0] / sample_rate
        if abs(shift_ms) / 1000.0 > duration:
            raise ValueError("The absolute value of shift_ms should be less than the duration of the audio.")
        shift_samples = int(shift_ms * sample_rate / 1000)
        if shift_samples > 0:
            sample[:-shift_samples] = sample[shift_samples:]
            sample[-shift_samples:] = 0
        elif shift_samples < 0:
            sample[-shift_samples:] = sample[:shift_samples]
            sample[:-shift_samples] = 0
        return sample

    @staticmethod
    def volume(sample, gain):
        sample *= 10.**(gain / 20.)
        return sample

    @staticmethod
    def resample(sample, orig_sr, target_sr):
        sample = librosa.resample(sample, orig_sr=orig_sr, target_sr=target_sr)
        return sample

    def add_noise(self, sample, sample_rate, noise_path, snr_dB, max_gain_db=300.0):
        noise_sample, sr = librosa.load(noise_path, sr=sample_rate)
        # Standardize audio volume to ensure that noise is not too loud.
        target_db = -20
        gain = min(max_gain_db, target_db - self.rms_db(sample))
        sample *= 10. ** (gain / 20.)

        sample_rms_db, noise_rms_db = self.rms_db(sample), self.rms_db(noise_sample)
        noise_gain_db = min(sample_rms_db - noise_rms_db - snr_dB, max_gain_db)
        noise_sample *= 10. ** (noise_gain_db / 20.)

        if noise_sample.shape[0] < sample.shape[0]:
            diff_duration = sample.shape[0] - noise_sample.shape[0]
            noise_sample = np.pad(noise_sample, (0, diff_duration), 'wrap')
        elif noise_sample.shape[0] > sample.shape[0]:
            start_frame = random.randint(0, noise_sample.shape[0] - sample.shape[0])
            noise_sample = noise_sample[start_frame:sample.shape[0] + start_frame]
        sample += noise_sample
        return sample

    @staticmethod
    def rms_db(sample):
        mean_square = np.mean(sample ** 2)
        return 10 * np.log10(mean_square)
