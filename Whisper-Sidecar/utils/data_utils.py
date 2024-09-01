import re
from dataclasses import dataclass
from typing import Any, List, Dict, Union

import torch
from zhconv import convert
import random
import os
import soundfile as sf
import numpy as np


def remove_punctuation(text: str or List[str]):
    punctuation = '!,.;:?、！，。；：？_�'
    if isinstance(text, str):
        text = re.sub(r'[{}]+'.format(punctuation), '', text).strip()
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = re.sub(r'[{}]+'.format(punctuation), '', t).strip()
            result_text.append(t)
        return result_text
    else:
        raise Exception(f'{type(text)} is not supported')


# Convert traditional Chinese into simplified Chinese
def to_simple(text: str or List[str]):
    if isinstance(text, str):
        text = convert(text, 'zh-cn')
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = convert(t, 'zh-cn')
            result_text.append(t)
        return result_text
    else:
        raise Exception(f'{type(text)} is not supported')


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor, num_spks=2, soft_prompt_len=0, target_asr=False):
        self.processor = processor
        self.num_spks = num_spks
        self.soft_prompt_len = soft_prompt_len
        if soft_prompt_len > 0:
            self.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|startofprev|>')
            self.prev_sot_token_id = self.decoder_start_token_id
        else:    
            self.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|startoftranscript|>')
        self.target_asr = target_asr

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        # input_features = [{"input_features": feature["input_features"][0]} for feature in features]
        # batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        is_eval_dataset = features[0]['is_eval_dataset']
        for_target_asr_eval = features[0]['for_target_asr_eval']

        batch = {}
        raw_audios = [feature['raw_audio'] for feature in features]

        if self.target_asr and random.random() < 0.2 and not is_eval_dataset:
            speakers = [f['speakers'] for f in features]       # (B, num_spks)
            # randomly select a speaker as the target speaker, and enroll the prompt audio.
            target_idx = [random.choice(range(self.num_spks)) for _ in range(len(speakers))]
            target_speaker = [speakers[i][idx] for i, idx in enumerate(target_idx)]
            train_split = [f['train_split'] for f in features]
            enroll_audio_path = [os.path.join("./dataset/enroll_audios/", train_split, target_spk, target_spk + ".wav") for target_spk, train_split in zip(target_speaker, train_split)]
            enroll_audios = [sf.read(path)[0] for path in enroll_audio_path]
            raw_audios = [np.concatenate([enroll[:int(16000 * 3)], np.zeros(int(16000 * 0.2)), sample]) for enroll, sample in zip(enroll_audios, raw_audios)]
            batch["target_speaker"] = torch.tensor(target_idx, dtype=torch.long)
        elif self.target_asr and is_eval_dataset and for_target_asr_eval:
            speakers = [f['speakers'] for f in features]
            # select all speakers as the target speaker, and enroll the prompt audio
            target_speaker = [speakers[i][j] for i in range(len(speakers)) for j in range(self.num_spks)]
            train_split = [f['train_split'] for f in features for _ in range(self.num_spks)]
            enroll_audio_path = [os.path.join("./dataset/enroll_audios/", train_split, target_spk, target_spk + ".wav") for target_spk, train_split in zip(target_speaker, train_split)]
            enroll_audios = [sf.read(path)[0] for path in enroll_audio_path]
            # interleave repeat raw_audios num_spks times
            raw_audios = [raw_audios[i//self.num_spks] for i in range(len(raw_audios)*self.num_spks)]
            raw_audios = [np.concatenate([enroll[:int(16000 * 3)], np.zeros(int(16000 * 0.2)), sample]) for enroll, sample in zip(enroll_audios, raw_audios)]
            batch["target_speaker"] = torch.tensor(list(range(self.num_spks)) * len(features), dtype=torch.long)

        batch["input_features"] = self.processor(audio=raw_audios, sampling_rate=16000, padding='max_length', return_tensors="pt").input_features
        # repeat num_spks times
        batch["input_features"] = batch["input_features"].repeat_interleave(self.num_spks, dim=0)

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"][i]} for feature in features for i in range(len(feature["labels"]))]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # insert fake token ids for soft prompt
        if self.soft_prompt_len > 0:
            soft_prompt = torch.ones((labels.shape[0], self.soft_prompt_len+1), dtype=torch.long) * self.prev_sot_token_id
            labels = torch.cat([soft_prompt, labels], dim=1)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
