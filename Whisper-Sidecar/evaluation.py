import argparse
import functools
import gc
import os
import itertools
import math
import re

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperProcessor
from model.modeling_whisper_sidecar import WhisperSidecarForConditionalGeneration

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding, remove_punctuation, to_simple
from metrics.compute_metrics import compute_wer
from utils.dataset import CustomDataset
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("test_data", type=str, default="dataset/libri2mix_test.jsonl", help="the path of the test set")
add_arg("model_path", type=str, default=r"output/checkpoint.pt", help="checkpoint path for evaluation")
add_arg("batch_size", type=int, default=32)
add_arg("num_workers", type=int, default=8)
add_arg("language", type=str, default="en", help="en or zh. if it is None, it is trained to be multilingual")
add_arg("remove_pun", type=bool, default=True,)
add_arg("timestamps", type=strtobool, default='false', help="whether to use timestamp data during training")
add_arg("min_audio_len", type=float, default=0.5, help="min audio length, in seconds")
add_arg("max_audio_len", type=float, default=30, help="max audio length, in seconds")
add_arg("local_files_only", type=bool, default=False, help="whether to only load the model locally")
add_arg("task", type=str, default="transcribe", choices=['transcribe', 'translate'])
add_arg("metric", type=str, default="wer", choices=['cer', 'wer'],)

# the following parameters need to be manually set to match those used during training.
add_arg("num_spks", type=int, default=2, help="max number of speakers in the training set")
add_arg("sidecar_loc", type=int, default=1, help="location of sidecar")
add_arg("soft_prompt_len", type=int, default=4, help="soft prompt in decoder input")
add_arg("target_asr", type=strtobool, default='false', help="whether to train the target asr task")

args = parser.parse_args()
print_arguments(args)

assert 'openai' == os.path.dirname(args.model_path) or os.path.exists(args.model_path), \
    f"The model file {args.model_path} does not exist."
os.system(f"cp -r {os.path.split(args.model_path)[0]}/tokenizer/* {args.model_path}")

processor = WhisperProcessor.from_pretrained(args.model_path,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)

model = WhisperSidecarForConditionalGeneration.from_pretrained(args.model_path,
                                                        device_map="auto",
                                                        local_files_only=args.local_files_only,
                                                        sidecar_loc=args.sidecar_loc,
                                                        num_spks=args.num_spks,
                                                        soft_prompt_len=args.soft_prompt_len,
                                                        target_asr=args.target_asr,
                                                        for_target_asr_eval=args.target_asr)
model.eval()

test_dataset = CustomDataset(data_list_path=args.test_data,
                             processor=processor,
                             timestamps=args.timestamps,
                             min_duration=args.min_audio_len,
                             max_duration=args.max_audio_len,
                             num_spks=args.num_spks,
                             is_eval_dataset=True,
                             for_target_asr_eval=args.target_asr
                             )
print(f"Test data: {len(test_dataset)}")

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, num_spks=args.num_spks, soft_prompt_len=args.soft_prompt_len, target_asr=args.target_asr)
eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, collate_fn=data_collator)

if args.soft_prompt_len > 0:
    model.generation_config.decoder_start_token_id = model.generation_config.prev_sot_token_id
    model.config.decoder_start_token_id = model.generation_config.prev_sot_token_id
    startoftranscript_id = processor.tokenizer.convert_tokens_to_ids('<|startoftranscript|>')
    force_list = list(range(100, 100+args.soft_prompt_len )) + [startoftranscript_id] + [i[1] for i in processor.get_decoder_prompt_ids()]
    model.config.forced_decoder_ids = [(i+1, x) for i, x in enumerate(force_list)]
else:
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids()
# forced_decoder_ids = processor.get_decoder_prompt_ids()
print(f"Forced decoding tokens: {model.config.forced_decoder_ids}")

metric = evaluate.load(f'metrics/{args.metric}.py')

total_words = 0
incorrect_words = 0
num_spks = args.num_spks
num_perms = math.perm(args.num_spks, args.num_spks)
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = model.generate(
                                        input_features=batch["input_features"].cuda(),
                                        # decoder_input_ids=batch["labels"][:, :4].cuda(),
                                        forced_decoder_ids=model.config.forced_decoder_ids,
                                        max_new_tokens=255,
                                        # num_beams=5,
                                        # repetition_penalty=1.2,
                                    ).cpu().numpy()
            if args.soft_prompt_len > 0:
                generated_tokens = generated_tokens[:, 1+args.soft_prompt_len:]
            labels = batch["labels"].view(-1, batch["labels"].shape[-1]).cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

            # convert prediction and ground truth tokens to text
            decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, basic_normalize=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True, basic_normalize=True)   # list of strings        

            if args.target_asr:
                decoded_preds = [pred for pred in decoded_preds if pred.strip()]
                num_spks = num_perms = 1
                
            if args.remove_pun:
                decoded_preds = remove_punctuation(decoded_preds)
                decoded_labels = remove_punctuation(decoded_labels)
            
            if args.language == 'zh':
                decoded_preds = to_simple(decoded_preds)
                decoded_labels = to_simple(decoded_labels)

                decoded_preds = [t.replace(' ', '') for t in decoded_preds]

                decoded_preds = [t.replace('', ' ').strip() for t in decoded_preds]
                decoded_labels = [t.replace('', ' ').strip() for t in decoded_labels]

                decoded_preds = [re.sub(r'([a-zA-Z0-9])\s+(?=[a-zA-Z0-9])', r'\1', t) for t in decoded_preds]


            decoded_preds = np.array(decoded_preds)
            decoded_labels = np.array(decoded_labels)

            # compare WER for different permutations, choose the lowest
            decoded_preds_perm = decoded_preds.reshape(-1, num_spks).repeat(num_perms, axis=0).reshape(-1, num_perms, num_spks)     # [batch, 2, 2]
            decoded_labels_perm = decoded_labels.reshape(-1, num_spks)
            decoded_labels_perm = np.split(decoded_labels_perm, decoded_labels_perm.shape[1], axis=1)
            decoded_labels_perm = list(itertools.permutations(decoded_labels_perm))
            decoded_labels_perm = np.stack([np.stack(t, axis=1) for t in decoded_labels_perm], axis=1).reshape(-1, num_perms, num_spks)

            wers_perm = []
            totals_perm = []
            incorrects_perm = []
            for b in zip(decoded_preds_perm, decoded_labels_perm):
                wers = []
                totals = []
                incorrects = []
                for p in zip(b[0], b[1]):
                    if len(wers) > 0 and any(w < 0.15 for w in wers):
                        # If there is a WER less than 0.15, it is considered correct and will not be calculated further.
                        wers.append(1)
                        totals.append(1)
                        incorrects.append(1)
                        continue
                    result = compute_wer(predictions=p[0], references=p[1])
                    wers.append(result["wer"])
                    totals.append(result["n_words"])
                    incorrects.append(result["n_incorrections"])
                wers_perm.append(wers)
                totals_perm.append(totals)
                incorrects_perm.append(incorrects)
                
            wers_perm = np.array(wers_perm)
            wers_order = np.argmin(wers_perm, axis=1)
            decoded_labels = decoded_labels_perm[np.arange(len(wers_perm)), wers_order].reshape(-1)
            total_words += np.array(totals_perm)[np.arange(len(wers_perm)), wers_order].sum()
            incorrect_words += np.array(incorrects_perm)[np.arange(len(wers_perm)), wers_order].sum()

            count = 0
            for pair in zip (decoded_labels, decoded_preds):
                print('\nLABEL: ', pair[0])
                print(' PRED: ', pair[1])
                count += 1
                if count % args.num_spks == 0:
                    print('\n'+'-' * 20)
            print('='*100, total_words, incorrect_words)

    # del generated_tokens, labels, batch
    # gc.collect()
    
print_arguments(args)
m = 100 * (incorrect_words / total_words)
print(f"Results: {args.metric}={round(m, 5)}")
