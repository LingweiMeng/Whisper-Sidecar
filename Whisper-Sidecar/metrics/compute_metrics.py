#!/usr/bin/env python

import numpy as np

from dataclasses import dataclass
from typing import Any
from evaluate import load
import math
import itertools
from jiwer import compute_measures
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding, remove_punctuation, to_simple
import re

def compute_wer(predictions, references):
    incorrect = 0
    total = 0
    totalS, totalD, totalI = 0, 0, 0
    for prediction, reference in zip(predictions, references):
        if len(reference) == 0:
            # TODO: check
            if len(prediction) == 0:
                continue
            else:
                incorrect += len(prediction)
                total += len(prediction)
        else:
            measures = compute_measures(reference, prediction)
            H, S, D, I = measures["hits"], measures["substitutions"], measures["deletions"], measures["insertions"]
            totalS += S
            totalD += D
            totalI += I
            incorrect += S + D + I
            total += S + D + H
    
    return {
        "wer": incorrect / total if total > 0 else 0,
        "n_words": total,
        "n_incorrections": incorrect,
        "n_substitutions": totalS,
        "n_deletions": totalD,
        "n_insertions": totalI,
    }


# @dataclass
# class WERCalculator:
#     processor: Any
#     wer_metric = load("wer")

#     def __call__(self, pred) -> dict:
        
#         pred_logits = pred.predictions
#         pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

#         pred_ids = np.argmax(pred_logits, axis=-1)
#         pred_str = self.processor.batch_decode(pred_ids)
    
#         # this will always work (even if the processor has a decoder LM)
#         label_str = self.processor.tokenizer.batch_decode(pred.label_ids, group_tokens=False)

#         wer = self.wer_metric.compute(predictions=pred_str, references=label_str)

#         return {"wer": wer}

# @dataclass
# class WhisperWERCalculator:
#     processor: Any
#     wer_metric = load("wer")

#     def __call__(self, pred) -> dict:
#         # pred.predictions: (logits, h)
#         pred_logits = pred.predictions[0]
#         pred_ids = np.argmax(pred_logits, axis=-1)
#         label_ids = pred.label_ids

#         # replace -100 with pad token
#         label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

#         # remove tokens after eos
#         eos = self.processor.tokenizer.eos_token_id
#         pred_ids = [
#             pred[:np.where(pred == eos)[0][0]] if eos in pred else pred
#             for pred in pred_ids
#         ]

#         pred_str = self.processor.batch_decode(
#             pred_ids,
#             skip_special_tokens=True,
#         )

#         label_str = self.processor.batch_decode(
#             label_ids,
#             skip_special_tokens=True,
#         )

#         # remove samples with empty ref
#         for i, (_, label) in enumerate(zip(pred_str, label_str)):
#             if not label:
#                 del pred_str[i]
#                 del label_str[i]

#         # save references and predictions to a txt
#         with open("refs_and_preds.txt", "w") as f:
#             for ref, pred in zip(label_str, pred_str):
#                 f.write(f"REF: {ref}\n")
#                 f.write(f"PRED: {pred}\n")

#         wer = 100 * self.wer_metric.compute(
#             predictions=pred_str, 
#             references=label_str,
#         )

#         return {"wer": wer}
    
@dataclass
class WhisperOverlapWERCalculator:
    processor: Any
    num_spks: int
    soft_prompt_len: int

    def __call__(self, pred) -> dict:
        num_perms = math.perm(self.num_spks, self.num_spks)
        pred_ids, label_ids = pred.predictions, pred.label_ids
        if self.soft_prompt_len > 0:
            pred_ids = pred_ids[:, 1+self.soft_prompt_len:]
            label_ids = label_ids[:, self.soft_prompt_len:]     # no decoder_start_token_id
        
        # Convert the predicted and ground truth tokens into text.
        if self.processor.tokenizer.language == 'zh':
            decoded_preds = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, basic_normalize=True)
            decoded_labels = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, basic_normalize=True) 

            decoded_preds = to_simple(decoded_preds)
            decoded_labels = to_simple(decoded_labels)

            decoded_preds = [t.replace(' ', '') for t in decoded_preds]

            decoded_preds = [t.replace('', ' ').strip() for t in decoded_preds]
            decoded_labels = [t.replace('', ' ').strip() for t in decoded_labels]

            decoded_preds = [re.sub(r'([a-zA-Z0-9])\s+(?=[a-zA-Z0-9])', r'\1', t) for t in decoded_preds]
        else:
            decoded_preds = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, normalize=True)
            decoded_labels = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, normalize=True)      

        decoded_preds = remove_punctuation(decoded_preds)
        decoded_labels = remove_punctuation(decoded_labels)
        
        # # remove samples with empty ref
        # for i, (_, label) in enumerate(zip(pred_str, label_str)):
        #     if not label:
        #         del pred_str[i]
        #         del label_str[i]

        # # save references and predictions to a txt
        # with open("refs_and_preds.txt", "w") as f:
        #     for ref, pred in zip(label_str, pred_str):
        #         f.write(f"REF: {ref}\n")
        #         f.write(f"PRED: {pred}\n")

        decoded_preds = np.array(decoded_preds)
        decoded_labels = np.array(decoded_labels)
        
        # compare and select the permutation with lowest WER
        decoded_preds_perm = decoded_preds.reshape(-1, self.num_spks).repeat(num_perms, axis=0).reshape(-1, num_perms, self.num_spks)     # [batch, num_perms, num_spks]
        decoded_labels_perm = decoded_labels.reshape(-1, self.num_spks)
        decoded_labels_perm = np.split(decoded_labels_perm, decoded_labels_perm.shape[1], axis=1)
        decoded_labels_perm = list(itertools.permutations(decoded_labels_perm))
        decoded_labels_perm = np.stack([np.stack(t, axis=1) for t in decoded_labels_perm], axis=1).reshape(-1, num_perms, self.num_spks)
        
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
        total_words = np.array(totals_perm)[np.arange(len(wers_perm)), wers_order].sum()
        incorrect_words = np.array(incorrects_perm)[np.arange(len(wers_perm)), wers_order].sum()


        wer = 100 * incorrect_words / total_words
        # for pair in zip (decoded_labels, decoded_preds):
        #     print('\nLABEL: ', pair[0])
        #     print(' PRED: ', pair[1])
        # print('='*100)
        # print(f"WER: {wer}, incorrect_words: {incorrect_words}, total_words: {total_words}")
        # print('='*100)

        return {"wer": wer}
    
    
def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pass
