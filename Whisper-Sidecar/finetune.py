import argparse
import functools
import os
import json
import torch

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor
from model.modeling_whisper_sidecar import WhisperSidecarForConditionalGeneration

from utils.callback import SaveCheckpointCallback
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.model_utils import load_from_checkpoint
from utils.dataset import CustomDataset
from utils.utils import print_arguments, make_inputs_require_grad, add_arguments, strtobool
from metrics.compute_metrics import WhisperOverlapWERCalculator


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# paths
add_arg("train_data", type=str, default="dataset/libri2mix_train.jsonl", help="path for training data")
add_arg("dev_data", type=str, default="dataset/libri2mix_dev.jsonl", help="path for dev data")
add_arg("base_model", type=str, default="openai/whisper-small", help="Whisper model")
add_arg("model_name", type=str, default=None, help="model name")
add_arg("output_dir", type=str, default="output/", help="save path")
add_arg("logging_steps", type=int, default=200, help="log every x steps")
add_arg("resume_from_checkpoint", type=str, default=None, help="checkpoint path for recovery training")

# training settings
add_arg("fp16",  type=bool, default=True, help="if use fp16")
add_arg("max_steps", type=int, default=400000, help="max training steps")
add_arg("warmup_steps", type=int, default=500, help="warming up steps for lr")
add_arg("eval_steps", type=int, default=2000, help="eval every this many steps")
add_arg("save_steps", type=int, default=2000, help="save checkpoint every x steps")
add_arg("learning_rate", type=float, default=2e-4, help="learning rate")
add_arg("lr_scheduler_type", type=str, default="linear", help="learning rate scheduler")
add_arg("eval_delay", type=int, default=15, help="eval after x epoches")
add_arg("gradient_accumulation_steps", type=int, default=1, help="gradient accumulation")
add_arg("per_device_train_batch_size", type=int, default=16, help="training batch size")
add_arg("per_device_eval_batch_size", type=int, default=4, help="eval batch size")
add_arg("num_workers", type=int, default=1, help="Number of threads for reading data")
add_arg("language", type=str, default="en", help="en or zh. if it is None, it is trained to be multilingual")
add_arg("timestamps", type=strtobool, default='false', help="whether to use timestamp data during training")
add_arg("min_audio_len", type=float, default=0.5,  help="min audio length, in seconds")
add_arg("max_audio_len", type=float, default=30, help="max audio length, in seconds")
add_arg("local_files_only", type=bool, default=False, help="whether to only load the model locally")
add_arg("task", type=str, default="transcribe", choices=['transcribe', 'translate'], help="whisper task")
add_arg("augment_config_path", type=str, default=None, help="data augmentation configuration file path")

# model custom config
add_arg("num_spks", type=int, default=2, help="max number of speakers in the training set")
add_arg("sidecar_loc", type=int, default=1, help="location of sidecar")
add_arg("soft_prompt_len", type=int, default=4, help="soft prompt in decoder input")
add_arg("target_asr", type=strtobool, default='false', help="whether to train the target asr task")

args = parser.parse_args()
world_size = int(os.environ.get("WORLD_SIZE", 1))

if 'whisper-' in args.base_model: 
    model_name = f"{os.path.basename(args.base_model).split('whisper-')[1]}_{args.model_name}"
else:
    model_name = f"{args.model_name}_{args.base_model.split('/')[-2]}"
output_dir = os.path.join(args.output_dir, model_name)

# Get WhisperProcessor, which includes cnn feature extractor and tokenizer
processor = WhisperProcessor.from_pretrained(args.base_model,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=True,
                                             local_files_only=args.local_files_only)
processor.save_pretrained(os.path.join(output_dir, "tokenizer"))

arg_path = os.path.join(output_dir, "finetune_config.json")
with open(arg_path, 'w') as f:
    json.dump(vars(args), f, ensure_ascii=False, indent=2)


train_dataset = CustomDataset(data_list_path=args.train_data,
                              processor=processor,
                              language=args.language,
                              timestamps=args.timestamps,
                              min_duration=args.min_audio_len,
                              max_duration=args.max_audio_len,
                              augment_config_path=args.augment_config_path,
                              num_spks=args.num_spks,
                              )
dev_dataset = CustomDataset(data_list_path=args.dev_data,
                             processor=processor,
                             language=args.language,
                             timestamps=args.timestamps,
                             min_duration=args.min_audio_len,
                             max_duration=args.max_audio_len,
                             num_spks=args.num_spks,
                             is_eval_dataset=True
                             )

device_map = "auto"
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

# load model
model_path = args.resume_from_checkpoint if args.resume_from_checkpoint else args.base_model
model = WhisperSidecarForConditionalGeneration.from_pretrained(model_path,
                                                        device_map=device_map,
                                                        local_files_only=args.local_files_only,
                                                        sidecar_loc=args.sidecar_loc,
                                                        num_spks=args.num_spks,
                                                        soft_prompt_len=args.soft_prompt_len,
                                                        target_asr=args.target_asr
                                                        )

# setting for decoder soft_prompt
if args.soft_prompt_len > 0:
    model.generation_config.decoder_start_token_id = model.generation_config.prev_sot_token_id
    model.config.decoder_start_token_id = model.generation_config.prev_sot_token_id
    startoftranscript_id = processor.tokenizer.convert_tokens_to_ids('<|startoftranscript|>')
    force_list = list(range(100, 100+args.soft_prompt_len )) + [startoftranscript_id] + [i[1] for i in processor.get_decoder_prompt_ids()]
    model.config.forced_decoder_ids = [(i+1, x) for i, x in enumerate(force_list)]
else:
    startoftranscript_id = processor.tokenizer.convert_tokens_to_ids('<|startoftranscript|>')
    model.generation_config.decoder_start_token_id = startoftranscript_id
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids()

model.config.suppress_tokens = []
model.generation_config.max_new_tokens = 200


# Register forward, otherwise the multi-GPU training will fail.
model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)


# init model parameters
for name, params in model.named_parameters():
    params.requires_grad = False
    if 'sidecar' in name or 'sep_enc' in name or 'sep_dec' in name or 'soft_prompt_embeds' in name or 'proj_target' in name:
        params.requires_grad = True
        if args.resume_from_checkpoint or "openai/whisper" not in args.base_model:
            continue
        if 'bias' in name:
            torch.nn.init.constant_(params, 0.0)
        if 'weight' in name or 'soft_prompt_embeds' in name:
            torch.nn.init.normal_(params, mean=0.0, std=0.02)


# training args
training_args = \
    Seq2SeqTrainingArguments(output_dir=output_dir,
                             per_device_train_batch_size=args.per_device_train_batch_size,
                             per_device_eval_batch_size=args.per_device_eval_batch_size,
                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                             learning_rate=args.learning_rate,
                             lr_scheduler_type=args.lr_scheduler_type,
                             warmup_steps=args.warmup_steps,
                             max_steps=args.max_steps,
                             fp16=args.fp16,
                             save_strategy="epoch",
                            #  save_steps=args.save_steps,
                             save_total_limit=30,
                             evaluation_strategy="epoch",
                            #  eval_steps=args.eval_steps,
                             eval_delay=args.eval_delay,
                             load_best_model_at_end=True,
                             report_to=["tensorboard"],
                             torch_compile=False,
                             optim='adamw_torch',
                             ddp_find_unused_parameters=False if ddp else None,
                             dataloader_num_workers=args.num_workers,
                             logging_steps=args.logging_steps,
                             remove_unused_columns=False,
                             label_names=["labels"],
                             predict_with_generate=True,
                             metric_for_best_model="wer",
                             greater_is_better=False,
                            )

if training_args.local_rank == 0 or training_args.local_rank == -1:
    print(model)
    print_arguments(args)
    print(f"world_size: {world_size}")
    print(f"Model Name：{model_name}")
    print(f"Training data：{len(train_dataset)}，Eval data：{len(dev_dataset)}")

wer_cal = WhisperOverlapWERCalculator(processor=processor, num_spks=args.num_spks, soft_prompt_len=args.soft_prompt_len)


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, num_spks=args.num_spks, soft_prompt_len=args.soft_prompt_len, target_asr=args.target_asr)
trainer = Seq2SeqTrainer(args=training_args,
                         model=model,
                         train_dataset=train_dataset,
                         eval_dataset=dev_dataset,
                         data_collator=data_collator,
                         tokenizer=processor.feature_extractor,
                         callbacks=[SaveCheckpointCallback],
                        #  preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                         compute_metrics=wer_cal
)
model.config.use_cache = False
trainer._load_from_checkpoint = load_from_checkpoint

# start training
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# save the last checkpoint
trainer.save_state()

if training_args.local_rank == 0 or training_args.local_rank == -1:
    model.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
