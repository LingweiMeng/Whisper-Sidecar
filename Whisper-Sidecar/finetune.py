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
from peft import LoraConfig, get_peft_model, PeftConfig, AdaLoraConfig, PeftModel, prepare_model_for_kbit_training


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
add_arg("num_workers", type=int, default=8, help="Number of threads for reading data")
add_arg("language", type=str, default="en", help="en or zh. if it is None, it is trained to be multilingual")
add_arg("timestamps", type=strtobool, default='false', help="whether to use timestamp data during training")
add_arg("min_audio_len", type=float, default=0.5,  help="min audio length, in seconds")
add_arg("max_audio_len", type=float, default=30, help="max audio length, in seconds")
add_arg("local_files_only", type=bool, default=False, help="whether to only load the model locally")
add_arg("task", type=str, default="transcribe", choices=['transcribe', 'translate'], help="whisper task")
add_arg("augment_config_path", type=str, default=None, help="data augmentation configuration file path")

# model custom config
add_arg("use_lora", type=strtobool, default='false', help="whether to train lora")
add_arg("use_adalora", type=strtobool,  default='false', help="whether use adalora instead of lora")
add_arg("num_spks", type=int, default=2, help="max number of speakers in the training set")
add_arg("sidecar_loc", type=int, default=1, help="location of sidecar")
add_arg("soft_prompt_len", type=int, default=4, help="soft prompt in decoder input")
add_arg("target_asr", type=strtobool, default='false', help="whether to train the target asr task")

args = parser.parse_args()
world_size = int(os.environ.get("WORLD_SIZE", 1))

# if 'whisper-' in args.base_model: 
#     model_name = f"{os.path.basename(args.base_model).split('whisper-')[1]}_{args.model_name}"
# else:
#     model_name = f"{args.model_name}_{args.base_model.split('/')[-2]}"

model_name = args.model_name
output_dir = os.path.join(args.output_dir, model_name)

try:
    peft_config = PeftConfig.from_pretrained(args.resume_from_checkpoint if args.resume_from_checkpoint is not None else args.base_model)
    args.use_adalora = True if peft_config.peft_type == "ADALORA" else args.use_adalora
    base_base_model = peft_config.base_model_name_or_path
    print("The base model is with LoRA.")
except:
    base_base_model = args.base_model
    peft_config = None


# Get WhisperProcessor, which includes cnn feature extractor and tokenizer
processor = WhisperProcessor.from_pretrained(base_base_model,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=True,
                                             local_files_only=args.local_files_only)
processor.save_pretrained(os.path.join(output_dir, "tokenizer"))

with open(os.path.join(output_dir, "finetune_config.json"), 'w') as f:
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
                                                        target_asr=args.target_asr,
                                                        attn_implementation="sdpa",

modules_to_save = ['sidecar', 'sep_enc', 'sep_dec', 'soft_prompt_embeds', 'proj_target']
print(f"Modules to train and save: {modules_to_save}")

if peft_config is not None:
    print(f'loading LoRA modules... if_train_lora = {args.use_lora}')
    peft_config.modules_to_save = modules_to_save
    model = PeftModel.from_pretrained(model, model_path, is_trainable=True if args.use_lora else False, config=peft_config)
elif args.use_lora:
    print(f'adding LoRA modules...')
    target_modules = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
    print(target_modules)
    if args.use_adalora:
        peft_config = AdaLoraConfig(init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
                                    lora_alpha=32, lora_dropout=0.1, orth_reg_weight=0.5, target_modules=target_modules, modules_to_save=modules_to_save)
    else:
        peft_config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules,
                                    lora_dropout=0.05, bias="none", modules_to_save=modules_to_save)
else:
    # init model parameters
    for name, params in model.named_parameters():
        params.requires_grad = False
        if any(e in name for e in modules_to_save) or ("lora" in name and args.use_lora):
            params.requires_grad = True

if not (args.resume_from_checkpoint or "sidecar" in args.base_model):
    model.param_init(modules_to_save)


# setting for decoder soft_prompt
if args.soft_prompt_len > 0:
    if not hasattr(model.generation_config, "prev_sot_token_id"):
        model.generation_config.prev_sot_token_id = 50361
    model.generation_config.decoder_start_token_id = model.generation_config.prev_sot_token_id
    model.config.decoder_start_token_id = model.generation_config.prev_sot_token_id
    startoftranscript_id = processor.tokenizer.convert_tokens_to_ids('<|startoftranscript|>')
    force_list = list(range(100, 100+args.soft_prompt_len )) + [startoftranscript_id] + [i[1] for i in processor.get_decoder_prompt_ids()]
    model.config.forced_decoder_ids = [(i+1, x) for i, x in enumerate(force_list)]
else:
    startoftranscript_id = processor.tokenizer.convert_tokens_to_ids('<|startoftranscript|>')
    model.generation_config.decoder_start_token_id = startoftranscript_id
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids()
print(f"Forced decoding tokens: {model.config.forced_decoder_ids}")

model.config.suppress_tokens = []
model.generation_config.max_new_tokens = 200


# Register forward, otherwise the multi-GPU training will fail.
model.get_encoder().conv1.register_forward_hook(make_inputs_require_grad)


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
    print(f"Model Name: {model_name}")
    print(f"Training data: {len(train_dataset)}, Eval data: {len(dev_dataset)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}, ", 
          f"All parameters: {sum(p.numel() for p in model.parameters())}")
    print("------------------------------------------------")


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

model.config.use_cache = True
model.generation_config.use_cache = True
trainer._load_from_checkpoint = load_from_checkpoint

# start training
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# save the last checkpoint
trainer.save_state()

if training_args.local_rank == 0 or training_args.local_rank == -1:
    model.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
