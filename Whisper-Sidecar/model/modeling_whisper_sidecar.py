import math
import itertools
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from .sidecar import Sidecar
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoderLayer, 
    WhisperEncoder, 
    WhisperDecoder,
    shift_tokens_right
)
from transformers import (
    WhisperConfig,
    WhisperModel, 
    WhisperForConditionalGeneration
)
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor
from transformers.utils import logging
from transformers.models.whisper.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE

logger = logging.get_logger(__name__)

class WhisperSidecarEncoder(WhisperEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: WhisperConfig, sidecar_loc: int = 1, num_spks: int = 2, target_asr: bool = False, for_target_asr_eval: bool = False):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.num_spks = num_spks
        self.num_perms = math.perm(self.num_spks, self.num_spks)
        if sidecar_loc and sidecar_loc >= -1:
            self.sidecar_loc = sidecar_loc
            self.sep_enc = nn.Conv1d(embed_dim, embed_dim, 3, stride=1, padding='same')
            self.sidecar = Sidecar(N=embed_dim, B=128, H=embed_dim, num_spks=self.num_spks)
            self.sep_dec = nn.Conv1d(embed_dim, embed_dim, 3, stride=1, padding='same')
            self.target_asr = target_asr
        else:
            assert self.num_spks == 1, "The num_spks should be 1 when sidecar is disabled."
            self.sidecar_loc = None
            self.target_asr = None

        if self.target_asr:
            self.proj_target = nn.Linear(embed_dim, 1)
            self.proj_target_2  = nn.Linear(150, 1)
            self.for_target_asr_eval = for_target_asr_eval

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        target_speaker=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        # In order to maintain a consistent batch_size during eval, 
        # num_spks was duplicated in the collator. 
        # Therefore, it needs to be removed here.
        input_features = input_features[::self.num_spks]
        if attention_mask is not None:
            attention_mask = attention_mask[::self.num_spks]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        if self.sidecar_loc == -1:
            hidden_states = hidden_states.transpose(1,2)
            hidden_states = self.sep_enc(hidden_states)
            hidden_states, m_sidecar = self.sidecar(hidden_states)
            hidden_states = self.sep_dec(hidden_states)
            hidden_states = hidden_states.transpose(1,2)            # (B*self.num_spks, T, C)

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if idx == self.sidecar_loc:
                hidden_states = hidden_states.transpose(1,2)
                hidden_states = self.sep_enc(hidden_states)
                hidden_states, m_sidecar = self.sidecar(hidden_states)
                hidden_states = self.sep_dec(hidden_states)
                hidden_states = hidden_states.transpose(1,2)            # (B*self.num_spks, T, C)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # target talker identifier
        if self.target_asr and (target_speaker is not None or self.for_target_asr_eval):
            target_spk_logits = self.proj_target(hidden_states[:, :150, :]).reshape(-1,self.num_spks, 150, 1).squeeze(-1)    # pooling, (B, self.num_spks)
            target_spk_logits = torch.relu(target_spk_logits)
            target_spk_logits = self.proj_target_2(target_spk_logits).squeeze(-1)    # pooling, (B, self.num_spks)
            hidden_states = hidden_states[:, 155:, :]

        if self.training:
            _, t, c = hidden_states.shape
            hidden_states_unbind = hidden_states.view(-1, self.num_spks, t, c).unbind(1)
            hidden_states_perm = list(itertools.permutations(hidden_states_unbind))
            hidden_states = torch.stack([torch.stack(t, dim=1) for t in hidden_states_perm], dim=1).view(-1, t, c)


        elif self.target_asr and self.for_target_asr_eval:
            # Only select the hidden_states of target spk
            target_spk_idx = torch.argmax(target_spk_logits, dim=1)

            # mask those not target spk
            mask = torch.arange(self.num_spks, device=target_spk_idx.device) != target_spk_idx.unsqueeze(1)     # (B, num_spks)
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            hidden_states = hidden_states.view(-1, self.num_spks, hidden_states.shape[-2], hidden_states.shape[-1])
            hidden_states = hidden_states.masked_fill(mask, -100).view(-1, hidden_states.shape[-2], hidden_states.shape[-1])    # Block other spks


        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        else:
            encoder_outputs = BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            )

            if self.target_asr and target_speaker is not None and self.training:
                encoder_outputs['target_spk_logits'] = target_spk_logits
            return encoder_outputs

class WhisperSidecarModel(WhisperModel):
    def __init__(self, config: WhisperConfig, sidecar_loc: int = 1, num_spks: int = 2, target_asr: bool = False, for_target_asr_eval: bool = False):
        super().__init__(config)
        self.num_spks = num_spks
        self.num_perms = math.perm(self.num_spks, self.num_spks)
        self.target_asr = target_asr

        self.encoder = WhisperSidecarEncoder(config, sidecar_loc, num_spks, target_asr, for_target_asr_eval)
        self.decoder = WhisperDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        target_speaker: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> from transformers import AutoFeatureExtractor, WhisperModel
         >>> from datasets import load_dataset

         >>> model = WhisperModel.from_pretrained("openai/whisper-base")
         >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
         >>> input_features = inputs.input_features
         >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
         >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
         >>> list(last_hidden_state.shape)
         [1, 2, 512]
         ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                target_speaker=target_speaker,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        outputs = Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

        if self.target_asr and "target_spk_logits" in encoder_outputs:
            outputs['target_spk_logits'] = encoder_outputs['target_spk_logits']
        return outputs


class WhisperSidecarForConditionalGeneration(WhisperForConditionalGeneration):
    def __init__(self, config: WhisperConfig, 
                 sidecar_loc: int = 1, 
                 num_spks: int = 2, 
                 soft_prompt_len: int = 0,
                 target_asr: bool = False,
                 for_target_asr_eval: bool = False):
        super().__init__(config)
        self.num_spks = num_spks
        self.num_perms = math.perm(self.num_spks, self.num_spks)
        self.soft_prompt_len = soft_prompt_len
        self.model = WhisperSidecarModel(config, sidecar_loc, self.num_spks, target_asr, for_target_asr_eval)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.target_asr = target_asr

        if self.soft_prompt_len > 0:
            self.soft_prompt_embeds = nn.Embedding(self.soft_prompt_len, config.d_model)
            self.register_soft_prompt_hook()

        # Initialize weights and apply final processing
        self.post_init()

    def param_init(self, modules_to_save):
        for name, params in self.named_parameters():
            if any(e in name for e in modules_to_save):
                if 'bias' in name:
                    torch.nn.init.constant_(params, 0.0)
                if 'weight' in name or 'soft_prompt_embeds' in name:
                    torch.nn.init.normal_(params, mean=0.0, std=0.02)

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        target_speaker: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            labels_input = labels.view(-1, self.num_spks, labels.shape[-1]).repeat(1, self.num_perms, 1).view(-1, labels.shape[-1])

            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels_input, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            target_speaker=target_speaker,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="none")
            targets = labels_input.to(lm_logits.device)

            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size).to(torch.float32), targets.reshape(-1))
            # PIT loss
            loss, loss_min_idx= loss.view(-1, self.num_perms, self.num_spks, targets.shape[-1]).mean(dim=-1).mean(2).min(1)
            loss = loss.mean()
            if self.target_asr:
                if hasattr(outputs, "target_spk_logits"):
                    # get the best permutation for target spk according to loss_min_idx
                    best_perm = [list(list(itertools.permutations(range(self.num_spks)))[i]) for i in loss_min_idx]
                    target_spk_logits = outputs['target_spk_logits'][torch.arange(len(best_perm)).unsqueeze(1), best_perm]
                    target_spk_loss = loss_fct(target_spk_logits.to(torch.float32), target_speaker).mean() * 0.01
                    acc = (torch.argmax(target_spk_logits, dim=1) == target_speaker).float().mean()
                    print(acc, target_spk_loss * 100)
                else:
                    target_spk_loss = self.model.encoder.proj_target(torch.zeros(1, 1, outputs[0].shape[-1], device=outputs[0].device)).sum() * 0
                    target_spk_loss += self.model.encoder.proj_target_2(torch.zeros(1, 1, 150, device=outputs[0].device)).sum() * 0
                loss += target_spk_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def register_soft_prompt_hook(self):

        def decoder_soft_prompt_hook(module, input_ids, inputs_embeds):
            if inputs_embeds.shape[1] == 1:
                # generating
                input_ids = input_ids[0].reshape(-1)
                if (100 <= input_ids[0] < 100+self.soft_prompt_len) and torch.all(input_ids.eq(input_ids[0])):
                    # relace the fake token with the soft prompt
                    inputs_embeds[:] = self.soft_prompt_embeds.weight[input_ids[0]-100]
            else:
                # training or teacher-forcing evaluation
                inputs_embeds[:, 1:self.soft_prompt_embeds.weight.shape[0]+1] = self.soft_prompt_embeds.weight
            return inputs_embeds

        self.model.decoder.embed_tokens.register_forward_hook(decoder_soft_prompt_hook)