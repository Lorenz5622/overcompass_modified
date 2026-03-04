# flake8: noqa
import csv
import math
import os
import re
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.device import is_npu_available

from opencompass.utils.logging import get_logger

from .huggingface_above_v4_33 import (
    HuggingFaceBaseModel,
    _convert_base_messages,
    _set_model_kwargs_torch_dtype,
)

QWEN_MOE_MODELING_PATH = '/home/cyx/qwen_moe'

# Default path for the per-token routing statistics TSV file.
_DEFAULT_TSV_PATH = 'moe_routing_token_stats_piqa.tsv'


class HuggingFaceQwenMoeCustom(HuggingFaceBaseModel):
    """HuggingFaceBaseModel variant that loads Qwen2MoE using a local
    modeling_moe.py instead of the built-in transformers implementation.

    During PPL evaluation, per-token routing summary metrics (top4_mean,
    delta_4_5_mean, entropy_mean for all layers and last-2 layers) together
    with the token NLL are written to a TSV file.  The path can be overridden
    by setting ``self._routing_tsv_path`` before calling ``get_ppl``.
    """

    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        if QWEN_MOE_MODELING_PATH not in sys.path:
            sys.path.insert(0, QWEN_MOE_MODELING_PATH)

        from Qwen_MoE.modeling.configuration_moe import Qwen2MoeConfig
        from Qwen_MoE.modeling.modeling_moe import Qwen2MoeForCausalLM

        DEFAULT_MODEL_KWARGS = dict(device_map='auto')
        model_kwargs = DEFAULT_MODEL_KWARGS
        model_kwargs.update(kwargs)
        model_kwargs = _set_model_kwargs_torch_dtype(model_kwargs)
        print("load local QWEN modeling_moe")
        logger = get_logger()
        logger.debug(f'using model_kwargs: {model_kwargs}')

        if is_npu_available():
            model_kwargs['device_map'] = 'npu'

        config = Qwen2MoeConfig.from_pretrained(path)
        self.model = Qwen2MoeForCausalLM.from_pretrained(path, config=config, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            peft_kwargs['is_trainable'] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False

    def get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None):
        """Compute PPL scores while recording per-token MoE routing metrics.

        For every valid (non-padding) token position in each input, the
        following columns are written to the TSV file (one row per token):

            input_text   : first 60 chars of the input string (tab/newline stripped)
            token_idx    : 0-based position index (predicting token at idx+1)
            token_nll    : per-token cross-entropy loss (NLL contribution)
            top4_all     : mean top-4 probability mass across all MoE layers
            delta_all    : mean (prob_rank4 - prob_rank5) across all layers
            entropy_all  : mean routing entropy across all layers
            top4_last2   : mean top-4 probability mass across last 2 MoE layers
            delta_last2  : mean (prob_rank4 - prob_rank5) across last 2 layers
            entropy_last2: mean routing entropy across last 2 layers

        Returns the same averaged ce_loss array as the parent implementation.
        """
        if QWEN_MOE_MODELING_PATH not in sys.path:
            sys.path.insert(0, QWEN_MOE_MODELING_PATH)
        from Qwen_MoE.modeling.modeling_moe import Qwen2MoeSparseMoeBlock

        assert self.tokenizer.pad_token, 'pad_token must be set'
        pad_token_id = self.tokenizer.pad_token_id
        messages = _convert_base_messages(inputs)

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len,
        )

        if self.drop_middle:
            assert len(inputs) == 1
            input_ids = self.tokenizer(inputs, padding=False, truncation=False)['input_ids']
            input_ids = torch.tensor(input_ids)
            if input_ids.shape[-1] > self.max_seq_len:
                input_ids = torch.cat(
                    [input_ids[:, :self.max_seq_len // 2],
                     input_ids[:, -self.max_seq_len // 2:]], dim=-1)
            tokens = {'input_ids': input_ids}
        else:
            tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)

        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

        # ------------------------------------------------------------------
        # Register forward hooks on every Qwen2MoeSparseMoeBlock.
        # Each hook immediately computes per-token summary metrics from the
        # router softmax probabilities and stores only those scalars
        # (3 floats per token), discarding the raw 60-dim probability vector.
        #
        # routing_stats[lkey]  : list of (top4, delta45, entropy) tuples,
        #                        one per token, ordered as the flattened
        #                        (batch_size * seq_len) dimension.
        # layer_indices[lkey]  : integer layer index extracted from the name.
        # ------------------------------------------------------------------
        routing_stats = {}   # lkey -> list[(top4, delta45, entropy)]
        layer_indices = {}   # lkey -> int
        hooks = []

        for name, module in self.model.named_modules():
            if isinstance(module, Qwen2MoeSparseMoeBlock):
                m = re.search(r'layers\.([0-9]+)\.mlp', name)
                layer_key = f'layer{m.group(1)}' if m else name
                layer_idx = int(m.group(1)) if m else -1
                layer_indices[layer_key] = layer_idx

                def make_hook(lkey):
                    def hook_fn(mod, inp, out):
                        # out = (final_hidden_states, router_logits)
                        # router_logits: (batch*seq_len, num_experts)
                        router_logits = out[1]
                        rw = F.softmax(router_logits.float(), dim=1)
                        rw_np = rw.cpu().detach().numpy()
                        stats = []
                        for token_probs in rw_np:
                            sp = sorted(token_probs, reverse=True)
                            s_top4 = float(sp[0] + sp[1] + sp[2] + sp[3]) \
                                if len(sp) >= 4 else float(sum(sp))
                            delta = float(sp[3] - sp[4]) if len(sp) >= 5 else 0.0
                            ent = 0.0
                            for p in token_probs:
                                p = float(p)
                                if p > 1e-12:
                                    ent -= p * math.log(p)
                            stats.append((s_top4, delta, ent))
                        routing_stats[lkey] = stats
                    return hook_fn

                hooks.append(module.register_forward_hook(make_hook(layer_key)))

        # ------------------------------------------------------------------
        # Forward pass – hooks fire here.
        # ------------------------------------------------------------------
        try:
            with torch.no_grad():
                outputs = self.model(**tokens)[0]  # (batch, seq_len, vocab_size)
        finally:
            for h in hooks:
                h.remove()

        # ------------------------------------------------------------------
        # Compute per-token cross-entropy losses (same as parent get_ppl).
        # ------------------------------------------------------------------
        batch_size, seq_len, vocab_size = outputs.shape
        shift_logits = outputs[:, :-1, :].contiguous().float()
        shift_labels = tokens['input_ids'][:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
            reduction='none',
        ).view(batch_size, seq_len - 1)   # (batch, seq_len-1)
        lens = (tokens['input_ids'] != pad_token_id).sum(-1).cpu().numpy()

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # (batch, seq_len-1)
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask
            lens -= np.array(mask_length)

        ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens

        # ------------------------------------------------------------------
        # Compute per-token routing summaries and write TSV.
        # ------------------------------------------------------------------
        self._write_token_stats_tsv(
            inputs=inputs,
            token_ids_np=tokens['input_ids'].cpu().numpy(),
            loss_np=loss.float().cpu().detach().numpy(),
            routing_stats=routing_stats,
            layer_indices=layer_indices,
            batch_size=batch_size,
            seq_len=seq_len,
            pad_token_id=pad_token_id,
        )

        return ce_loss

    # ------------------------------------------------------------------
    # Helper: write per-token routing statistics to a TSV file.
    # ------------------------------------------------------------------
    def _write_token_stats_tsv(
        self,
        inputs,
        token_ids_np,
        loss_np,
        routing_stats,
        layer_indices,
        batch_size,
        seq_len,
        pad_token_id,
    ):
        """Append per-token routing stats rows to the TSV file."""
        # Determine layer ordering and which ones are the "last 2".
        sorted_layers = sorted(layer_indices.items(), key=lambda x: x[1])  # [(lkey, idx)]
        all_lkeys = [lk for lk, _ in sorted_layers]
        last_2_lkeys = (
            {lk for lk, _ in sorted_layers[-2:]}
            if len(sorted_layers) >= 2
            else set(all_lkeys)
        )

        tsv_path = getattr(self, '_routing_tsv_path', _DEFAULT_TSV_PATH)
        write_header = not os.path.exists(tsv_path)

        with open(tsv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            if write_header:
                # Fixed summary columns + one triplet per MoE layer.
                per_layer_header = []
                for lkey in all_lkeys:
                    per_layer_header += [
                        f'top4_{lkey}', f'delta_{lkey}', f'entropy_{lkey}',
                    ]
                writer.writerow([
                    'input_text', 'token_idx', 'token_nll',
                    'top4_all', 'delta_all', 'entropy_all',
                    'top4_last2', 'delta_last2', 'entropy_last2',
                ] + per_layer_header)

            for i in range(batch_size):
                num_valid = int((token_ids_np[i] != pad_token_id).sum())
                # Truncate input text for readability in the TSV.
                label = inputs[i][:60].replace('\t', ' ').replace('\n', ' ')

                for t in range(num_valid - 1):
                    # loss[i, t] is the NLL for predicting token at position t+1.
                    nll = float(loss_np[i, t])
                    # Exclude masked-out / invalid positions (commonly nll==0).
                    # This keeps downstream routing-vs-NLL analysis clean.
                    if nll <= 0.0:
                        continue
                    # Skip padding prediction positions.
                    if token_ids_np[i, t + 1] == pad_token_id:
                        continue

                    # Index into the flattened (batch*seq_len) routing stats.
                    stat_idx = i * seq_len + t

                    top4_all_v, delta_all_v, ent_all_v = [], [], []
                    top4_l2_v, delta_l2_v, ent_l2_v = [], [], []
                    # Per-layer values: [top4_l0, delta_l0, ent_l0, top4_l1, ...]
                    per_layer_vals = []

                    for lkey in all_lkeys:
                        if lkey not in routing_stats:
                            per_layer_vals += ['', '', '']
                            continue
                        stats = routing_stats[lkey]
                        if stat_idx >= len(stats):
                            per_layer_vals += ['', '', '']
                            continue
                        s_top4, delta, ent = stats[stat_idx]
                        top4_all_v.append(s_top4)
                        delta_all_v.append(delta)
                        ent_all_v.append(ent)
                        if lkey in last_2_lkeys:
                            top4_l2_v.append(s_top4)
                            delta_l2_v.append(delta)
                            ent_l2_v.append(ent)
                        per_layer_vals += [
                            f'{s_top4:.6f}', f'{delta:.6f}', f'{ent:.6f}',
                        ]

                    if not top4_all_v:
                        continue

                    def _mean(lst):
                        return sum(lst) / len(lst)

                    writer.writerow([
                        label,
                        t,
                        f'{nll:.6f}',
                        f'{_mean(top4_all_v):.6f}',
                        f'{_mean(delta_all_v):.6f}',
                        f'{_mean(ent_all_v):.6f}',
                        f'{_mean(top4_l2_v):.6f}' if top4_l2_v else '',
                        f'{_mean(delta_l2_v):.6f}' if delta_l2_v else '',
                        f'{_mean(ent_l2_v):.6f}' if ent_l2_v else '',
                    ] + per_layer_vals)
