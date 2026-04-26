import math
from typing import Dict, List, Optional

import numpy as np
import torch

from .huggingface import HuggingFaceDynamicMoE


class HuggingFaceYuanMoE(HuggingFaceDynamicMoE):
    """Yuan-specific MoE wrapper with isolated router statistics."""

    _YUAN_ROUTER_TYPES = (
        'ParallelAttention_router',
        'CrossAttentionRouter',
        'CorssAttentionRouter',
    )
    _FLOPS_MULTIPLY_ADD = 2.0

    def __init__(
        self,
        path: str,
        hf_cache_dir: Optional[str] = None,
        max_seq_len: int = 2048,
        tokenizer_path: Optional[str] = None,
        tokenizer_kwargs: dict = dict(),
        peft_path: Optional[str] = None,
        tokenizer_only: bool = False,
        model_kwargs: dict = dict(device_map="auto"),
        generation_kwargs: dict = dict(),
        meta_template: Optional[Dict] = None,
        extract_pred_after_decode: bool = False,
        batch_padding: bool = False,
        pad_token_id: Optional[int] = None,
        mode: str = "none",
        num_extra_tokens: int = 50,
        moe_package_name: str = "Qwen_MoE",
        moe_modeling_module: str = "modeling_moe_yuan",
        moe_config_module: str = "configuration_moe_yuan",
    ):
        model_kwargs.setdefault('enable_expert_stats', True)
        model_kwargs.setdefault('enable_kpredictor_entropy_stats', False)
        model_kwargs.setdefault('enable_token_routing_tsv', False)
        model_kwargs.setdefault('enable_routing_eval', False)
        self._enable_theoretical_flops_stats = bool(
            model_kwargs.pop('enable_theoretical_flops_stats', True))
        self._yuan_router_stats = {}
        self._init_yuan_flops_stats()
        super().__init__(
            path=path,
            hf_cache_dir=hf_cache_dir,
            max_seq_len=max_seq_len,
            tokenizer_path=tokenizer_path,
            tokenizer_kwargs=tokenizer_kwargs,
            peft_path=peft_path,
            tokenizer_only=tokenizer_only,
            model_kwargs=model_kwargs,
            generation_kwargs=generation_kwargs,
            meta_template=meta_template,
            extract_pred_after_decode=extract_pred_after_decode,
            batch_padding=batch_padding,
            pad_token_id=pad_token_id,
            mode=mode,
            num_extra_tokens=num_extra_tokens,
            moe_package_name=moe_package_name,
            moe_modeling_module=moe_modeling_module,
            moe_config_module=moe_config_module,
        )
        if not tokenizer_only:
            print('[INFO] Yuan router statistics enabled')
            if self._enable_theoretical_flops_stats:
                print('[INFO] Yuan theoretical FLOPs statistics enabled')

    def _init_yuan_flops_stats(self) -> None:
        self._yuan_flops_stats = {
            'total_flops': 0.0,
            'attention_flops': 0.0,
            'router_flops': 0.0,
            'expert_flops': 0.0,
            'dense_mlp_flops': 0.0,
            'lm_head_flops': 0.0,
            'total_input_tokens': 0,
            'total_scored_tokens': 0,
            'total_samples': 0,
            'per_layer': {},
        }

    def _ensure_yuan_flops_stats(self) -> None:
        if not getattr(self, '_yuan_flops_stats', None):
            self._init_yuan_flops_stats()

    @staticmethod
    def _linear_flops(num_rows: int, in_dim: int, out_dim: int) -> float:
        return (HuggingFaceYuanMoE._FLOPS_MULTIPLY_ADD
                * float(num_rows) * float(in_dim) * float(out_dim))

    def _attention_layer_flops(self, seq_len: int) -> Dict[str, float]:
        hidden = int(self.model.config.hidden_size)
        heads = int(self.model.config.num_attention_heads)
        head_dim = hidden // max(heads, 1)
        token_count = int(seq_len)
        qkv_o = 4.0 * self._linear_flops(token_count, hidden, hidden)
        attn_scores = (self._FLOPS_MULTIPLY_ADD * float(heads)
                       * float(seq_len) * float(seq_len) * float(head_dim))
        attn_apply = attn_scores
        total = qkv_o + attn_scores + attn_apply
        return {
            'attention_flops': total,
            'attention_linear_flops': qkv_o,
            'attention_score_flops': attn_scores,
            'attention_apply_flops': attn_apply,
        }

    def _router_layer_flops(self, token_count: int) -> Dict[str, float]:
        hidden = int(self.model.config.hidden_size)
        num_experts = int(self.model.config.num_experts)
        linear = 3.0 * self._linear_flops(token_count, hidden, num_experts)
        pairwise = (2.0 * self._FLOPS_MULTIPLY_ADD * float(token_count)
                    * float(num_experts) * float(num_experts))
        total = linear + pairwise
        return {
            'router_flops': total,
            'router_linear_flops': linear,
            'router_attention_flops': pairwise,
        }

    def _dense_mlp_layer_flops(self, token_count: int) -> float:
        hidden = int(self.model.config.hidden_size)
        intermediate = int(self.model.config.intermediate_size)
        return 3.0 * self._linear_flops(token_count, hidden, intermediate)

    def _expert_mlp_layer_flops(self, token_count: int, topk: int) -> float:
        hidden = int(self.model.config.hidden_size)
        intermediate = int(self.model.config.intermediate_size)
        return (3.0
                * self._linear_flops(token_count * topk, hidden,
                                     intermediate))

    def _lm_head_flops(self, token_count: int) -> float:
        hidden = int(self.model.config.hidden_size)
        vocab_size = int(self.model.config.vocab_size)
        return self._linear_flops(token_count, hidden, vocab_size)

    def _accumulate_theoretical_flops_stats(
        self,
        seq_lens: List[int],
        scored_token_count: int,
    ) -> None:
        if not self._enable_theoretical_flops_stats:
            return

        self._ensure_yuan_flops_stats()
        switch_layers = {}
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for idx, layer in enumerate(self.model.model.layers):
                mlp = getattr(layer, 'mlp', None)
                if getattr(mlp, 'use_switch', False):
                    switch_layers[idx] = min(
                        int(getattr(mlp, 'router_topk', 0)),
                        int(getattr(mlp, 'num_experts', 0)),
                    )

        total_input_tokens = int(sum(seq_lens))
        self._yuan_flops_stats['total_input_tokens'] += total_input_tokens
        self._yuan_flops_stats['total_scored_tokens'] += int(scored_token_count)
        self._yuan_flops_stats['total_samples'] += len(seq_lens)
        lm_head_flops = self._lm_head_flops(total_input_tokens)
        self._yuan_flops_stats['lm_head_flops'] += lm_head_flops
        self._yuan_flops_stats['total_flops'] += lm_head_flops

        for layer_idx in range(int(self.model.config.num_hidden_layers)):
            layer_stats = self._yuan_flops_stats['per_layer'].setdefault(
                int(layer_idx),
                {
                    'attention_flops': 0.0,
                    'router_flops': 0.0,
                    'expert_flops': 0.0,
                    'dense_mlp_flops': 0.0,
                    'total_flops': 0.0,
                    'input_tokens': 0,
                    'topk': None,
                },
            )
            for seq_len in seq_lens:
                seq_len_int = int(seq_len)
                if seq_len_int <= 0:
                    continue
                attn = self._attention_layer_flops(seq_len_int)
                layer_stats['attention_flops'] += attn['attention_flops']
                layer_stats['total_flops'] += attn['attention_flops']
                layer_stats['input_tokens'] += seq_len_int
                self._yuan_flops_stats['attention_flops'] += (
                    attn['attention_flops'])
                self._yuan_flops_stats['total_flops'] += attn['attention_flops']

                topk = switch_layers.get(layer_idx)
                if topk and topk > 0:
                    router = self._router_layer_flops(seq_len_int)
                    expert_flops = self._expert_mlp_layer_flops(
                        seq_len_int, topk)
                    layer_stats['router_flops'] += router['router_flops']
                    layer_stats['expert_flops'] += expert_flops
                    layer_stats['total_flops'] += (
                        router['router_flops'] + expert_flops)
                    layer_stats['topk'] = int(topk)
                    self._yuan_flops_stats['router_flops'] += (
                        router['router_flops'])
                    self._yuan_flops_stats['expert_flops'] += expert_flops
                    self._yuan_flops_stats['total_flops'] += (
                        router['router_flops'] + expert_flops)
                else:
                    dense_mlp_flops = self._dense_mlp_layer_flops(seq_len_int)
                    layer_stats['dense_mlp_flops'] += dense_mlp_flops
                    layer_stats['total_flops'] += dense_mlp_flops
                    self._yuan_flops_stats['dense_mlp_flops'] += (
                        dense_mlp_flops)
                    self._yuan_flops_stats['total_flops'] += dense_mlp_flops

    def _accumulate_yuan_router_stats(
        self,
        layer_idx: int,
        router_probs_bse: torch.Tensor,
        valid_token_mask: torch.Tensor,
    ) -> None:
        seq_for_loss = min(router_probs_bse.size(1), valid_token_mask.size(1))
        if seq_for_loss <= 0:
            return

        probs = router_probs_bse[:, :seq_for_loss, :]
        mask = valid_token_mask[:, :seq_for_loss]
        token_count = int(mask.sum().item())
        if token_count <= 0:
            return

        p_safe = probs.float().clamp_min(1e-12)
        entropy = -(p_safe * p_safe.log()).sum(dim=-1)
        valid_probs = probs[mask].detach().cpu()
        prob_sum = valid_probs.sum(dim=0)

        layer_stats = self._yuan_router_stats.setdefault(
            int(layer_idx),
            {
                'total_tokens': 0,
                'entropy_sum': 0.0,
                'prob_sum': None,
            },
        )
        layer_stats['total_tokens'] += token_count
        layer_stats['entropy_sum'] += float(entropy[mask].sum().item())
        if layer_stats['prob_sum'] is None:
            layer_stats['prob_sum'] = prob_sum
        else:
            layer_stats['prob_sum'] += prob_sum

    def get_ppl(self, inputs: List[str], mask_length=None):
        """Compute PPL while collecting Yuan router softmax stats."""
        assert self.tokenizer.pad_token, 'pad_token must be set'
        pad_token_id = self.tokenizer.pad_token_id

        tokens = self.tokenizer.batch_encode_plus(
            inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len,
        )
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

        batch_size_tok, seq_len_tok = tokens['input_ids'].shape
        valid_token_mask = (tokens['input_ids'][:, 1:] != pad_token_id)
        seq_lens = (
            (tokens['input_ids'] != pad_token_id)
            .sum(dim=-1)
            .detach()
            .cpu()
            .tolist()
        )
        if mask_length is not None:
            for i, mlen in enumerate(mask_length):
                cutoff = max(int(mlen) - 1, 0)
                if cutoff > 0:
                    valid_token_mask[i, :cutoff] = False
        scored_token_count = int(valid_token_mask.sum().item())
        self._accumulate_theoretical_flops_stats(
            seq_lens=seq_lens,
            scored_token_count=scored_token_count,
        )

        hooks = []
        for _, module in self.model.named_modules():
            if (type(module).__name__ == 'SwitchMLP'
                    and getattr(module, 'use_switch', False)
                    and hasattr(module, 'router')):
                layer_idx = int(getattr(module, 'layer_num', -1))

                def make_hook(cur_layer_idx, switch_module):
                    def hook_fn(mod, inp, out):
                        router_logits = self._to_batch_seq_expert(
                            out, batch_size=batch_size_tok, seq_len=seq_len_tok)
                        if router_logits is None:
                            return
                        router_probs = torch.softmax(router_logits.float(), dim=-1)
                        self._accumulate_yuan_router_stats(
                            cur_layer_idx, router_probs, valid_token_mask)

                        topk = min(
                            int(getattr(switch_module, 'router_topk', 0)),
                            int(getattr(switch_module, 'num_experts', 0)),
                        )
                        if topk > 0 and self._enable_expert_stats:
                            chosen_k = torch.full(
                                (valid_token_mask.size(0),),
                                topk,
                                dtype=torch.int64,
                                device=valid_token_mask.device,
                            )
                            self._accumulate_wrapper_k_usage_stats(
                                layer_idx=cur_layer_idx,
                                chosen_k=chosen_k,
                                valid_token_mask=valid_token_mask,
                            )
                    return hook_fn

                router_module = getattr(module, 'router')
                if type(router_module).__name__ in self._YUAN_ROUTER_TYPES:
                    hooks.append(
                        router_module.register_forward_hook(
                            make_hook(layer_idx, module)))

        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=tokens['input_ids'],
                    attention_mask=tokens.get('attention_mask'),
                    return_dict=False,
                )
                logits = outputs[0]
        finally:
            for hook in hooks:
                hook.remove()

        batch_size, seq_len, vocab_size = logits.shape
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = tokens['input_ids'][:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
            reduction='none',
        ).view(batch_size, seq_len - 1)
        lens = (tokens['input_ids'] != pad_token_id).sum(-1).cpu().numpy()

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask
            lens -= np.array(mask_length)

        return loss.float().sum(-1).cpu().detach().numpy() / lens

    def get_expert_usage_stats(self):
        stats = super().get_expert_usage_stats()
        if stats is None:
            return None

        stats['cross_attention_router_stats_enabled'] = True
        stats['global_cross_attention_router_entropy_mean_nats'] = None
        stats['global_cross_attention_router_entropy_mean_bits'] = None
        stats['theoretical_flops_enabled'] = bool(
            self._enable_theoretical_flops_stats)
        stats['global_theoretical_flops'] = None
        stats['global_theoretical_flops_per_input_token'] = None
        stats['global_theoretical_flops_per_scored_token'] = None
        stats['global_theoretical_flops_per_sample'] = None

        total_entropy_sum = 0.0
        total_entropy_tokens = 0
        per_layer = []

        for layer_entry in stats['per_layer']:
            layer_idx = int(layer_entry['layer_idx'])
            router_stats = self._yuan_router_stats.get(layer_idx)
            if router_stats and router_stats['total_tokens'] > 0:
                entropy_mean_nats = (
                    router_stats['entropy_sum'] / router_stats['total_tokens'])
                entropy_mean_bits = entropy_mean_nats / math.log(2.0)
                prob_mean = (
                    router_stats['prob_sum'] / router_stats['total_tokens'])
                prob_means = {
                    str(i): float(v)
                    for i, v in enumerate(prob_mean.tolist())
                }
                layer_entry['cross_attention_router_total_tokens'] = (
                    router_stats['total_tokens'])
                layer_entry['cross_attention_router_entropy_mean_nats'] = (
                    entropy_mean_nats)
                layer_entry['cross_attention_router_entropy_mean_bits'] = (
                    entropy_mean_bits)
                layer_entry['cross_attention_router_prob_mean'] = prob_means
                layer_entry['cross_attention_router_prob_means'] = prob_means
                total_entropy_sum += router_stats['entropy_sum']
                total_entropy_tokens += router_stats['total_tokens']
            if self._enable_theoretical_flops_stats:
                flops_stats = self._yuan_flops_stats.get('per_layer', {}).get(
                    layer_idx)
                if flops_stats:
                    layer_entry['theoretical_flops'] = (
                        flops_stats['total_flops'])
                    layer_entry['attention_theoretical_flops'] = (
                        flops_stats['attention_flops'])
                    layer_entry['router_theoretical_flops'] = (
                        flops_stats['router_flops'])
                    layer_entry['expert_theoretical_flops'] = (
                        flops_stats['expert_flops'])
                    layer_entry['dense_mlp_theoretical_flops'] = (
                        flops_stats['dense_mlp_flops'])
                    layer_entry['theoretical_flops_input_tokens'] = (
                        flops_stats['input_tokens'])
            per_layer.append(layer_entry)

        stats['per_layer'] = per_layer
        if total_entropy_tokens > 0:
            stats['global_cross_attention_router_entropy_mean_nats'] = (
                total_entropy_sum / total_entropy_tokens)
            stats['global_cross_attention_router_entropy_mean_bits'] = (
                stats['global_cross_attention_router_entropy_mean_nats']
                / math.log(2.0))
        if self._enable_theoretical_flops_stats and self._yuan_flops_stats:
            total_input_tokens = int(
                self._yuan_flops_stats.get('total_input_tokens', 0))
            total_scored_tokens = int(
                self._yuan_flops_stats.get('total_scored_tokens', 0))
            total_samples = int(self._yuan_flops_stats.get('total_samples', 0))
            total_flops = float(self._yuan_flops_stats.get('total_flops', 0.0))
            stats['global_theoretical_flops'] = total_flops
            stats['global_attention_theoretical_flops'] = float(
                self._yuan_flops_stats.get('attention_flops', 0.0))
            stats['global_router_theoretical_flops'] = float(
                self._yuan_flops_stats.get('router_flops', 0.0))
            stats['global_expert_theoretical_flops'] = float(
                self._yuan_flops_stats.get('expert_flops', 0.0))
            stats['global_dense_mlp_theoretical_flops'] = float(
                self._yuan_flops_stats.get('dense_mlp_flops', 0.0))
            stats['global_lm_head_theoretical_flops'] = float(
                self._yuan_flops_stats.get('lm_head_flops', 0.0))
            if total_input_tokens > 0:
                stats['global_theoretical_flops_per_input_token'] = (
                    total_flops / total_input_tokens)
            if total_scored_tokens > 0:
                stats['global_theoretical_flops_per_scored_token'] = (
                    total_flops / total_scored_tokens)
            if total_samples > 0:
                stats['global_theoretical_flops_per_sample'] = (
                    total_flops / total_samples)
        return stats

    def reset_expert_usage_stats(self):
        self._yuan_router_stats = {}
        self._init_yuan_flops_stats()
        super().reset_expert_usage_stats()
