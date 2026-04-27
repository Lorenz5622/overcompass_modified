import os
import sys
from typing import Dict, Optional

from .huggingface_dm import HuggingFaceDMMoE


class HuggingFaceAttDMMoE(HuggingFaceDMMoE):
    """ATT-DM-specific MoE wrapper with package-path import fallback."""

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
        moe_package_name: str = "qwen_moe",
        moe_modeling_module: str = "modeling_moe_att_dm",
        moe_config_module: str = "configuration_moe_att_dm",
        moe_code_root: Optional[str] = None,
    ):
        if moe_code_root:
            moe_code_root = os.path.abspath(os.path.expanduser(moe_code_root))
            if os.path.isdir(moe_code_root) and moe_code_root not in sys.path:
                sys.path.insert(0, moe_code_root)
                if not tokenizer_only:
                    print(f"[INFO] Added moe_code_root to sys.path: {moe_code_root}")

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
            print("[INFO] ATT-DM faithful FLOPs statistics enabled")

    def _cross_attention_router_flops(self, token_count: int) -> float:
        """Estimate ATT-DM router FLOPs from actual forward structure."""
        hidden = int(self.model.config.hidden_size)
        num_experts = int(self.model.config.num_experts)
        d_router = int(getattr(self.model.config, 'router_dim', hidden))

        query = self._linear_flops(token_count, hidden, d_router)
        expert_proj = self._linear_flops(token_count * num_experts, hidden,
                                         d_router)
        key_proj = self._linear_flops(token_count * num_experts, d_router,
                                      d_router)
        attn_scores = (self._FLOPS_MULTIPLY_ADD * float(token_count)
                       * float(num_experts) * float(d_router))
        return query + expert_proj + key_proj + attn_scores

    def _expert_mlp_layer_flops(self, token_count: int, topk: int) -> float:
        """ATT-DM evaluates every expert before routing weights are applied."""
        hidden = int(self.model.config.hidden_size)
        intermediate = int(self.model.config.intermediate_size)
        num_experts = int(self.model.config.num_experts)
        expert_forward = (3.0
                          * self._linear_flops(token_count * num_experts,
                                               hidden, intermediate))
        combine = (self._FLOPS_MULTIPLY_ADD * float(token_count)
                   * float(num_experts) * float(hidden))
        return expert_forward + combine
