# ============================================================================
# 需要在 /home/cyx/Predict_MoE/Predict_MoE/modeling/modeling_moe_infer.py 中应用的补丁
# ============================================================================

# 1. 在 SwitchMLP.__init__ 方法的末尾（约第 414 行，else: self.mlp = ... 之后）添加：

"""
                # === [NEW] 专家调用统计累积器（仅在 Predict MoE 中启用） ===
                self.enable_expert_stats = bool(getattr(config, "enable_expert_stats", False))
                if self.enable_expert_stats:
                    self._k_call_counts = {}  # {k_value: count}
                    self._total_tokens = 0
                    print(f"[Layer {layer_idx}] Expert usage statistics enabled")
"""

# 2. 在 SwitchMLP.forward 方法中，在 return output_total 之前（约第 817 行）添加：

"""
        # === [NEW] 记录专家调用统计 ===
        if self.use_switch and getattr(self, "enable_expert_stats", False):
            if isinstance(dynamic_k, torch.Tensor):
                # token-wise k: 统计每个k值的出现次数
                k_vals = dynamic_k.flatten().cpu().tolist()
                for k_val in k_vals:
                    self._k_call_counts[k_val] = self._k_call_counts.get(k_val, 0) + 1
                self._total_tokens += len(k_vals)
            elif dynamic_k is not None:
                # layer-wise k
                num_tokens = hidden_states.size(0) * hidden_states.size(1)
                k_int = int(dynamic_k)
                self._k_call_counts[k_int] = self._k_call_counts.get(k_int, 0) + num_tokens
                self._total_tokens += num_tokens
"""

# 3. 在 SwitchMLP 类的末尾添加两个新方法（约第 817 行 return 之后）：

"""
    def get_expert_usage_stats(self):
        \"\"\"获取该层的专家使用统计信息\"\"\"
        if not getattr(self, "enable_expert_stats", False):
            return None
        if self._total_tokens == 0:
            return {
                'distribution': {},
                'total_tokens': 0,
                'avg_k': 0.0
            }
        avg_k = sum(k * cnt for k, cnt in self._k_call_counts.items()) / self._total_tokens
        return {
            'distribution': dict(self._k_call_counts),
            'total_tokens': self._total_tokens,
            'avg_k': avg_k
        }
    
    def reset_expert_usage_stats(self):
        \"\"\"重置统计信息\"\"\"
        if getattr(self, "enable_expert_stats", False):
            self._k_call_counts.clear()
            self._total_tokens = 0
"""

# ============================================================================
# 使用说明
# ============================================================================
"""
将上述三段代码按照注释说明的位置添加到 modeling_moe_infer.py 中。

添加完成后，评测时统计功能会自动启用（仅在 Predict MoE 模型中）。
Dynamic MoE（modeling_moe_ori）不会受影响，因为它没有这些方法。

统计功能通过 enable_expert_stats 配置项控制，已在 dynamic_moe.py 配置文件中启用。
"""
