"""
Qwen2.5-7B-Instruct 完全忠于官方实现的从零复现
==================================================

严格按照 HuggingFace transformers modeling_qwen2.py 的官方实现来写，
包括：
  - RMSNorm: float32 方差计算 + dtype 还原（与官方一致）
  - RoPE: rotate_half 方式（前后半拆分，非交错）
  - GQA: repeat_kv expand+reshape 方式（与官方一致）
  - SwiGLU MLP: gate_proj/up_proj 无 bias, down_proj 无 bias
  - Attention: q/k/v_proj 有 bias, o_proj 无 bias
  - DynamicCache: KV Cache 与官方 Cache.update() 行为一致
  - tie_word_embeddings=False: lm_head 使用独立权重
  - use_sliding_window=False: 7B-Instruct 不使用滑动窗口
用法:
    python qwen25_faithful.py --model_path /path/to/Qwen2.5-7B-Instruct --prompt "你好"

依赖: torch, safetensors, transformers(仅用 tokenizer)
"""

import argparse
import json
import math
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # 只使用第一张 GPU
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


# =============================================================================
# 0. 配置 — 完全从模型目录的文件中读取, 不硬编码任何默认值
# =============================================================================

# config.json 中必须存在的字段 (缺少任何一个都说明模型文件有问题)
REQUIRED_CONFIG_KEYS = [
    "vocab_size", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads",
    "intermediate_size", "rms_norm_eps", "rope_theta",
]


def load_config(model_path: str) -> dict:
    """
    完全从模型目录读取配置，不使用任何硬编码默认值。

    读取顺序:
      1. config.json — 模型架构参数 (必须存在)
      2. generation_config.json — 生成参数 (eos_token_id 等, 覆盖 config.json)
    """
    # --- 1. config.json (必须) ---
    config_file = os.path.join(model_path, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"在 {model_path} 中未找到 config.json，无法确定模型架构参数"
        )

    with open(config_file, "r") as f:
        config = json.load(f)
    print(f"[Config] 从 {config_file} 加载")

    # 校验必须字段
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise ValueError(f"config.json 缺少必要字段: {missing}")

    # --- 2. generation_config.json (可选, 覆盖生成相关参数) ---
    gen_config_file = os.path.join(model_path, "generation_config.json")
    if os.path.exists(gen_config_file):
        with open(gen_config_file, "r") as f:
            gen_config = json.load(f)
        print(f"[Config] 从 {gen_config_file} 加载生成参数")
        # generation_config.json 中的 eos_token_id 优先级更高
        # (Qwen2.5-7B-Instruct: eos_token_id=[151645, 151643])
        for key in ["eos_token_id", "bos_token_id", "pad_token_id",
                     "do_sample", "top_p", "temperature", "max_new_tokens",
                     "repetition_penalty"]:
            if key in gen_config:
                config[key] = gen_config[key]
    else:
        print("[Config] 未找到 generation_config.json，使用 config.json 中的生成参数")

    # --- 3. 派生参数 ---
    config["head_dim"] = config.get("head_dim") or (
        config["hidden_size"] // config["num_attention_heads"]
    )
    config["num_key_value_groups"] = (
        config["num_attention_heads"] // config["num_key_value_heads"]
    )

    # 确保 eos_token_id 是列表
    eos = config.get("eos_token_id", [])
    if isinstance(eos, int):
        eos = [eos]
    config["eos_token_id"] = eos

    # 打印完整配置
    print(f"[Config] 模型架构:")
    print(f"  hidden_size={config['hidden_size']}, "
          f"num_hidden_layers={config['num_hidden_layers']}, "
          f"num_attention_heads={config['num_attention_heads']}, "
          f"num_key_value_heads={config['num_key_value_heads']}")
    print(f"  head_dim={config['head_dim']}, "
          f"intermediate_size={config['intermediate_size']}, "
          f"vocab_size={config['vocab_size']}")
    print(f"  rope_theta={config['rope_theta']}, "
          f"rms_norm_eps={config['rms_norm_eps']}, "
          f"tie_word_embeddings={config.get('tie_word_embeddings', False)}")
    print(f"  eos_token_id={config['eos_token_id']}")
    return config


# =============================================================================
# 1. 权重加载
# =============================================================================

def load_weights(model_path: str) -> dict:
    weights = {}
    model_dir = Path(model_path)

    st_files = sorted(model_dir.glob("*.safetensors"))
    if st_files:
        from safetensors.torch import load_file
        for f in st_files:
            print(f"[Weights] 加载 {f.name} ...")
            weights.update(load_file(str(f), device="cpu"))
        print(f"[Weights] 共 {len(weights)} 个参数 (safetensors)")
        return weights

    bin_files = sorted(model_dir.glob("pytorch_model*.bin"))
    if bin_files:
        for f in bin_files:
            print(f"[Weights] 加载 {f.name} ...")
            weights.update(torch.load(str(f), map_location="cpu", weights_only=True))
        print(f"[Weights] 共 {len(weights)} 个参数 (pytorch bin)")
        return weights

    raise FileNotFoundError(f"在 {model_path} 中未找到权重文件")


# =============================================================================
# 1b. 多卡设备分配
# =============================================================================

def build_device_map(num_layers: int, devices: list[str]) -> dict:
    """
    构建设备分配方案 (pipeline parallel / 按层切分)

    策略:
      - embedding 和 lm_head 放在第一个设备（它们共享显存开销较大的 vocab 矩阵）
      - final norm 放在最后一个设备
      - 28 层 Transformer 均匀分配到所有设备

    例如 2 卡:
      cuda:0: embed_tokens, lm_head, layers 0-13
      cuda:1: layers 14-27, model.norm

    返回:
      {
        "embed": "cuda:0",
        "norm": "cuda:1",
        "lm_head": "cuda:0",
        "layers": ["cuda:0", "cuda:0", ..., "cuda:1", "cuda:1"]  # 长度 = num_layers
      }
    """
    n_devices = len(devices)
    layers_per_device = num_layers // n_devices
    remainder = num_layers % n_devices

    layer_devices = []
    idx = 0
    for d in range(n_devices):
        # 前 remainder 个设备各多分 1 层
        count = layers_per_device + (1 if d < remainder else 0)
        for _ in range(count):
            layer_devices.append(devices[d])
            idx += 1

    return {
        "embed": devices[0],
        "lm_head": devices[0],
        "norm": layer_devices[-1],    # final norm 跟随最后一层
        "layers": layer_devices,
    }


def dispatch_weights(
    weights: dict,
    device_map: dict,
    dtype: torch.dtype,
    num_layers: int,
):
    """
    按 device_map 将每个权重搬到对应的设备。

    命名规则 (Qwen2):
      model.embed_tokens.weight          -> device_map["embed"]
      model.layers.{i}.xxx               -> device_map["layers"][i]
      model.norm.weight                  -> device_map["norm"]
      lm_head.weight                     -> device_map["lm_head"]
    """
    import re

    layer_pattern = re.compile(r"model\.layers\.(\d+)\.")

    for name in list(weights.keys()):
        # 确定这个权重属于哪个设备
        m = layer_pattern.match(name)
        if m:
            layer_idx = int(m.group(1))
            target_device = device_map["layers"][layer_idx]
        elif name.startswith("model.embed_tokens"):
            target_device = device_map["embed"]
        elif name.startswith("model.norm"):
            target_device = device_map["norm"]
        elif name.startswith("lm_head"):
            target_device = device_map["lm_head"]
        else:
            target_device = device_map["embed"]  # fallback

        if weights[name].is_floating_point():
            weights[name] = weights[name].to(dtype=dtype, device=target_device)
        else:
            weights[name] = weights[name].to(device=target_device)


# =============================================================================
# 2. DynamicCache — 与官方 transformers 的 DynamicCache 行为一致
# =============================================================================

class DynamicCache:
    """
    官方 transformers 的 DynamicCache 简化复现。

    每一层维护一个 (key_states, value_states) 对。
    update() 方法将新的 KV 拼接到已有缓存上，返回完整的 KV。

    与官方行为的对应:
      - past_key_values.update(key_states, value_states, layer_idx)
        将新的 key/value 追加到 layer_idx 层的缓存中
      - past_key_values.get_seq_length() 返回已缓存的 token 数
    """

    def __init__(self):
        self.key_cache: list[torch.Tensor] = []    # 每层一个 tensor
        self.value_cache: list[torch.Tensor] = []

    def update(
        self,
        key_states: torch.Tensor,   # (batch, num_kv_heads, new_seq_len, head_dim)
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将新的 KV 追加到缓存中，返回完整的 KV（历史 + 新）。

        与官方 DynamicCache.update() 一致：
        - 首次调用时初始化该层的缓存
        - 后续调用 torch.cat 拼接
        """
        if layer_idx >= len(self.key_cache):
            # 首次写入该层
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # 追加到已有缓存
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2  # 在 seq_len 维度拼接
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self) -> int:
        """返回当前缓存的总 token 数"""
        if len(self.key_cache) == 0:
            return 0
        # 取第 0 层的 seq_len 维度 (dim=2)
        return self.key_cache[0].shape[2]


# =============================================================================
# 3. RMSNorm — 与官方 Qwen2RMSNorm 完全一致
# =============================================================================

def rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    debug: bool = False,
) -> torch.Tensor:
    """
    官方实现 (modeling_qwen2.py Qwen2RMSNorm.forward):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    关键点:
    1. 先转 float32 计算方差（数值稳定性）
    2. rsqrt = 1/sqrt
    3. 最后乘 weight 并转回原 dtype
    """
    input_dtype = hidden_states.dtype

    # 转 float32 做计算
    hidden_states_f32 = hidden_states.to(torch.float32)
    variance = hidden_states_f32.pow(2).mean(-1, keepdim=True)
    hidden_states_normed = hidden_states_f32 * torch.rsqrt(variance + eps)

    # 乘以可学习权重，转回原 dtype
    output = weight * hidden_states_normed.to(input_dtype)

    if debug:
        print(f"      [RMSNorm] var_mean={variance.mean().item():.6f}, "
              f"out_mean={output.mean().item():.6f}, out_std={output.std().item():.6f}")

    return output


# =============================================================================
# 4. RoPE — 与官方 rotate_half + apply_rotary_pos_emb 完全一致
# =============================================================================

def compute_default_rope_parameters(config: dict, device=None) -> tuple:
    """
    官方 Qwen2RotaryEmbedding.compute_default_rope_parameters:
        base = config.rope_theta
        dim = head_dim
        inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
        attention_factor = 1.0
    """
    base = config["rope_theta"]
    dim = config["head_dim"]

    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(
            device=device, dtype=torch.float
        ) / dim)
    )
    attention_scaling = 1.0

    return inv_freq, attention_scaling


def compute_rope_embeddings(
    inv_freq: torch.Tensor,
    attention_scaling: float,
    position_ids: torch.Tensor,   # (batch, seq_len)
    x_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    官方 Qwen2RotaryEmbedding.forward:
        inv_freq_expanded = inv_freq[None, :, None].expand(batch, -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling
    """
    batch_size = position_ids.shape[0]

    # (1, dim/2, 1) -> (batch, dim/2, 1)
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1).to(device)
    # (batch, seq_len) -> (batch, 1, seq_len)
    position_ids_expanded = position_ids[:, None, :].float().to(device)

    # 矩阵乘法: (batch, dim/2, 1) @ (batch, 1, seq_len) -> (batch, dim/2, seq_len)
    # transpose -> (batch, seq_len, dim/2)
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

    # 拼接: (batch, seq_len, dim/2) -> (batch, seq_len, dim)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling

    return cos.to(dtype=x_dtype), sin.to(dtype=x_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    官方 rotate_half:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    注意：这是前后半拆分（不是交错），与 LLaMA 一致。
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,    # (batch, num_heads, seq_len, head_dim)
    k: torch.Tensor,    # (batch, num_kv_heads, seq_len, head_dim)
    cos: torch.Tensor,  # (batch, seq_len, head_dim)
    sin: torch.Tensor,  # (batch, seq_len, head_dim)
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    官方 apply_rotary_pos_emb:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    unsqueeze_dim=1 时: cos/sin (batch, 1, seq_len, head_dim) 可广播到
    q/k (batch, heads, seq_len, head_dim)
    """
    cos = cos.unsqueeze(unsqueeze_dim)  # (batch, 1, seq_len, head_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# =============================================================================
# 5. repeat_kv — 与官方完全一致
# =============================================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    官方 repeat_kv:
        (batch, num_kv_heads, slen, head_dim)
        -> expand to (batch, num_kv_heads, n_rep, slen, head_dim)
        -> reshape to (batch, num_kv_heads * n_rep, slen, head_dim)

    与 torch.repeat_interleave(x, dim=1, repeats=n_rep) 等价但更高效。
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# =============================================================================
# 6. Attention — 使用 SDPA (与官方默认行为一致)
# =============================================================================

def attention_forward(
    query: torch.Tensor,        # (batch, num_q_heads, seq_len, head_dim)
    key: torch.Tensor,          # (batch, num_kv_heads, kv_len, head_dim)
    value: torch.Tensor,        # (batch, num_kv_heads, kv_len, head_dim)
    attention_mask: Optional[torch.Tensor],  # (batch, 1, seq_len, kv_len) or None
    num_key_value_groups: int,
    scaling: float,             # head_dim ** -0.5
    debug: bool = False,
) -> torch.Tensor:
    """
    注意力计算 — 默认使用 SDPA, 与官方 transformers 默认行为一致。

    官方 transformers 默认 attn_implementation="sdpa", 使用
    torch.nn.functional.scaled_dot_product_attention, 它内部会:
    1. 对 causal mask 做优化 (is_causal=True 时用 Flash Attention 内核)
    2. 精度处理更好 (不需要手动用 min_dtype)

    当 SDPA 不可用时 (torch < 2.0), 回退到手动 eager 实现。
    """
    # 扩展 KV heads
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    # 尝试使用 SDPA
    if hasattr(F, "scaled_dot_product_attention"):
        # SDPA 模式 (与官方默认一致)
        if attention_mask is None:
            # Decode (seq_len=1) 或无需 mask: 用 is_causal=False (单 token 看全部)
            # Prefill 时 attention_mask 不为 None
            attn_output = F.scaled_dot_product_attention(
                query, key_states, value_states,
                attn_mask=None,
                is_causal=False,
                scale=scaling,
            )
        elif query.shape[2] == key_states.shape[2]:
            # Prefill: Q 和 KV 长度相同, 可以用 is_causal=True 让 SDPA 内部优化
            attn_output = F.scaled_dot_product_attention(
                query, key_states, value_states,
                attn_mask=None,
                is_causal=True,
                scale=scaling,
            )
        else:
            # 其它情况: 显式传入 mask
            attn_output = F.scaled_dot_product_attention(
                query, key_states, value_states,
                attn_mask=attention_mask,
                is_causal=False,
                scale=scaling,
            )

        if debug:
            print(f"      [Attn] SDPA, Q={query.shape}, KV={key_states.shape}")

    else:
        # Eager 回退 (torch < 2.0)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if debug:
            print(f"      [Attn] Eager, Q={query.shape}, KV={key_states.shape}")

    # transpose: (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


# =============================================================================
# 7. Qwen2Attention — 与官方 Qwen2Attention.forward 一致
# =============================================================================

def qwen2_attention(
    hidden_states: torch.Tensor,       # (batch, seq_len, hidden_size)
    position_embeddings: tuple,        # (cos, sin) 各 (batch, seq_len, head_dim)
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[DynamicCache],
    layer_idx: int,
    weights: dict,
    config: dict,
    debug: bool = False,
) -> tuple[torch.Tensor, Optional[DynamicCache]]:
    """
    官方 Qwen2Attention.forward 的忠实复现:
        1. QKV 投影 (q/k/v 有 bias)
        2. reshape + transpose 到多头格式
        3. apply_rotary_pos_emb
        4. past_key_values.update (KV Cache)
        5. attention_forward (SDPA)
        6. reshape + O 投影 (无 bias)
    """
    prefix = f"model.layers.{layer_idx}.self_attn"
    hidden_size = config["hidden_size"]
    num_q_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    num_kv_groups = config["num_key_value_groups"]
    scaling = head_dim ** -0.5

    input_shape = hidden_states.shape[:-1]  # (batch, seq_len)
    # hidden_shape 用于 view: (*input_shape, num_heads, head_dim)

    # --- 1. QKV 投影 ---
    # 官方: q/k/v_proj 有 bias=True
    q_w = weights[f"{prefix}.q_proj.weight"]
    q_b = weights[f"{prefix}.q_proj.bias"]
    k_w = weights[f"{prefix}.k_proj.weight"]
    k_b = weights[f"{prefix}.k_proj.bias"]
    v_w = weights[f"{prefix}.v_proj.weight"]
    v_b = weights[f"{prefix}.v_proj.bias"]

    query_states = F.linear(hidden_states, q_w, q_b)  # (batch, seq, num_q_heads*head_dim)
    key_states   = F.linear(hidden_states, k_w, k_b)  # (batch, seq, num_kv_heads*head_dim)
    value_states = F.linear(hidden_states, v_w, v_b)  # (batch, seq, num_kv_heads*head_dim)

    if debug:
        print(f"    [Attn] Q proj: shape={query_states.shape}, mean={query_states.mean().item():.6f}")
        print(f"    [Attn] K proj: shape={key_states.shape}, mean={key_states.mean().item():.6f}")
        print(f"    [Attn] V proj: shape={value_states.shape}, mean={value_states.mean().item():.6f}")

    # --- 2. reshape + transpose ---
    # 官方: view(hidden_shape).transpose(1, 2)
    # hidden_shape = (*input_shape, -1, head_dim) = (batch, seq, num_heads, head_dim)
    batch_size, seq_len = input_shape
    query_states = query_states.view(batch_size, seq_len, num_q_heads, head_dim).transpose(1, 2)
    key_states   = key_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    # 现在: (batch, num_heads, seq_len, head_dim)

    # --- 3. RoPE ---
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if debug:
        print(f"    [Attn] After RoPE: Q mean={query_states.mean().item():.6f}, "
              f"K mean={key_states.mean().item():.6f}")

    # --- 4. KV Cache ---
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, layer_idx
        )

    if debug:
        kv_len = key_states.shape[2]
        print(f"    [Attn] KV Cache: kv_len={kv_len} (new_tokens={seq_len}, cached={kv_len - seq_len})")

    # --- 5. Attention ---
    attn_output = attention_forward(
        query_states, key_states, value_states,
        attention_mask=attention_mask,
        num_key_value_groups=num_kv_groups,
        scaling=scaling,
        debug=debug,
    )
    # (batch, seq_len, num_q_heads, head_dim)

    # --- 6. reshape + O 投影 ---
    # 官方: attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    # O 投影 (无 bias)
    o_w = weights[f"{prefix}.o_proj.weight"]
    attn_output = F.linear(attn_output, o_w)  # 无 bias

    if debug:
        print(f"    [Attn] Output (after O proj): mean={attn_output.mean().item():.6f}, "
              f"std={attn_output.std().item():.6f}")

    return attn_output


# =============================================================================
# 8. MLP — 与官方 Qwen2MLP 一致
# =============================================================================

def qwen2_mlp(
    x: torch.Tensor,    # (batch, seq_len, hidden_size)
    layer_idx: int,
    weights: dict,
    debug: bool = False,
) -> torch.Tensor:
    """
    官方 Qwen2MLP.forward:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    gate/up/down_proj 均无 bias。
    act_fn = silu
    """
    prefix = f"model.layers.{layer_idx}.mlp"

    gate_w = weights[f"{prefix}.gate_proj.weight"]
    up_w   = weights[f"{prefix}.up_proj.weight"]
    down_w = weights[f"{prefix}.down_proj.weight"]

    # SiLU(gate(x)) * up(x)
    gate_out = F.silu(F.linear(x, gate_w))   # 无 bias
    up_out   = F.linear(x, up_w)              # 无 bias

    if debug:
        print(f"    [MLP] gate (after SiLU): mean={gate_out.mean().item():.6f}, "
              f"std={gate_out.std().item():.6f}, shape={gate_out.shape}")
        print(f"    [MLP] up: mean={up_out.mean().item():.6f}, "
              f"std={up_out.std().item():.6f}")

    hidden = gate_out * up_out

    if debug:
        print(f"    [MLP] gate*up: mean={hidden.mean().item():.6f}, "
              f"std={hidden.std().item():.6f}")

    # down_proj
    output = F.linear(hidden, down_w)  # 无 bias

    if debug:
        print(f"    [MLP] output: mean={output.mean().item():.6f}, "
              f"std={output.std().item():.6f}")

    return output


# =============================================================================
# 9. DecoderLayer — 与官方 Qwen2DecoderLayer.forward 一致
# =============================================================================

def qwen2_decoder_layer(
    hidden_states: torch.Tensor,
    layer_idx: int,
    weights: dict,
    config: dict,
    attention_mask: Optional[torch.Tensor],
    position_embeddings: tuple,
    past_key_values: Optional[DynamicCache],
    debug: bool = False,
) -> torch.Tensor:
    """
    官方 Qwen2DecoderLayer.forward:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    """
    prefix = f"model.layers.{layer_idx}"
    eps = config["rms_norm_eps"]

    if debug:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")
        print(f"  输入: mean={hidden_states.mean().item():.6f}, "
              f"std={hidden_states.std().item():.6f}, shape={hidden_states.shape}")

    # --- Pre-Attention Norm ---
    residual = hidden_states
    ln1_w = weights[f"{prefix}.input_layernorm.weight"]
    hidden_states = rms_norm(hidden_states, ln1_w, eps=eps, debug=debug)

    if debug:
        print(f"  [1] input_layernorm: mean={hidden_states.mean().item():.6f}, "
              f"std={hidden_states.std().item():.6f}")

    # --- Self Attention ---
    hidden_states = qwen2_attention(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        layer_idx=layer_idx,
        weights=weights,
        config=config,
        debug=debug,
    )

    # --- Residual ---
    hidden_states = residual + hidden_states

    if debug:
        print(f"  [2] residual+attn: mean={hidden_states.mean().item():.6f}, "
              f"std={hidden_states.std().item():.6f}")

    # --- Post-Attention Norm ---
    residual = hidden_states
    ln2_w = weights[f"{prefix}.post_attention_layernorm.weight"]
    hidden_states = rms_norm(hidden_states, ln2_w, eps=eps, debug=debug)

    if debug:
        print(f"  [3] post_attention_layernorm: mean={hidden_states.mean().item():.6f}, "
              f"std={hidden_states.std().item():.6f}")

    # --- MLP ---
    hidden_states = qwen2_mlp(hidden_states, layer_idx, weights, debug=debug)

    # --- Residual ---
    hidden_states = residual + hidden_states

    if debug:
        print(f"  [4] residual+mlp: mean={hidden_states.mean().item():.6f}, "
              f"std={hidden_states.std().item():.6f}")

    return hidden_states


# =============================================================================
# 10. Qwen2Model.forward — 与官方完全一致
# =============================================================================

def qwen2_model_forward(
    input_ids: torch.Tensor,         # (batch, seq_len)
    weights: dict,
    config: dict,
    inv_freq: torch.Tensor,
    attention_scaling: float,
    past_key_values: Optional[DynamicCache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    output_hidden_states: bool = False,
    debug: bool = False,
    debug_layers: Optional[list] = None,
) -> tuple[torch.Tensor, Optional[DynamicCache], Optional[list]]:
    """
    官方 Qwen2Model.forward:
        1. inputs_embeds = embed_tokens(input_ids)
        2. position_ids = arange(seq) + past_seen_tokens
        3. 构造 causal_mask
        4. position_embeddings = rotary_emb(inputs_embeds, position_ids)
        5. 逐层 decoder_layer
        6. norm
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # --- 1. Embedding ---
    embed_w = weights["model.embed_tokens.weight"]
    hidden_states = F.embedding(input_ids, embed_w)

    if debug:
        print(f"\n[Embedding] shape={hidden_states.shape}, "
              f"mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")

    # 收集逐层 hidden states (用于与 HF 对比)
    all_hidden_states = []
    if output_hidden_states:
        all_hidden_states.append(hidden_states.detach().cpu())

    # --- 2. Position IDs ---
    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(seq_len, device=device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)  # (1, seq_len)

    if debug:
        print(f"[Position IDs] {position_ids[0].tolist()[:20]}{'...' if seq_len > 20 else ''}")

    # --- 3. Causal Mask ---
    if attention_mask is None:
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        kv_len = past_len + seq_len

        # 对于 decode (seq_len=1): mask 为 None, 让 attention 函数跳过 mask
        # 单 token 可以看到所有已有 KV, 不需要 mask
        if seq_len == 1:
            causal_mask = None
        else:
            # Prefill: 构造标准因果掩码
            # 使用 bool mask 然后用 masked_fill, 避免 min_dtype 精度问题
            # causal_mask[i][j] = 0 if j <= i + past_len, else min_value
            min_val = torch.finfo(hidden_states.dtype).min
            causal_mask = torch.full(
                (seq_len, kv_len), fill_value=min_val,
                device=device, dtype=hidden_states.dtype
            )
            # triu(diagonal=1): 对角线及以下为 0 (当前位置可以看到自身)
            # past_len=0 时: 标准 triu(diagonal=1)
            # 即 mask[i][j] = min_val if j > i, else 0
            causal_mask = torch.triu(causal_mask, diagonal=past_len + 1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, kv_len)
    else:
        causal_mask = attention_mask

    if debug:
        print(f"[Causal Mask] shape={causal_mask.shape}")

    # --- 4. RoPE embeddings ---
    # 官方: position_embeddings = self.rotary_emb(hidden_states, position_ids)
    cos, sin = compute_rope_embeddings(
        inv_freq, attention_scaling, position_ids,
        x_dtype=hidden_states.dtype, device=device,
    )
    position_embeddings = (cos, sin)

    if debug:
        print(f"[RoPE] cos shape={cos.shape}, sin shape={sin.shape}")

    # --- 5. Transformer layers ---
    # 多卡支持: 当 hidden_states 流经不同设备的层时, 自动搬移
    num_layers = config["num_hidden_layers"]
    device_map = config.get("device_map", None)

    for layer_idx in range(num_layers):
        layer_debug = debug and (debug_layers is None or layer_idx in debug_layers)

        # 多卡: 检查该层是否在不同设备上, 如果是则搬移所有张量
        if device_map is not None:
            layer_device = torch.device(device_map["layers"][layer_idx])
            if hidden_states.device != layer_device:
                if debug:
                    print(f"\n  [Device] 跨卡搬移: {hidden_states.device} -> {layer_device}")
                hidden_states = hidden_states.to(layer_device)
                causal_mask = causal_mask.to(layer_device)
                cos_moved = position_embeddings[0].to(layer_device)
                sin_moved = position_embeddings[1].to(layer_device)
                position_embeddings = (cos_moved, sin_moved)

        hidden_states = qwen2_decoder_layer(
            hidden_states,
            layer_idx=layer_idx,
            weights=weights,
            config=config,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            debug=layer_debug,
        )

        if layer_debug:
            print(f"  Layer {layer_idx:2d}: mean={hidden_states.mean().item():.6f}, "
                  f"std={hidden_states.std().item():.6f}")

    # --- 6. Final Norm ---
    # 多卡: norm 可能在不同设备
    final_ln_w = weights["model.norm.weight"]
    if hidden_states.device != final_ln_w.device:
        hidden_states = hidden_states.to(final_ln_w.device)
    hidden_states = rms_norm(hidden_states, final_ln_w, eps=config["rms_norm_eps"])

    if debug:
        print(f"\n[Final Norm] mean={hidden_states.mean().item():.6f}, "
              f"std={hidden_states.std().item():.6f}")

    return hidden_states, past_key_values


# =============================================================================
# 11. Qwen2ForCausalLM.forward — LM Head
# =============================================================================

def qwen2_causal_lm_forward(
    input_ids: torch.Tensor,
    weights: dict,
    config: dict,
    inv_freq: torch.Tensor,
    attention_scaling: float,
    past_key_values: Optional[DynamicCache] = None,
    position_ids: Optional[torch.Tensor] = None,
    debug: bool = False,
    debug_layers: Optional[list] = None,
) -> tuple[torch.Tensor, Optional[DynamicCache]]:
    """
    官方 Qwen2ForCausalLM.forward:
        outputs = self.model(input_ids, ...)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
    """
    hidden_states, past_key_values = qwen2_model_forward(
        input_ids, weights, config,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        past_key_values=past_key_values,
        position_ids=position_ids,
        debug=debug,
        debug_layers=debug_layers,
    )

    # LM Head
    # 官方: tie_word_embeddings=False for 7B-Instruct, 所以用独立的 lm_head.weight
    if config.get("tie_word_embeddings", False):
        lm_head_w = weights["model.embed_tokens.weight"]
        if debug:
            print("[LM Head] tied weights (复用 embedding)")
    else:
        lm_head_w = weights["lm_head.weight"]
        if debug:
            print(f"[LM Head] 独立权重: shape={lm_head_w.shape}")

    # 多卡: hidden_states 可能在最后一层的设备上, lm_head 在另一个设备
    if hidden_states.device != lm_head_w.device:
        hidden_states = hidden_states.to(lm_head_w.device)

    logits = F.linear(hidden_states, lm_head_w)  # 无 bias

    if debug:
        print(f"[Logits] shape={logits.shape}, "
              f"mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")

    return logits, past_key_values


# =============================================================================
# 12. 生成 (带 KV Cache)
# =============================================================================

def generate(
    prompt_ids: torch.Tensor,       # (1, prompt_len)
    weights: dict,
    config: dict,
    inv_freq: torch.Tensor,
    attention_scaling: float,
    tokenizer=None,                 # 用于打印每步生成的文字
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    top_p: float = 0.9,
    debug: bool = False,
    debug_layers: Optional[list] = None,
) -> list[int]:
    """
    带 KV Cache 的自回归生成:
      - 第一步 (prefill): 将整个 prompt 一次性前向，缓存所有层的 KV
      - 后续步骤 (decode): 每次只输入 1 个新 token，复用缓存的 KV
    """
    device = prompt_ids.device
    prompt_len = prompt_ids.shape[1]

    # 初始化 KV Cache
    past_key_values = DynamicCache()

    # ========== Prefill ==========
    print(f"\n{'*'*70}")
    print(f"* PREFILL: 处理 {prompt_len} 个 prompt tokens")
    print(f"{'*'*70}")

    with torch.no_grad():
        logits, past_key_values = qwen2_causal_lm_forward(
            prompt_ids, weights, config,
            inv_freq=inv_freq,
            attention_scaling=attention_scaling,
            past_key_values=past_key_values,
            debug=debug,
            debug_layers=debug_layers,
        )

    # 取最后一个位置的 logits
    next_logits = logits[0, -1, :]
    next_token = _sample_token(next_logits, temperature, top_p, tokenizer, debug)

    generated_ids = prompt_ids[0].tolist() + [next_token]

    # 检查 EOS
    # config["eos_token_id"] 已在 load_config 中统一为列表
    # Qwen2.5-7B-Instruct: [151645, 151643] = [<|im_end|>, <|endoftext|>]
    eos_ids = config["eos_token_id"]

    # 额外停止符: <|im_start|> (151644)
    # 如果模型在 assistant 回复中生成了新的 <|im_start|>, 说明它在幻觉下一轮对话
    # 官方 HF generate 通过 stopping_criteria 或 chat template 逻辑处理此情况
    # 我们也应该在此处停止
    if tokenizer is not None:
        # 从 tokenizer 中动态获取 im_start token id
        im_start_token = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        if im_start_token:
            stop_ids = set(eos_ids) | set(im_start_token)
        else:
            stop_ids = set(eos_ids)
    else:
        stop_ids = set(eos_ids)

    print(f"[Stop IDs] 停止符: {stop_ids}")

    if next_token in stop_ids:
        print(f"\n[EOS] Prefill 后即遇到结束符 {next_token}")
        return generated_ids

    # ========== Decode ==========
    for step in range(1, max_new_tokens):
        print(f"\n{'*'*70}")
        print(f"* DECODE step {step}/{max_new_tokens}, "
              f"cache_len={past_key_values.get_seq_length()}")
        print(f"{'*'*70}")

        # 只输入最新的 1 个 token
        new_input = torch.tensor([[next_token]], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, past_key_values = qwen2_causal_lm_forward(
                new_input, weights, config,
                inv_freq=inv_freq,
                attention_scaling=attention_scaling,
                past_key_values=past_key_values,
                debug=debug,
                debug_layers=debug_layers,
            )

        next_logits = logits[0, -1, :]
        next_token = _sample_token(next_logits, temperature, top_p, tokenizer, debug)
        generated_ids.append(next_token)

        if next_token in stop_ids:
            print(f"\n[EOS] 遇到结束符 {next_token}，停止生成")
            break

    return generated_ids


def _sample_token(
    logits: torch.Tensor, temperature: float, top_p: float,
    tokenizer=None, debug: bool = False,
) -> int:
    """采样下一个 token: greedy 或 top-p sampling"""

    # Top-5 候选 (调试用)
    top5_probs, top5_ids = torch.topk(F.softmax(logits, dim=-1), 5)

    if temperature == 0.0 or temperature < 1e-8:
        next_token = logits.argmax().item()
        prob = F.softmax(logits, dim=-1)[next_token].item()
    else:
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()

        next_token_idx = torch.multinomial(sorted_probs, 1)
        next_token = sorted_indices[next_token_idx].item()
        prob = probs[next_token].item()

    # 打印 token 详情: ID, 概率, 解码文字
    token_text = repr(tokenizer.decode([next_token])) if tokenizer else "?"
    if debug:
        print(f"[Token] id={next_token}, prob={prob:.4f}, text={token_text}")

    if debug:
        # Top-5 详情
        top5_texts = []
        for tid, tp in zip(top5_ids.tolist(), top5_probs.tolist()):
            t = repr(tokenizer.decode([tid])) if tokenizer else "?"
            top5_texts.append(f"{tid}({t}):{tp:.4f}")
        print(f"[Top-5] {', '.join(top5_texts)}")

    return next_token


# =============================================================================
# 13. 验证: 与 HuggingFace 官方输出对比
# =============================================================================

def verify_against_hf(model_path: str, prompt: str, our_logits: torch.Tensor,
                      our_hidden_states: Optional[list] = None):
    """
    加载官方模型到 GPU，对比 logits + 生成完整回复，验证完立即释放显存。
    our_logits 应已在 CPU 上。
    """
    import gc
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"\n{'='*70}")
        print("验证: 与 HuggingFace 官方模型对比")
        print(f"{'='*70}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 加载到 GPU
        print("  加载官方模型到 GPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # --- 1. Prefill: 拿 logits + 逐层 hidden states 对比 ---
        with torch.no_grad():
            hf_outputs = model(**inputs, output_hidden_states=True)
            hf_logits = hf_outputs.logits.cpu()
            # hidden_states[0] = embedding, hidden_states[i+1] = layer i 的输出
            hf_hidden = [h.cpu() for h in hf_outputs.hidden_states]
        del hf_outputs

        # --- 2. Generate: 让官方模型生成完整回复 ---
        print("  官方模型生成中...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,  # greedy，方便和我们的结果对比
            )
        # 只取新生成的部分
        new_ids = generated_ids[0, inputs["input_ids"].shape[1]:]
        hf_reply = tokenizer.decode(new_ids, skip_special_tokens=True)

        # ============ 释放官方模型 + 清理显存 ============
        del generated_ids, inputs, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  官方模型已释放，显存已清理")

        # ============ 比较 logits ============
        our_last = our_logits[0, -1, :].float()
        hf_last = hf_logits[0, -1, :].float()

        diff = (our_last - hf_last).abs()
        print(f"\n  [Logits 对比]")
        print(f"    差异: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

        our_topk = our_last.topk(5)
        hf_topk = hf_last.topk(5)
        print(f"    Our top-5 tokens:  {our_topk.indices.tolist()}")
        print(f"    HF  top-5 tokens:  {hf_topk.indices.tolist()}")
        print(f"    Top-5 匹配: {our_topk.indices.tolist() == hf_topk.indices.tolist()}")

        # ============ 打印官方生成结果 ============
        print(f"\n  [官方模型生成的回复]")
        print(f"  {hf_reply}")

        # ============ 逐层 hidden states 对比 ============
        if our_hidden_states is not None and len(hf_hidden) > 0:
            print(f"\n  [逐层 Hidden States 对比]")
            print(f"  {'Layer':>8s}  {'Max Diff':>12s}  {'Mean Diff':>12s}  {'Our std':>12s}  {'HF std':>12s}")
            num_compare = min(len(our_hidden_states), len(hf_hidden))
            for i in range(num_compare):
                our_h = our_hidden_states[i].float().cpu()
                hf_h = hf_hidden[i].float()
                diff_h = (our_h - hf_h).abs()
                label = "embed" if i == 0 else f"L{i-1} out" if i < num_compare - 1 else "final"
                print(f"  {label:>8s}  {diff_h.max().item():>12.6f}  {diff_h.mean().item():>12.6f}  "
                      f"{our_h.std().item():>12.6f}  {hf_h.std().item():>12.6f}")
        else:
            print("\n  [逐层对比] 未提供 our_hidden_states，跳过")

    except Exception as e:
        print(f"  验证跳过: {e}")


# =============================================================================
# 14. 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-7B-Instruct 忠于官方的从零实现 (带 KV Cache)"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型权重目录")
    parser.add_argument("--prompt", type=str, default="你好",
                        help="输入提示语")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度 (0=greedy)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p 采样参数")
    parser.add_argument("--debug", action="store_true",
                        help="打印每层详细信息")
    parser.add_argument("--debug_layers", type=str, default=None,
                        help="只对指定层详细打印，如 '0,1,27'")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="auto",
                        help="运行设备: 'cuda', 'cpu', 或 'auto'(自动检测)")
    parser.add_argument("--verify", action="store_true",
                        help="加载 HF 官方模型对比验证")
    args = parser.parse_args()

    debug_layers = None
    if args.debug_layers:
        debug_layers = [int(x) for x in args.debug_layers.split(",")]

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    compute_dtype = dtype_map[args.dtype]

    # 加载配置
    config = load_config(args.model_path)

    # 加载权重
    weights = load_weights(args.model_path)

    # =====================================================================
    # 设备分配: 支持单卡 / 多卡 / CPU
    # =====================================================================
    num_layers = config["num_hidden_layers"]

    if args.device == "auto":
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            device_map = build_device_map(num_layers, ["cpu"])
        elif n_gpus == 1:
            device_map = build_device_map(num_layers, ["cuda:0"])
        else:
            gpu_list = [f"cuda:{i}" for i in range(n_gpus)]
            device_map = build_device_map(num_layers, gpu_list)
    elif "," in args.device:
        # 用户显式指定多卡, 如 --device "cuda:0,cuda:1"
        gpu_list = [d.strip() for d in args.device.split(",")]
        device_map = build_device_map(num_layers, gpu_list)
    else:
        device_map = build_device_map(num_layers, [args.device])

    print(f"\n[Device Map] 设备分配方案:")
    print(f"  embed/lm_head -> {device_map['embed']}")
    print(f"  final_norm    -> {device_map['norm']}")
    for i in range(num_layers):
        if i == 0 or i == num_layers - 1 or device_map["layers"][i] != device_map["layers"][i - 1]:
            end = i
            while end + 1 < num_layers and device_map["layers"][end + 1] == device_map["layers"][i]:
                end += 1
            if i == 0 or device_map["layers"][i] != device_map["layers"][i - 1]:
                print(f"  layers {i}-{end}   -> {device_map['layers'][i]}")

    # 按 device_map 分配权重到各设备
    print(f"\n[Setup] 转换权重到 {args.dtype} 并按 device_map 分配...")
    dispatch_weights(weights, device_map, compute_dtype, num_layers)

    # 打印部分权重清单
    print(f"\n[Weights] 部分权重:")
    for i, name in enumerate(sorted(weights.keys())):
        if i < 15:
            print(f"  {name}: {weights[name].shape} {weights[name].dtype} {weights[name].device}")
    print(f"  ... 共 {len(weights)} 个")

    # inv_freq 放到 embed 所在的设备 (第一层所在设备)
    first_device = torch.device(device_map["embed"])
    inv_freq, attention_scaling = compute_default_rope_parameters(config, device=first_device)
    print(f"[RoPE] inv_freq shape={inv_freq.shape}, device={inv_freq.device}")

    # 把 device_map 存到 config 中, 供 forward 使用
    config["device_map"] = device_map

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 构造 ChatML 输入
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"\n[Prompt] {repr(prompt_text[:200])}{'...' if len(prompt_text) > 200 else ''}")

    # input_ids 放到 embedding 所在的设备
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(first_device)
    print(f"[Tokenize] shape={input_ids.shape}, device={input_ids.device}, 前20: {input_ids[0, :20].tolist()}")

    # 生成
    generated_ids = generate(
        input_ids, weights, config,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        tokenizer=tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        debug=args.debug,
        debug_layers=debug_layers,
    )

    # 解码
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # 只取新生成的部分, 并在第一个停止符处截断
    # 停止符包括 eos_token_id + <|im_start|> (模型幻觉下一轮对话时的标记)
    new_ids = generated_ids[input_ids.shape[1]:]
    eos_ids = config["eos_token_id"]
    im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    all_stop_ids = set(eos_ids) | set(im_start_ids)
    for i, tid in enumerate(new_ids):
        if tid in all_stop_ids:
            new_ids = new_ids[:i]  # 不包含停止符本身
            break
    new_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    print(f"\n{'='*70}")
    print(f"完整输出 (含特殊 token):")
    print(f"{'='*70}")
    print(full_text)
    print(f"\n{'='*70}")
    print(f"生成的回复:")
    print(f"{'='*70}")
    print(new_text)

    # 可选验证
    if args.verify:
        # 重新做一次 prefill 拿 logits 用于对比
        cache = DynamicCache()
        with torch.no_grad():
            verify_input = tokenizer.encode(prompt_text, return_tensors="pt").to(first_device)
            logits, _ = qwen2_causal_lm_forward(
                verify_input, weights, config,
                inv_freq=inv_freq,
                attention_scaling=attention_scaling,
                past_key_values=cache,
                debug=False,
            )
        verify_against_hf(args.model_path, args.prompt, logits.cpu())


def load_qwen_runtime(model_path: str,
                      dtype: str = "bfloat16",
                      device: str = "auto") -> dict:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    compute_dtype = dtype_map[dtype]

    config = load_config(model_path)
    weights = load_weights(model_path)

    num_layers = config["num_hidden_layers"]

    if device == "auto":
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            device_map = build_device_map(num_layers, ["cpu"])
        elif n_gpus == 1:
            device_map = build_device_map(num_layers, ["cuda:0"])
        else:
            gpu_list = [f"cuda:{i}" for i in range(n_gpus)]
            device_map = build_device_map(num_layers, gpu_list)
    elif "," in device:
        gpu_list = [d.strip() for d in device.split(",")]
        device_map = build_device_map(num_layers, gpu_list)
    else:
        device_map = build_device_map(num_layers, [device])

    dispatch_weights(weights, device_map, compute_dtype, num_layers)

    first_device = torch.device(device_map["embed"])
    inv_freq, attention_scaling = compute_default_rope_parameters(config, device=first_device)
    config["device_map"] = device_map

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    return {
        "config": config,
        "weights": weights,
        "tokenizer": tokenizer,
        "inv_freq": inv_freq,
        "attention_scaling": attention_scaling,
        "device": first_device,
    }
def generate_answer(runtime: dict,
                    question: str,
                    context: str,
                    max_new_tokens: int = 128,
                    temperature: float = 0.0,
                    top_p: float = 0.9) -> str:
    tokenizer = runtime["tokenizer"]
    config = runtime["config"]
    weights = runtime["weights"]
    inv_freq = runtime["inv_freq"]
    attention_scaling = runtime["attention_scaling"]
    device = runtime["device"]

    prompt = (
        "Answer the question using only the given context. "
        "Give a short factual answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    messages = [
        {"role": "system", "content": "You are a helpful question answering assistant."},
        {"role": "user", "content": prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    generated_ids = generate(
        prompt_ids=input_ids,
        weights=weights,
        config=config,
        inv_freq=inv_freq,
        attention_scaling=attention_scaling,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        debug=False,
    )

    new_ids = generated_ids[input_ids.shape[1]:]
    eos_ids = config["eos_token_id"]
    im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    stop_ids = set(eos_ids) | set(im_start_ids)

    for i, tid in enumerate(new_ids):
        if tid in stop_ids:
            new_ids = new_ids[:i]
            break

    answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return answer

if __name__ == "__main__":
    main()