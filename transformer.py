from typing import Optional, Tuple, MutableMapping
from typing import Union
import math
from contextlib import nullcontext

import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend

from einops import rearrange

from utils import si_module, default, exists, load_ckpt

CACHE_FILL_VALUE = -1

def get_cache_len(cache: Optional[Tensor]) -> int:
    """
    cache: (batch, seq_len, 2, kv_heads, head_dim)
    """
    if cache is None:
        return 0
    nonzeros = T.any(cache.flatten(2) != CACHE_FILL_VALUE, dim=-1)
    length = nonzeros.sum(dim=-1).int()
    assert T.all(length == length[0])
    return length[0]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, offset: int = 0):
    assert (
        cos.shape[1] >= offset + x.shape[1]
    ), f"Offset and/or input sequence is too large,\
        \n offset: {offset}, seq_len: {x.shape[1]}, max: {cos.shape[1]}"

    cos_out = cos[:, offset : offset + x.shape[1], :, :]
    sin_out = sin[:, offset : offset + x.shape[1], :, :]

    return (x * cos_out) + (rotate_half(x) * sin_out)


# Adapted from https://github.com/foundation-model-stack/foundation-model-stack
class ShapeRotator:
    def __init__(
        self,
        dim: int,
        end: int,
        theta: float = 10_000,
    ):
        super().__init__()
        self.dim = dim
        self.ratio = theta
        self.cached_freqs: MutableMapping[int, MutableMapping[int, torch.Tensor]] = {}
        self.max_seq_len_cached: MutableMapping[int, int] = {}
        self.ntk_scaling = False
        self.max_seq_len = end

    def compute_freqs_cis(self, device, max_seq_len=None):
        alpha = 1
        dev_idx = device.index
        max_seq_len = default(max_seq_len, self.max_seq_len)

        if dev_idx not in self.cached_freqs:
            self.cached_freqs[dev_idx] = {}
        if dev_idx not in self.max_seq_len_cached:
            self.max_seq_len_cached[dev_idx] = 0


        if self.max_seq_len_cached[dev_idx] > 0:
            return 1
        max_seq_len = max(max_seq_len, self.max_seq_len)

        if (
            1 in self.cached_freqs[dev_idx]
            and max_seq_len <= self.max_seq_len_cached[dev_idx]
        ):
            return 1

        ratio = self.ratio
        dim = self.dim

        freqs = 1.0 / (ratio ** (torch.arange(0, dim, 2, device=device).float() / dim))

        t = torch.arange(max_seq_len, device=device, dtype=freqs.dtype)
        freqs = torch.einsum("i,j->ij", t, freqs)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)

        cos_to_cache = emb.cos()[None, :, None, :]
        sin_to_cache = emb.sin()[None, :, None, :]

        self.max_seq_len_cached[dev_idx] = max_seq_len

        self.cached_freqs[dev_idx][alpha] = torch.stack(
            [
                cos_to_cache,
                sin_to_cache,
            ],
            dim=-1,
        )

        return alpha

    def rotate(
        self,
        q: Tensor,
        k: Tensor,
        offset: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args
        ----
        q : torch.Tensor
            Embedded query tensor, expected size is B x S x H x Eh
        k : torch.Tensor
            Embedded query tensor, expected size is B x S x H x Eh
        """
        assert len(q.size()) == 4
        assert len(k.size()) == 4

        seq_len = self.max_seq_len
        alpha = self.compute_freqs_cis(q.device, seq_len)
        freqs = self.cached_freqs[q.device.index][alpha]

        freqs = freqs.float()  # 1 L D/2 2 2
        q_out = apply_rotary_pos_emb(q, freqs[..., 0], freqs[..., 1], offset=offset).type_as(q)
        k_out = apply_rotary_pos_emb(k, freqs[..., 0], freqs[..., 1], offset=offset).type_as(k)

        return q_out.view_as(q), k_out.view_as(k)

class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, bias=False)

class Norm(nn.Module):
    def __init__(self,
            dim: int,
            eps: float = 1e-5,) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(T.ones((dim,)))

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, (self.weight.shape[0],), weight=self.weight, bias=None, eps=self.eps)
    

class FFNN(nn.Module):
    def __init__(self,
            dim: int,
            expand_dim: int = None,):
        super().__init__()
        expand_dim = default(expand_dim, 256 * ((int(2 * 4 * dim / 3) + 256 - 1) // 256))
        self.dim = dim
        self.expand_dim = expand_dim

        self.gateup_proj = Linear(dim, 2*expand_dim)
        self.down_proj = Linear(expand_dim, dim)

    def forward(self, x):
        gate, up = self.gateup_proj(x).chunk(2, dim=-1)
        return self.down_proj(up * F.silu(gate))

class GQA(nn.Module):
    def __init__(self,
            dim: int, 
            n_head: int, 
            shape_rotator: ShapeRotator,
            kv_heads: Optional[int] = None,
            eps: float = 1e-5,
            causal: bool = True,):
        super().__init__()
        self.n_heads = n_head
        self.kv_heads = default(kv_heads, n_head)
        self.head_dim = dim // n_head
        self.causal = causal

        self.proj_qkv = Linear(dim, self.head_dim*(n_head+2*self.kv_heads))

        self.norm_q = Norm(self.head_dim*n_head, eps=eps)
        self.norm_k = Norm(self.head_dim*self.kv_heads, eps=eps)

        self.attn_out = Linear(dim, dim)

        self.shape_rotator = shape_rotator

    def _sdpa(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        k = k.repeat_interleave(self.n_heads // self.kv_heads, dim=2)
        v = v.repeat_interleave(self.n_heads // self.kv_heads, dim=2)
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2), 
            k.transpose(1, 2), 
            v.transpose(1, 2), 
            is_causal=False if (q.size(1) != k.size(1)) else self.causal,
        )
        x = x.transpose(1, 2).contiguous()
        return x

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, kv_cache: Optional[Tensor] = None,):
        cache_len = get_cache_len(kv_cache)
        q, k = self.shape_rotator.rotate(q, k, offset=cache_len)
        if exists(kv_cache):
            k = T.cat([kv_cache[:, :cache_len, 0], k], dim=1)
            v = T.cat([kv_cache[:, :cache_len, 1], v], dim=1)
            kv_cache[:, :k.size(1), 0] = k
            kv_cache[:, :v.size(1), 1] = v
        x = self._sdpa(q, k, v)
        return self.attn_out(rearrange(x, 'b s h d -> b s (h d)'))

    def _project(self, x):
        full_q, full_k, full_v = self.proj_qkv(x).chunk(3, dim=-1)
        normed_full_q = self.norm_q(full_q).to(full_q.dtype)
        normed_full_k = self.norm_k(full_k).to(full_k.dtype)

        q = rearrange(normed_full_q, 'b s (h d) -> b s h d', h=self.n_heads)
        k = rearrange(normed_full_k, 'b s (h d) -> b s h d', h=self.kv_heads)
        v = rearrange(full_v, 'b s (h d) -> b s h d', h=self.kv_heads)
        return q, k, v

    def forward(self,
            x: Tensor,
            kv: Optional[Tensor] = None,):
        """
        x: (B, S, D)
        kv: (B, S, H, D)
        """
        q, k, v = self._project(x)
        return self._attend(q, k, v, kv_cache=kv)
        

class PreNormAttn(nn.Module):
    def __init__(self,
            dim: int, 
            n_head: int,
            shape_rotator: ShapeRotator,
            kv_heads: Optional[int] = None,
            eps: float = 1e-5,
            causal: bool = True,):
        super().__init__()
        self.attn_norm = Norm(dim, eps=eps)
        self.attn = GQA(dim, n_head, shape_rotator, kv_heads, eps=eps, causal=causal)

    def forward(self, x: Tensor, kv: Optional[Tensor] = None) -> Tensor:
        """
        x: (B, S, D)
        kv: (B, S, H, D)
        """
        return x + self.attn(self.attn_norm(x), kv)

class PreNormFFNN(nn.Module):
    def __init__(self,
            dim: int, 
            ff_dim: int,
            eps: float = 1e-5,):
        super().__init__()
        self.ffnn_norm = Norm(dim, eps=eps)
        self.ffnn = FFNN(dim, ff_dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ffnn(self.ffnn_norm(x))

class Block(nn.Module):
    def __init__(self,
            dim: int,
            layer_id: int = 0,
            n_head: int = 16,
            kv_heads: Optional[int] = None,
            ff_dim: Optional[int] = None,
            eps: float = 1e-5,
            causal: bool = True,
            shape_rotator: ShapeRotator = None):
        super().__init__()
        self.attn = PreNormAttn(dim, n_head, shape_rotator, kv_heads, eps=eps, causal=causal)
        self.ffnn = PreNormFFNN(dim, ff_dim, eps=eps)
        self.dim = dim
        self.layer_id = layer_id
        self.head_dim = dim // n_head
        self.expand_dim = self.ffnn.ffnn.expand_dim

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.dim)
        nn.init.trunc_normal_(self.ffnn.ffnn.gateup_proj.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.attn.attn.proj_qkv.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.attn.attn.attn_out.weight, std=std, a=-3 * std, b=3 * std)

        xstd = 1.0 / math.sqrt(self.expand_dim)
        nn.init.trunc_normal_(self.ffnn.ffnn.down_proj.weight, std=xstd, a=-3 * xstd, b=3 * xstd)

    def forward(self, x: Tensor, kv: Optional[Tensor] = None) -> Tensor:
        """
        x: (B, S, D)
        kv: (B, S, H, D)
        """
        h = self.attn(x, kv)
        out = self.ffnn(h)
        return out



class GPTOutput(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.dim = dim
        self.norm = Norm(dim)
        self.output = Linear(dim, vocab_size)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.dim**2)
        nn.init.trunc_normal_(self.output.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return self.output(self.norm(x))
    
@si_module
class Stack(nn.Module):
    class Config:
        layers: int
        dim: int
        seq_len: int
        n_head: int = 32
        ff_dim: int = None
        kv_heads: int = None
        eps: float = 1e-5
        theta: Union[int, float] = 10_000
        causal: bool = True

        from_pretrained: Optional[Tuple[str, int]] = None

    def __init__(self, c: Config):
        super().__init__()

        from_pretrained = c.from_pretrained
        if exists(from_pretrained):
            checkpoint = load_ckpt(c.from_pretrained)

        self.shape_rotator = ShapeRotator(c.dim//c.n_head, c.seq_len, theta=c.theta)

        self.layers = nn.ModuleList([
            Block(
                dim=c.dim,
                layer_id=l,
                n_head=c.n_head,
                kv_heads=c.kv_heads,
                ff_dim=c.ff_dim,
                eps=c.eps,
                causal=c.causal,
                shape_rotator=self.shape_rotator,
            ) for l in range(c.layers)
        ])

        kv_heads = c.kv_heads or c.n_head
        head_dim = c.dim // c.n_head
        cache_shape = [c.layers, c.seq_len, 2, kv_heads, head_dim]
        self.cache_shape = cache_shape
        self.cache = [None] * c.layers

        if exists(from_pretrained):
            self.load_state_dict(checkpoint)

    def init_cache(self, bsize, device, dtype, length:int=None):
        if self.cache_shape is None:
            return
        cache_shape = self.cache_shape.copy()
        cache_shape[1] = length or cache_shape[1]
        self.cache = T.full((bsize, *cache_shape), CACHE_FILL_VALUE, device=device, dtype=dtype).transpose(0, 1)

    def deinit_cache(self):
        self.cache = [None] * len(self.cache)

    def forward(self, x: Tensor) -> Tensor:
        for l, layer in enumerate(self.layers):
            x = layer(x, kv=self.cache[l])
        return x
