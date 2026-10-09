"""Custom SAM ImageEncoderViT in MLX matching DeepSeek-OCR's vLLM variant.

Key differences vs mlx_vlm SAM:
- No fixed 96x96 resize; spatial size is derived from input (H/patch, W/patch)
- Absolute position embedding is interpolated to current grid (bicubic)
- No SAM-HD branch; simple neck + two stride-2 downsamples
- Output is NHWC
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from PIL import Image


class MLPBlock(nn.Module):
    def __init__(
        self, embedding_dim: int, mlp_dim: int, act: type[nn.Module] = nn.GELU
    ):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def __call__(self, x: mx.array) -> mx.array:
        return self.lin2(self.act(self.lin1(x)))


def _get_abs_pos(abs_pos: mx.array, tgt_grid: int) -> mx.array:
    """Resize absolute position embedding [1, H, W, C] to [1, tgt, tgt, C]."""
    arr = np.array(abs_pos.astype(mx.float32))  # (1, H, W, C)
    _, H, W, C = arr.shape
    if H == tgt_grid and W == tgt_grid:
        return abs_pos
    # to (C, H, W)
    arr = np.transpose(arr, (3, 1, 2, 0))[:, :, :, 0]
    try:
        resample = Image.Resampling.BICUBIC
    except Exception:
        resample = 3
    out = np.empty((C, tgt_grid, tgt_grid), dtype=np.float32)
    for c in range(C):
        im = Image.fromarray(arr[c], mode="F")
        imr = im.resize((tgt_grid, tgt_grid), resample=resample)
        out[c] = np.array(imr, dtype=np.float32)
    out = np.transpose(out, (1, 2, 0))  # (tgt, tgt, C)
    return mx.array(out)[None, :]


def _interp_rel_pos(rel_pos: mx.array, max_rel_dist: int) -> mx.array:
    """Linearly interpolate relative position embeddings to length max_rel_dist.

    rel_pos: [L, C] -> [max_rel_dist, C]
    """
    rel = np.array(rel_pos.astype(mx.float32))  # (L, C)
    L, C = rel.shape
    if L == max_rel_dist:
        return rel_pos
    xp = np.arange(L, dtype=np.float32)
    x = np.linspace(0, L - 1, num=max_rel_dist, dtype=np.float32)
    out = np.empty((max_rel_dist, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = np.interp(x, xp, rel[:, c])
    return mx.array(out)


def _get_rel_pos(q_size: int, k_size: int, rel_pos: mx.array) -> mx.array:
    """Get relative positional embeddings according to relative positions.

    Args:
        q_size: size of query q (height or width)
        k_size: size of key k (height or width)
        rel_pos: [L, C]
    Returns:
        [q_size, k_size, C] indexed embeddings
    """
    rel = np.array(rel_pos)
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel.shape[0] != max_rel_dist:
        rel = np.array(_interp_rel_pos(rel_pos, max_rel_dist))
    # Scale coords if shapes differ
    q_coords = (np.arange(q_size)[:, None] * max(k_size / q_size, 1.0)).astype(
        np.float32
    )
    k_coords = (np.arange(k_size)[None, :] * max(q_size / k_size, 1.0)).astype(
        np.float32
    )
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    relative_coords = relative_coords.astype(np.int64)
    out = rel[relative_coords]
    return mx.array(out)


def window_partition(x: mx.array, window_size: int) -> tuple[mx.array, tuple[int, int]]:
    """Partition into non-overlapping windows with padding if needed.
    Input/Output: NHWC mx.array.
    Returns (windows: [B*num_windows, window, window, C], (Hp, Wp))
    """
    arr = np.array(x)
    B, H, W, C = arr.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
    Hp, Wp = H + pad_h, W + pad_w
    arr = arr.reshape(
        B, Hp // window_size, window_size, Wp // window_size, window_size, C
    )
    windows = arr.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return mx.array(windows), (Hp, Wp)


def window_unpartition(
    windows: mx.array,
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
) -> mx.array:
    """Reverse window partition and remove padding. Input windows: NHWC mx.array."""
    arr = np.array(windows)
    Hp, Wp = pad_hw
    H, W = hw
    B = arr.shape[0] // (Hp * Wp // window_size // window_size)
    arr = arr.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    arr = arr.transpose(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        arr = arr[:, :H, :W, :]
    return mx.array(arr)


def _add_decomposed_rel_pos(
    attn: mx.array,
    q: mx.array,
    rel_pos_h: mx.array,
    rel_pos_w: mx.array,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> mx.array:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = _get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = _get_rel_pos(q_w, k_w, rel_pos_w)

    # q: (B, q_h*q_w, C)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    # einsum equivalents using numpy then back to mx
    r_q_np = np.array(r_q)
    Rh_np = np.array(Rh)
    Rw_np = np.array(Rw)
    rel_h = np.einsum("bhwc,hkc->bhwk", r_q_np, Rh_np)
    rel_w = np.einsum("bhwc,wkc->bhwk", r_q_np, Rw_np)
    attn_np = np.array(attn)
    attn_np = (
        attn_np.reshape(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).reshape(B, q_h * q_w, k_h * k_w)
    return mx.array(attn_np)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None
            self.rel_pos_h = mx.zeros((2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = mx.zeros((2 * input_size[1] - 1, head_dim))

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, _ = x.shape
        # qkv: (3, B, nHeads, H*W, head_dim)
        qkv = (
            self.qkv(x)
            .reshape(B, H * W, 3, self.num_heads, -1)
            .transpose(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1)

        def attn_op(q, k, v):
            attn = (q * self.scale) @ k.transpose(0, -1, -2)
            if self.use_rel_pos:
                attn = _add_decomposed_rel_pos(
                    attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
                )
            attn = mx.softmax(attn, axis=-1)
            out = (
                (attn @ v)
                .reshape(B, self.num_heads, H, W, -1)
                .transpose(0, 2, 3, 1, 4)
                .reshape(B, H, W, -1)
            )
            return out

        out = attn_op(q, k, v)
        return self.proj(out)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rel_pos: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        # Match the LayerNorm epsilon in DeepSeek-OCR's SAM encoder.
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio))
        self.window_size = window_size

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = x
        x_norm = self.norm1(x)
        if self.window_size > 0:
            H, W = int(x_norm.shape[1]), int(x_norm.shape[2])
            x_win, pad_hw = window_partition(x_norm, self.window_size)
            x_attn = self.attn(x_win)
            x_restored = window_unpartition(x_attn, self.window_size, pad_hw, (H, W))
            x = shortcut + x_restored
        else:
            x = shortcut + self.attn(x_norm)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Expect NHWC input
        return self.proj(x)


class ImageEncoderViT_MLX(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        use_abs_pos: bool = True,
        use_rel_pos: bool = True,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11),
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[mx.array] = None
        if use_abs_pos:
            self.pos_embed = mx.zeros(
                (1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = []
        for i in range(depth):
            blk = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(blk)

        self.neck = [
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            nn.LayerNorm(out_chans, eps=1e-6),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm(out_chans, eps=1e-6),
        ]

        # Two stride-2 downsamples to reach 16x16 from 64x64 (or 10x10 from 40x40)
        self.downsamples = [
            nn.Conv2d(out_chans, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        # x: NHWC
        x = self.patch_embed(x)
        H, W = x.shape[1], x.shape[2]
        if self.pos_embed is not None:
            x = x + _get_abs_pos(self.pos_embed, H)
        for blk in self.blocks:
            x = blk(x)
        # neck convs (expect NHWC)
        for n in self.neck:
            x = n(x)
        # downsample convs
        for d in self.downsamples:
            x = d(x)
        return x
