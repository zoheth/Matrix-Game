from wan.modules.attention import flash_attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from wan.modules.action_context import ActionContext, BlockMaskFactory
from wan.modules.flashinfer_attention import CausalSelfAttention, FlashInferPlanner
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import os
from einops import rearrange
import torch.distributed as dist
from typing import Optional, List, Dict

# 允许通过环境变量选择使用模块化版本
USE_MODULAR_ACTION = os.environ.get("USE_MODULAR_ACTION", "1") == "1"

if USE_MODULAR_ACTION:
    try:
        from .modular_action import ActionModule
        print("[INFO] Using modular ActionModule implementation")
    except ImportError as e:
        print(f"[WARNING] Failed to import modular ActionModule: {e}")
        print("[INFO] Falling back to original ActionModule")
        from .action_module import ActionModule
else:
    from .action_module import ActionModule

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
# flex_attention = torch.compile(
#     flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    f, h, w = grid_sizes.tolist()

    for i in range(len(x)):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanAttentionBlock(nn.Module):
    """
    Transformer block with self-attention, cross-attention, optional action injection, and FFN.

    Architecture:
        x → [AdaLN + Self-Attention + Gate] → [Cross-Attention] → [Action?] → [AdaLN + FFN + Gate] → out

    Key improvements over original:
    - Eliminated nested function anti-pattern (cross_attn_ffn is now _apply_cross_attn_and_ffn)
    - Clearer variable naming (e → ada_params)
    - Reduced parameter explosion via **action_kwargs
    - Optimized tensor operations with strategic broadcast
    - Better separation of concerns (action logic isolated)
    """

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 cross_attn_norm=False,
                 action_config={},
                 block_idx=0,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Action module (conditional instantiation)
        if len(action_config) != 0 and block_idx in action_config['blocks']:
            from wan.modules.modular_action.action_config import ActionConfig
            # Create ActionConfig from dict and override local_attn_size
            config_dict = {**action_config, 'local_attn_size': local_attn_size}
            self.action_model = ActionModule(ActionConfig.from_dict(config_dict))
        else:
            self.action_model = None

        # Normalization layers
        self.norm1 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # Attention layers - use CausalSelfAttention with num_frame_per_block
        self.self_attn = CausalSelfAttention(
            dim, num_heads, local_attn_size, sink_size, num_frame_per_block, qk_norm, eps
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        # AdaLN modulation parameters [1, 6, C] for (shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        ada_params: torch.Tensor,
        grid_sizes: tuple,
        freqs: torch.Tensor,
        context: torch.Tensor,
        block_mask: BlockMask,
        kv_cache: Optional[dict] = None,
        current_start: int = 0,
        action_context: Optional[ActionContext] = None,
        planner: Optional[FlashInferPlanner] = None,
    ) -> torch.Tensor:
        r"""
        Forward pass through the attention block.

        Args:
            x: Hidden states [B, L, C]
            ada_params: AdaLN modulation parameters [B, F, 6, C] where F = num_frames
            grid_sizes: Spatial-temporal grid (num_frames, height, width) as tuple
            freqs: RoPE frequencies [max_len, head_dim/2]
            context: Visual context for cross-attention
            block_mask: Attention mask for self-attention
            kv_cache: KV cache for self-attention
            current_start: Current position in sequence (for cache indexing)
            action_context: Optional ActionContext encapsulating all action-related parameters
            planner: FlashInferPlanner (must be pre-planned before calling)

        Returns:
            Updated hidden states [B, L, C]
        """
        assert ada_params.ndim == 4, f"Expected ada_params.ndim=4, got {ada_params.ndim}"
        assert isinstance(grid_sizes, tuple) and len(grid_sizes) == 3, \
            f"Expected grid_sizes to be tuple of length 3, got {type(grid_sizes)}"

        L = x.shape[1]
        num_frames = ada_params.shape[1]
        frame_seqlen = L // num_frames

        # Combine learned modulation with input modulation
        # [1, 6, C] + [B, F, 6, C] → [B, F, 6, C]
        combined_modulation = self.modulation.unsqueeze(1) + ada_params

        # Split into 6 components: [B, F, 1, C] each after chunking
        (
            shift_msa, scale_msa, gate_msa, 
            shift_ffn, scale_ffn, gate_ffn
        ) = rearrange(combined_modulation, 'b f six c -> six b f 1 c')

        x_mod = self._adaln_modulate(self.norm1(x), shift_msa, scale_msa, f=num_frames)
        y = self.self_attn(
            x_mod,
            grid_sizes,
            freqs,
            kv_cache,
            current_start,
            planner,
        )
        x = self._adaln_gated_residual(x, y, gate_msa, f=num_frames)

        x = self._apply_condition_attn(
            x, context, grid_sizes,
            current_start, action_context
        )

        x_mod = self._adaln_modulate(self.norm2(x), shift_ffn, scale_ffn, f=num_frames)
        y = self.ffn(x_mod)
        x = self._adaln_gated_residual(x, y, gate_ffn, f=num_frames)

        return x
    
    def _adaln_modulate(self, x_normed: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, f: int) -> torch.Tensor:
        """
        Input:  x_normed [B, T, C], shift/scale [B, F, 1, C]
        Output: [B, T, C] (Modulated)
        """
        x_view = rearrange(x_normed, 'b (f l) c -> b f l c', f=f)
        x_mod = x_view * (1 + scale) + shift
        return rearrange(x_mod, 'b f l c -> b (f l) c')

    def _adaln_gated_residual(self, x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor, f: int) -> torch.Tensor:
        """
        Input:  x [B, T, C] (Residual), y [B, T, C] (Branch), gate [B, F, 1, C]
        Output: [B, T, C] (x + gate * y)
        """
        y_view = rearrange(y, 'b (f l) c -> b f l c', f=f)
        y_gated = rearrange(y_view * gate, 'b f l c -> b (f l) c')
        return x + y_gated

    def _apply_condition_attn(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        grid_sizes: tuple,  # (F, H, W)
        current_start: int,
        action_context: Optional[ActionContext]
    ) -> torch.Tensor:
        """
        Apply cross-attention, optional action module, and FFN with AdaLN.

        This method replaces the nested cross_attn_ffn function.

        Args:
            action_context: Optional ActionContext encapsulating all action parameters
        """
        # Cross-attention
        with torch.profiler.record_function("CausalWanAttentionBlock/cross_attn"):
            x = x + self.cross_attn(
                self.norm3(x.to(context.dtype)),
                context
            )

        # Optional action module
        if self.action_model is not None:
            if action_context is None or not action_context.has_any_condition:
                raise ValueError(
                    "ActionModule is enabled but no ActionContext provided. "
                    "Either pass action_context or use legacy action_kwargs."
                )

            with torch.profiler.record_function("CausalWanAttentionBlock/action_module"):
                # Compute start_frame from current_start
                spatial_tokens_per_frame = int(grid_sizes[1] * grid_sizes[2])
                start_frame = current_start // spatial_tokens_per_frame

                x = self.action_model(
                    x.to(context.dtype),
                    grid_sizes,
                    mouse_condition=action_context.mouse_cond,
                    keyboard_condition=action_context.keyboard_cond,
                    is_causal=True,
                    kv_cache_mouse=action_context.kv_cache_mouse,
                    kv_cache_keyboard=action_context.kv_cache_keyboard,
                    start_frame=start_frame,
                    num_frame_per_block=action_context.num_frame_per_block,
                )

        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C] - Input features (L1 = F * S)
            e(Tensor): Shape [B, F, 1, C] - Conditioning / AdaLN parameters
        """
        combined_style = e + self.modulation.unsqueeze(1)

        shift, scale = rearrange(combined_style, 'b f two c -> two b f 1 c', two=2)

        x = self.norm(x)

        x = rearrange(x, 'b (f s) c -> b f s c', f=e.shape[1])

        x = x * (1 + scale) + shift

        return self.head(x)


def get_rope_freqs_complex(max_seq_len: int, dim: int, theta: float = 10000.0):
    """生成的形状为 [L, D/2] 的复数频率"""
    assert dim % 2 == 0
    # 使用 float64 保证精度
    indices = torch.arange(0, dim, 2, dtype=torch.float64)
    # 计算 theta_i
    freqs = 1.0 / (theta ** (indices / dim))
    # 计算 m * theta_i
    t = torch.arange(max_seq_len, dtype=torch.float64)
    freqs = torch.outer(t, freqs)  # [L, D/2]
    # 转为复数 e^{im\theta}
    return torch.polar(torch.ones_like(freqs), freqs).to(torch.float32)

class CausalWanModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=36,
                 dim=1536,
                 ffn_dim=8960,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=12,
                 num_layers=30,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 cross_attn_norm=True,
                 action_config={},
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention in frames (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            num_frame_per_block (`int`, *optional*, defaults to 1):
                Number of frames per block for block-aligned causal attention mask
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['i2v']
        self.model_type = model_type
        self.use_action_module = len(action_config) > 0
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.num_frame_per_block = num_frame_per_block
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
            

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                local_attn_size, sink_size, self.num_frame_per_block,
                qk_norm, cross_attn_norm, action_config=action_config, eps=eps, block_idx=idx
            )
            for idx in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        head_dim = dim // num_heads

        split_size = head_dim // 6
        dim_h = split_size * 2
        dim_w = split_size * 2
        dim_t = head_dim - dim_h - dim_w
        max_pos = 1024
        self.freqs = torch.cat([
            rope_params(max_pos, dim_t),
            rope_params(max_pos, dim_h),
            rope_params(max_pos, dim_w)
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.block_mask = None
        self.block_mask_keyboard = None
        self.block_mask_mouse = None
        self.use_rope_keyboard = True

    def reset_crossattn_cache(self):
        """Reset cross-attention cache in all blocks."""
        for block in self.blocks:
            block.cross_attn.reset_cache()

    def _get_or_create_masks(
        self,
        device: torch.device,
        num_frames: int,
        frame_seqlen: int,
        block_mask: Optional[BlockMask] = None,
        block_mask_mouse: Optional[BlockMask] = None,
        block_mask_keyboard: Optional[BlockMask] = None
    ):
        """
        Get block masks from arguments or lazily create them.

        This method provides backward compatibility with the old lazy initialization
        while allowing Pipeline to provide pre-computed masks.

        Args:
            device: Device to create masks on
            num_frames: Number of frames
            frame_seqlen: Sequence length per frame
            block_mask: Optional pre-computed visual mask (from Pipeline)
            block_mask_mouse: Optional pre-computed mouse mask
            block_mask_keyboard: Optional pre-computed keyboard mask

        Returns:
            Tuple of (block_mask, block_mask_mouse, block_mask_keyboard)
        """
        # Visual mask: use provided or create lazily
        if block_mask is None:
            if self.block_mask is None:
                self.block_mask = self._prepare_blockwise_causal_attn_mask(
                    device, num_frames=num_frames, frame_seqlen=frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size
                )
            block_mask = self.block_mask
        else:
            # Update cached mask if provided externally
            self.block_mask = block_mask

        # Keyboard mask: use provided or create lazily
        if block_mask_keyboard is None and (self.use_action_module):
            if self.block_mask_keyboard is None:
                if self.use_rope_keyboard == False:
                    self.block_mask_keyboard = self._prepare_blockwise_causal_attn_mask_keyboard(
                        device, num_frames=num_frames, frame_seqlen=frame_seqlen,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask_keyboard = self._prepare_blockwise_causal_attn_mask_action(
                        device, num_frames=num_frames, frame_seqlen=1,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
            block_mask_keyboard = self.block_mask_keyboard
        elif block_mask_keyboard is not None:
            self.block_mask_keyboard = block_mask_keyboard

        # Mouse mask: use provided or create lazily
        if block_mask_mouse is None and (self.use_action_module):
            if self.block_mask_mouse is None:
                self.block_mask_mouse = self._prepare_blockwise_causal_attn_mask_action(
                    device, num_frames=num_frames, frame_seqlen=1,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size
                )
            block_mask_mouse = self.block_mask_mouse
        elif block_mask_mouse is not None:
            self.block_mask_mouse = block_mask_mouse

        return block_mask, block_mask_mouse, block_mask_keyboard

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 9,
        frame_seqlen: int = 880, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_keyboard(
        device: torch.device | str, num_frames: int = 9,
        frame_seqlen: int = 880, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length2 = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length2 = math.ceil(total_length2 / 32) * 32 - total_length2
        padded_length_kv2 = math.ceil(num_frames / 32) * 32 - num_frames
        ends2 = torch.zeros(total_length2 + padded_length2,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices2 = torch.arange(
            start=0,
            end=total_length2,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )
        cnt = num_frame_per_block
        for tmp in frame_indices2:
            ends2[tmp:tmp + frame_seqlen * num_frame_per_block] = cnt
            cnt += num_frame_per_block

        def attention_mask2(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends2[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends2[q_idx]) & (kv_idx >= (ends2[q_idx] - local_attn_size))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask2 = create_block_mask(attention_mask2, B=None, H=None, Q_LEN=total_length2 + padded_length2,
                                       KV_LEN=num_frames + padded_length_kv2, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")

        return block_mask2

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_action(
        device: torch.device | str, num_frames: int = 9,
        frame_seqlen: int = 1, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length2 = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length2 = math.ceil(total_length2 / 32) * 32 - total_length2
        padded_length_kv2 = math.ceil(num_frames / 32) * 32 - num_frames
        ends2 = torch.zeros(total_length2 + padded_length2,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices2 = torch.arange(
            start=0,
            end=total_length2,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )
        cnt = num_frame_per_block
        for tmp in frame_indices2:
            ends2[tmp:tmp + frame_seqlen * num_frame_per_block] = cnt
            cnt += num_frame_per_block

        def attention_mask2(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends2[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends2[q_idx]) & (kv_idx >= (ends2[q_idx] - local_attn_size))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask2 = create_block_mask(attention_mask2, B=None, H=None, Q_LEN=total_length2 + padded_length2,
                                       KV_LEN=num_frames + padded_length_kv2, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")

        return block_mask2

    def forward(
        self,
        x,
        t,
        visual_context,
        cond_concat,
        action_context: Optional[ActionContext] = None,
        kv_cache: dict = None,
        kv_cache_mouse: Optional[List[dict]] = None,
        kv_cache_keyboard: Optional[List[dict]] = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if action_context is not None:
            assert self.use_action_module == True
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = torch.cat([x, cond_concat], dim=1) # B C' F H W

        # embeddings
        with torch.profiler.record_function("CausalWanModel/patch_embedding"):
            x = self.patch_embedding(x)
            # grid_sizes is now a simple tuple (F, H, W)
            grid_sizes = tuple(x.shape[2:])
            # Use rearrange for contiguous output
            x = rearrange(x, 'b c f h w -> b (f h w) c')
            assert x.shape[1] <= 15 * 1 * 880

        with torch.profiler.record_function("CausalWanModel/time_embedding"):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
            e0 = self.time_projection(e).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        # context
        with torch.profiler.record_function("CausalWanModel/visual_embedding"):
            context_lens = None
            context = self.img_emb(visual_context)

        # Ensure ActionContext has correct num_frame_per_block from model config
        if action_context is not None:
            action_context.num_frame_per_block = self.num_frame_per_block

        # Check if using PagedCache and setup FlashInfer planner if so
        planner = None
        first_cache = kv_cache[0] if kv_cache else None
        is_paged_cache = first_cache is not None and hasattr(first_cache, 'get_flashinfer_meta')

        if is_paged_cache:
            head_dim = self.dim // self.num_heads
            planner = FlashInferPlanner(
                num_heads=self.num_heads,
                head_dim=head_dim,
                page_size=first_cache.page_size,
            )

        with torch.profiler.record_function("CausalWanModel/transformer_blocks"):
            for block_index, block in enumerate(self.blocks):
                with torch.profiler.record_function(f"CausalWanModel/block_{block_index}"):
                    # Update ActionContext with block-specific KV caches
                    if action_context is not None:
                        action_context.kv_cache_mouse = kv_cache_mouse[block_index] if kv_cache_mouse else None
                        action_context.kv_cache_keyboard = kv_cache_keyboard[block_index] if kv_cache_keyboard else None

                    # Inference path: direct block call
                    x = block(
                        x,
                        ada_params=e0,
                        grid_sizes=grid_sizes,
                        freqs=self.freqs,
                        context=context,
                        block_mask=self.block_mask,
                        action_context=action_context,
                        kv_cache=kv_cache[block_index],
                        current_start=current_start,
                        planner=planner,
                    )

        # head
        with torch.profiler.record_function("CausalWanModel/head"):
            x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        with torch.profiler.record_function("CausalWanModel/unpatchify"):
            x = self.unpatchify(x, grid_sizes)
        return x

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        bs = x.shape[0]
        x = x.view(bs, *grid_sizes, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(bs, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return x
        

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
        if self.use_action_module:
            for block in self.blocks:
                if block.action_model is not None:
                    # Initialize mouse injector output projection
                    if block.action_model.mouse_injector is not None:
                        nn.init.zeros_(block.action_model.mouse_injector.proj_mouse.weight)
                        if block.action_model.mouse_injector.proj_mouse.bias is not None:
                            nn.init.zeros_(block.action_model.mouse_injector.proj_mouse.bias)
                    # Initialize keyboard injector output projection
                    if block.action_model.keyboard_injector is not None:
                        nn.init.zeros_(block.action_model.keyboard_injector.proj_keyboard.weight)
                        if block.action_model.keyboard_injector.proj_keyboard.bias is not None:
                            nn.init.zeros_(block.action_model.keyboard_injector.proj_keyboard.bias)