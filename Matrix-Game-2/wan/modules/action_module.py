from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange
from flash_attn import flash_attn_func
import torch
import torch.nn as nn
from .posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
import math
from torch.nn.attention.flex_attention import flex_attention
from torch.profiler import record_function
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import os
import urllib.request

# Configure matplotlib for Chinese font support
def setup_chinese_font():
    """Setup Chinese font for matplotlib"""
    # Try to find existing Chinese fonts in system
    chinese_fonts = []
    for font in fm.fontManager.ttflist:
        font_name = font.name
        # Check for CJK fonts specifically (must contain 'CJK' or 'SC' or 'CN')
        if any(keyword in font_name for keyword in ['CJK', 'SC', 'SimHei', 'SimSun', 'WenQuanYi', 'Source Han', 'Microsoft YaHei']):
            chinese_fonts.append(font.name)
            print(f"[字体检测] 发现字体: {font.name}")

    if chinese_fonts:
        print(f"[字体] 使用中文字体: {chinese_fonts[0]}")
        matplotlib.rcParams['font.sans-serif'] = [chinese_fonts[0], 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        return

    # Fallback: try generic Noto Sans CJK names
    print("[字体] 尝试使用通用CJK字体名称...")
    for font_name in ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Serif CJK SC']:
        matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        print(f"[字体] 设置为: {font_name}")
        return

    # If no Chinese font found, try to download one
    print("[字体] 未找到中文字体，尝试下载 Noto Sans SC...")
    font_dir = os.path.expanduser("~/.fonts")
    os.makedirs(font_dir, exist_ok=True)
    font_path = os.path.join(font_dir, "NotoSansSC-Regular.otf")

    if not os.path.exists(font_path):
        try:
            url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf"
            print(f"[字体] 正在从 {url} 下载字体...")
            urllib.request.urlretrieve(url, font_path)
            print(f"[字体] 字体已下载到 {font_path}")

            # Add the font to matplotlib
            fm.fontManager.addfont(font_path)
            matplotlib.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
            print("[字体] 字体配置完成")
        except Exception as e:
            print(f"[字体] 下载字体失败: {e}")
            print("[字体] 请手动安装中文字体，如: sudo apt-get install fonts-noto-cjk")
            # Fallback: use system default (will show boxes for Chinese)
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    else:
        # Font file exists, just add it
        try:
            fm.fontManager.addfont(font_path)
            matplotlib.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
            print(f"[字体] 使用已下载的字体: {font_path}")
        except Exception as e:
            print(f"[字体] 加载字体失败: {e}")
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

    matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Setup font when module loads
setup_chinese_font()

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except:
    from flash_attn import flash_attn_func
    FLASH_ATTN_3_AVAILABLE = False


DISABLE_COMPILE = False  # get os env
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
    

class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class ActionModule(nn.Module):
    """
    action module from https://arxiv.org/pdf/2501.08325
    鼠标控制信号的输入是一个 L*D 的向量
    键盘同样
    """

    # Class variables for collecting attention across all layers
    _attn_weights_buffer = []
    _attn_collection_enabled = False
    _attn_save_counter = 0

    def __init__(
        self, 
        mouse_dim_in: int = 2,
        keyboard_dim_in: int = 6,
        hidden_size: int = 128,
        img_hidden_size: int = 1536,
        keyboard_hidden_dim: int = 1024,
        mouse_hidden_dim: int = 1024,
        vae_time_compression_ratio: int = 4, 
        windows_size: int = 3,
        heads_num: int = 16,
        patch_size: list = [1, 2, 2],
        qk_norm: bool = True,
        qkv_bias: bool = False,
        rope_dim_list: list = [8, 28, 28],
        rope_theta = 256,
        mouse_qk_dim_list = [8, 28, 28],
        enable_mouse = True,
        enable_keyboard = True,
        local_attn_size = 6,
        blocks = [],
    ):
        device = None
        
        super().__init__()
        self.local_attn_size = local_attn_size
        self.enable_mouse = enable_mouse
        self.enable_keyboard = enable_keyboard

        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        if self.enable_keyboard:
            self.keyboard_embed = nn.Sequential(nn.Linear(keyboard_dim_in, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))

        self.mouse_qk_dim_list = mouse_qk_dim_list
        self.heads_num = heads_num
        if self.enable_mouse:
            c = mouse_hidden_dim
            self.mouse_mlp = torch.nn.Sequential(
                torch.nn.Linear(mouse_dim_in * vae_time_compression_ratio * windows_size + img_hidden_size, c, bias=True),
                torch.nn.GELU(approximate="tanh"),
                torch.nn.Linear(c, c),
                torch.nn.LayerNorm(c),
            )
            
            head_dim = c // heads_num
            self.t_qkv = nn.Linear(c, c*3, bias=qkv_bias)
            self.img_attn_q_norm = (
                WanRMSNorm(head_dim, eps=1e-6)
                if qk_norm
                else nn.Identity()
            )
            self.img_attn_k_norm = (
                WanRMSNorm(head_dim, eps=1e-6)
                if qk_norm
                else nn.Identity()
            )
            self.proj_mouse = nn.Linear(c, img_hidden_size, bias=qkv_bias)

        if self.enable_keyboard:
            head_dim_key = keyboard_hidden_dim // heads_num
            self.key_attn_q_norm = (
                WanRMSNorm(head_dim_key, eps=1e-6)
                if qk_norm
                else nn.Identity()
            )
            self.key_attn_k_norm = (
                WanRMSNorm(head_dim_key, eps=1e-6)
                if qk_norm
                else nn.Identity()
            )
            
            self.mouse_attn_q = nn.Linear(img_hidden_size, keyboard_hidden_dim, bias=qkv_bias)
            self.keyboard_attn_kv = nn.Linear(hidden_size * windows_size * vae_time_compression_ratio, keyboard_hidden_dim * 2, bias=qkv_bias)
            self.proj_keyboard = nn.Linear(keyboard_hidden_dim, img_hidden_size, bias=qkv_bias)

        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.windows_size = windows_size
        self.patch_size = patch_size
        self.freqs_cos, self.freqs_sin = self.get_rotary_pos_embed(7500, self.patch_size[1], self.patch_size[2], 64, self.mouse_qk_dim_list, start_offset=0)

    def patchify(self, x, patch_size):
        """
        x : (N C T H W)
        """
        pt, ph, pw = self.patch_size
        t, h, w = x.shape[2] //  pt, x.shape[3] // ph, x.shape[4] // pw
        c = x.shape[1]
        x = x.reshape(shape=(x.shape[0], c, t , pt, h , ph, w , pw))
        x = torch.einsum("nctohpwq->nthwcopq", x)
        x = x.reshape(shape=(x.shape[0], t*h*w,  c*pt*ph*pw))
        return x

    def unpatchify(self, x, t, h, w, patch_size):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c =  x.shape[2] // patch_size #self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def get_rotary_pos_embed(self, video_length, height, width, head_dim, rope_dim_list = None, start_offset=0):
        target_ndim = 3
        ndim = 5 - 2
        
        latents_size = [video_length+start_offset, height, width]

        if isinstance(self.patch_size, int):
            assert all(s % self.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.patch_size for s in latents_size]
        elif isinstance(self.patch_size, list):
            assert all(
                s % self.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [
                s // self.patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos[-video_length*rope_sizes[1]*rope_sizes[2]//self.patch_size[0]:], freqs_sin[-video_length*rope_sizes[1]*rope_sizes[2]//self.patch_size[0]:]

    def forward(self, x, tt, th, tw, mouse_condition=None, keyboard_condition=None, block_mask_mouse=None, block_mask_keyboard=None, is_causal=False, kv_cache_mouse=None, kv_cache_keyboard=None, start_frame=0, use_rope_keyboard=True, num_frame_per_block=3):
        '''
        hidden_states: B, tt*th*tw, C
        mouse_condition: B, N_frames, C1
        keyboard_condition: B, N_frames, C2
        '''
        with record_function("Action_Module"):
            assert use_rope_keyboard == True

            B, N_frames, C = keyboard_condition.shape
            assert tt*th*tw == x.shape[1]
            assert ((N_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0
            N_feats = int((N_frames - 1) / self.vae_time_compression_ratio) + 1

            # Defined freqs_cis early so it's available for both mouse and keyboard
            freqs_cis = (self.freqs_cos, self.freqs_sin)

            assert (N_feats == tt and ((is_causal and kv_cache_mouse == None) or not is_causal)) or ((N_frames - 1) // self.vae_time_compression_ratio + 1 == start_frame + num_frame_per_block and is_causal)

            if self.enable_mouse and mouse_condition is not None:
                hidden_states = rearrange(x, "B (T S) C -> (B S) T C", T=tt, S=th*tw) # 65*272*480 -> 17*(272//16)*(480//16) -> 8670
                B, N_frames, C = mouse_condition.shape
            else:
                hidden_states = x
            # padding

            pad_t = self.vae_time_compression_ratio * self.windows_size
            if self.enable_mouse and mouse_condition is not None:
                pad = mouse_condition[:, 0:1, :].expand(-1, pad_t, -1)
                mouse_condition = torch.cat([pad, mouse_condition], dim=1)
                if is_causal and kv_cache_mouse is not None: 
                    mouse_condition = mouse_condition[:, self.vae_time_compression_ratio*(N_feats - num_frame_per_block - self.windows_size) + pad_t:, :] 
                    group_mouse = [mouse_condition[:, self.vae_time_compression_ratio*(i - self.windows_size) + pad_t:i * self.vae_time_compression_ratio + pad_t,:] for i in range(num_frame_per_block)]
                else:
                    group_mouse = [mouse_condition[:, self.vae_time_compression_ratio*(i - self.windows_size) + pad_t:i * self.vae_time_compression_ratio + pad_t,:] for i in range(N_feats)]
                
                group_mouse = torch.stack(group_mouse, dim = 1)

                S = th * tw 
                group_mouse = group_mouse.unsqueeze(-1).expand(B, num_frame_per_block, pad_t, C, S)
                group_mouse = group_mouse.permute(0, 4, 1, 2, 3).reshape(B * S, num_frame_per_block, pad_t * C) 

                group_mouse = torch.cat([hidden_states, group_mouse], dim = -1)
                group_mouse = self.mouse_mlp(group_mouse)
            # qkv
                mouse_qkv = self.t_qkv(group_mouse)
                q, k, v = rearrange(mouse_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num) # BHW F H C
                q = self.img_attn_q_norm(q).to(v)
                k = self.img_attn_k_norm(k).to(v)        
            # rope embd

            
            # freqs_cis = (self.freqs_cos, self.freqs_sin)
            
            
                q, k = apply_rotary_emb(q, k, freqs_cis, start_offset = start_frame, head_first=False)
            ## TODO: adding cache here
                if is_causal:
                    if kv_cache_mouse is None:
                        assert q.shape[0] ==  k.shape[0] and q.shape[0] % 880 == 0 # == 880, f"{q.shape[0]},{k.shape[0]}"
                        padded_length = math.ceil(q.shape[1] / 32) * 32 - q.shape[1]
                        padded_q = torch.cat(
                            [q,
                                torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                            device=q.device, dtype=v.dtype)],
                            dim=1
                        )
                        padded_k = torch.cat(
                            [k, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                                    device=k.device, dtype=v.dtype)],
                            dim=1
                        )
                        padded_v = torch.cat(
                            [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                            device=v.device, dtype=v.dtype)],
                            dim=1
                        )
                        attn = flex_attention(
                            query=padded_q.transpose(2, 1), # after: B, HW, F, C
                            key=padded_k.transpose(2, 1),
                            value=padded_v.transpose(2, 1),
                            block_mask=block_mask_mouse
                        )[:, :, :-padded_length].transpose(2, 1)
                    else:
                        current_start = start_frame
                        current_end = current_start + q.shape[1]
                    
                        assert q.shape[1] == num_frame_per_block
                        sink_size = 0
                        max_attention_size = self.local_attn_size
                        sink_tokens = sink_size * 1
                        kv_cache_size = kv_cache_mouse["k"].shape[1]
                        num_new_tokens = q.shape[1]
                    

                        if (current_end > kv_cache_mouse["global_end_index"].item()) and (
                            num_new_tokens + kv_cache_mouse["local_end_index"].item() > kv_cache_size):
                            num_evicted_tokens = num_new_tokens + kv_cache_mouse["local_end_index"].item() - kv_cache_size
                            num_rolled_tokens = kv_cache_mouse["local_end_index"].item() - num_evicted_tokens - sink_tokens
                            kv_cache_mouse["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                                kv_cache_mouse["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                            kv_cache_mouse["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                                kv_cache_mouse["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        # Insert the new keys/values at the end
                            local_end_index = kv_cache_mouse["local_end_index"].item() + current_end - \
                                kv_cache_mouse["global_end_index"].item() - num_evicted_tokens
                            local_start_index = local_end_index - num_new_tokens
                        else:
                            local_end_index = kv_cache_mouse["local_end_index"].item() + current_end - kv_cache_mouse["global_end_index"].item()
                            local_start_index = local_end_index - num_new_tokens
                        kv_cache_mouse["k"][:, local_start_index:local_end_index] = k
                        kv_cache_mouse["v"][:, local_start_index:local_end_index] = v

                        if FLASH_ATTN_3_AVAILABLE:
                            attn, attn_prob = flash_attn_interface.flash_attn_func(
                                q,
                                kv_cache_mouse["k"][:, max(0, local_end_index - max_attention_size):local_end_index],
                                kv_cache_mouse["v"][:, max(0, local_end_index - max_attention_size):local_end_index],
                            )
                        else:
                            attn = flash_attn_func(
                                q,
                                kv_cache_mouse["k"][:, max(0, local_end_index - max_attention_size):local_end_index],
                                kv_cache_mouse["v"][:, max(0, local_end_index - max_attention_size):local_end_index],
                            )
                            print(q.shape)
                            print(kv_cache_mouse["k"][:, max(0, local_end_index - max_attention_size):local_end_index].shape)

                            # Collect attention heatmap on 3rd iteration (start_frame == 3)
                            if start_frame == 42:
                                # Get q and k tensors
                                q_tensor = q  # [880, 3, 16, 64]
                                k_tensor = kv_cache_mouse["k"][:, max(0, local_end_index - max_attention_size):local_end_index]  # [880, 6, 16, 64]

                                # Compute naive attention scores: Q @ K^T / sqrt(d)
                                q_attn = q_tensor.transpose(1, 2)  # [880, 16, 3, 64]
                                k_attn = k_tensor.transpose(1, 2)  # [880, 16, 6, 64]

                                # Compute attention scores
                                scale = 1.0 / math.sqrt(q_tensor.shape[-1])  # 1/sqrt(64)
                                attn_scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) * scale  # [880, 16, 3, 6]

                                # Apply softmax to get attention weights
                                attn_weights = torch.softmax(attn_scores, dim=-1)  # [880, 16, 3, 6]

                                # Add to buffer for averaging across layers
                                ActionModule._attn_weights_buffer.append(attn_weights.detach())
                                ActionModule._attn_collection_enabled = True
                                ActionModule._attn_save_counter += 1

                                # Check if we should save (after collecting enough layers, e.g., 20+)
                                if len(ActionModule._attn_weights_buffer) >= 20:
                                    print(f"[Attention Heatmap] Collected {len(ActionModule._attn_weights_buffer)} layers, saving averaged heatmap...")

                                    # Average all collected attention weights across layers
                                    attn_avg_layers = torch.stack(ActionModule._attn_weights_buffer).mean(dim=0)  # [880, 16, 3, 6]

                                    # Create output directory
                                    output_dir = "attention_heatmaps"
                                    os.makedirs(output_dir, exist_ok=True)

                                    # 批次池化：将880个batch重塑为22组，每组40个batch
                                    batch_size = attn_avg_layers.shape[0]  # 880
                                    pool_size = 40  # 每40个batch做一次池化
                                    num_groups = batch_size // pool_size  # 22

                                    # 重塑并池化：[880, 16, 3, 6] -> [22, 40, 16, 3, 6] -> [22, 3, 6]
                                    attn_pooled = attn_avg_layers[:num_groups * pool_size].reshape(num_groups, pool_size, 16, 3, 6)
                                    attn_pooled = attn_pooled.mean(dim=(1, 2))  # [22, 3, 6] - 对pool_size和heads维度求平均

                                    # 重塑为 [22*3, 6] = [66, 6] 用于可视化
                                    attn_heatmap = attn_pooled.reshape(-1, 6)  # [66, 6]

                                    # 转移到CPU并转换为numpy
                                    attn_heatmap_np = attn_heatmap.cpu().float().numpy()

                                    # 创建详细热力图
                                    fig, ax = plt.subplots(figsize=(8, 12))
                                    im = ax.imshow(attn_heatmap_np, aspect='auto', cmap='hot', interpolation='nearest')

                                    # 添加颜色条
                                    plt.colorbar(im, ax=ax, label='注意力权重')

                                    # 设置标签和标题
                                    ax.set_xlabel('K位置', fontsize=12)
                                    ax.set_ylabel('Q位置（池化后的批次组）', fontsize=12)
                                    ax.set_title(f'注意力热力图 - 帧{start_frame}（{len(ActionModule._attn_weights_buffer)}层平均）\n' +
                                               f'[{num_groups}个批次组 × 3个Q token] × [6个K token]\n' +
                                               'K[0-2]: 第1次迭代（较早历史），K[3-5]: 第2次迭代（较近历史）',
                                               fontsize=11, pad=15)

                                    # 添加x轴标签
                                    ax.set_xticks([0, 1, 2, 3, 4, 5])
                                    ax.set_xticklabels(['K0\n(第1次)', 'K1\n(第1次)', 'K2\n(第1次)',
                                                       'K3\n(第2次)', 'K4\n(第2次)', 'K5\n(第2次)'])

                                    # 添加垂直线分隔K的两个部分
                                    ax.axvline(x=2.5, color='cyan', linestyle='--', linewidth=2, alpha=0.7,
                                              label='分界线：第1次 vs 第2次迭代')
                                    ax.legend(loc='upper right', fontsize=9)

                                    # 每3行添加一条水平网格线（分隔批次组）
                                    for i in range(0, num_groups * 3, 3):
                                        if i > 0:
                                            ax.axhline(y=i - 0.5, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

                                    plt.tight_layout()
                                    plt.savefig(os.path.join(output_dir, f'attn_heatmap_detailed_layeravg_frame{start_frame}.png'), dpi=200)
                                    plt.close()

                                    # 创建汇总视图
                                    attn_summary = attn_pooled.mean(dim=0)  # [3, 6]
                                    attn_summary_np = attn_summary.cpu().float().numpy()

                                    fig, ax = plt.subplots(figsize=(8, 5))
                                    im = ax.imshow(attn_summary_np, aspect='auto', cmap='hot', interpolation='nearest')
                                    plt.colorbar(im, ax=ax, label='注意力权重')

                                    ax.set_xlabel('K位置', fontsize=12)
                                    ax.set_ylabel('Q位置（当前token）', fontsize=12)
                                    ax.set_title(f'注意力汇总 - 帧{start_frame}（{len(ActionModule._attn_weights_buffer)}层平均）\n' +
                                               'K[0-2]: 第1次迭代（较早），K[3-5]: 第2次迭代（较近）',
                                               fontsize=11, pad=15)

                                    ax.set_xticks([0, 1, 2, 3, 4, 5])
                                    ax.set_xticklabels(['K0\n(第1次)', 'K1\n(第1次)', 'K2\n(第1次)',
                                                       'K3\n(第2次)', 'K4\n(第2次)', 'K5\n(第2次)'])
                                    ax.set_yticks([0, 1, 2])
                                    ax.set_yticklabels(['Q0\n(第3次)', 'Q1\n(第3次)', 'Q2\n(第3次)'])

                                    ax.axvline(x=2.5, color='cyan', linestyle='--', linewidth=2, alpha=0.7)

                                    plt.tight_layout()
                                    plt.savefig(os.path.join(output_dir, f'attn_summary_layeravg_frame{start_frame}.png'), dpi=150)
                                    plt.close()

                                    # 计算统计信息
                                    first_half_attn = attn_summary_np[:, :3].mean()  # K[0-2]: 第1次迭代
                                    second_half_attn = attn_summary_np[:, 3:].mean()  # K[3-5]: 第2次迭代

                                    print(f"[注意力统计] K[0-2]（第1次迭代，较早历史）: {first_half_attn:.4f}")
                                    print(f"[注意力统计] K[3-5]（第2次迭代，较近历史）: {second_half_attn:.4f}")
                                    print(f"[注意力统计] 比例（较近/较早）: {second_half_attn/first_half_attn:.4f}")
                                    print(f"[注意力热力图] 已保存到 {output_dir}/")

                                    # Clear buffer for next round
                                    ActionModule._attn_weights_buffer.clear()
                                    ActionModule._attn_save_counter = 0

                        kv_cache_mouse["global_end_index"].fill_(current_end)
                        kv_cache_mouse["local_end_index"].fill_(local_end_index)
                else:
                    attn = flash_attn_func(
                            q, # 880, f, 16, 64
                            k, # 880, f, 16, 64
                            v, # 880, f, 16, 64
                        )
            # Compute cu_squlens and max_seqlen for flash attention
            # qk norm
                attn = rearrange(attn, '(b S) T h d -> b (T S) (h d)',b=B)
            
                hidden_states = rearrange(x, "(B S) T C -> B (T S) C", B=B)
                attn = self.proj_mouse(attn)
            
                hidden_states = hidden_states + attn
        
            if self.enable_keyboard and keyboard_condition is not None:
                pad = keyboard_condition[:, 0:1, :].expand(-1, pad_t, -1)
                keyboard_condition = torch.cat([pad, keyboard_condition], dim=1)
                if is_causal and kv_cache_keyboard is not None:
                    keyboard_condition = keyboard_condition[:, self.vae_time_compression_ratio*(N_feats - num_frame_per_block - self.windows_size) + pad_t:, :] # keyboard_condition[:, self.vae_time_compression_ratio*(start_frame - self.windows_size) + pad_t:start_frame * self.vae_time_compression_ratio + pad_t,:]
                    keyboard_condition = self.keyboard_embed(keyboard_condition)
                    group_keyboard = [keyboard_condition[:, self.vae_time_compression_ratio*(i - self.windows_size) + pad_t:i * self.vae_time_compression_ratio + pad_t,:] for i in range(num_frame_per_block)]
                else:
                    keyboard_condition = self.keyboard_embed(keyboard_condition)
                    group_keyboard = [keyboard_condition[:, self.vae_time_compression_ratio*(i - self.windows_size) + pad_t:i * self.vae_time_compression_ratio + pad_t,:] for i in range(N_feats)]
                group_keyboard = torch.stack(group_keyboard, dim = 1) # B F RW C
                group_keyboard = group_keyboard.reshape(shape=(group_keyboard.shape[0],group_keyboard.shape[1],-1))
            # apply cross attn
                mouse_q = self.mouse_attn_q(hidden_states)
                keyboard_kv = self.keyboard_attn_kv(group_keyboard)

                B, L, HD = mouse_q.shape
                D = HD // self.heads_num
                q = mouse_q.view(B, L, self.heads_num, D)

                B, L, KHD = keyboard_kv.shape
                k, v = keyboard_kv.view(B, L, 2, self.heads_num, D).permute(2, 0, 1, 3, 4)
        
            # Compute cu_squlens and max_seqlen for flash attention
            # qk norm
            
                q = self.key_attn_q_norm(q).to(v)
                k = self.key_attn_k_norm(k).to(v)
                S = th * tw
                assert S == 880
            # position embed 
                if use_rope_keyboard: 
                    B, TS, H, D = q.shape
                    T_ = TS // S 
                    q = q.view(B, T_, S, H, D).transpose(1, 2).reshape(B * S, T_, H, D)
                    q, k = apply_rotary_emb(q, k, freqs_cis, start_offset = start_frame,head_first=False)

                    k1, k2, k3, k4 = k.shape
                    k = k.expand(S, k2, k3, k4)
                    v = v.expand(S, k2, k3, k4)


                    if is_causal:
                        if kv_cache_keyboard is None:
                            assert q.shape[0] == k.shape[0] and q.shape[0] % 880 == 0 

                            padded_length = math.ceil(q.shape[1] / 32) * 32 - q.shape[1]
                            padded_q = torch.cat(
                                [q,
                                    torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                                device=q.device, dtype=v.dtype)],
                                dim=1
                            )
                            padded_k = torch.cat(
                                [k, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                                        device=k.device, dtype=v.dtype)],
                                dim=1
                            )
                            padded_v = torch.cat(
                                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                                device=v.device, dtype=v.dtype)],
                                dim=1
                            )
                            attn = flex_attention(
                                query=padded_q.transpose(2, 1), # after: B, HW, F, C
                                key=padded_k.transpose(2, 1),
                                value=padded_v.transpose(2, 1),
                                block_mask=block_mask_keyboard
                            )[:, :, :-padded_length].transpose(2, 1)
                        else:
                            current_start = start_frame
                            current_end = current_start + k.shape[1]
                            assert k.shape[1] == num_frame_per_block
                            sink_size = 0
                            max_attention_size = self.local_attn_size
                            sink_tokens = sink_size * 1
                            kv_cache_size = kv_cache_keyboard["k"].shape[1]
                            num_new_tokens = k.shape[1]

                            if (current_end > kv_cache_keyboard["global_end_index"].item()) and (
                                num_new_tokens + kv_cache_keyboard["local_end_index"].item() > kv_cache_size):
                                num_evicted_tokens = num_new_tokens + kv_cache_keyboard["local_end_index"].item() - kv_cache_size
                                num_rolled_tokens = kv_cache_keyboard["local_end_index"].item() - num_evicted_tokens - sink_tokens
                                kv_cache_keyboard["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                                    kv_cache_keyboard["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                                kv_cache_keyboard["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                                    kv_cache_keyboard["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                            # Insert the new keys/values at the end
                                local_end_index = kv_cache_keyboard["local_end_index"].item() + current_end - \
                                    kv_cache_keyboard["global_end_index"].item() - num_evicted_tokens
                                local_start_index = local_end_index - num_new_tokens
                            else:
                                local_end_index = kv_cache_keyboard["local_end_index"].item() + current_end - kv_cache_keyboard["global_end_index"].item()
                                local_start_index = local_end_index - num_new_tokens
                            assert k.shape[0] == 880 # BS == 1 or the cache should not be saved/ load method should be modified
                            kv_cache_keyboard["k"][:, local_start_index:local_end_index] = k[:1]
                            kv_cache_keyboard["v"][:, local_start_index:local_end_index] = v[:1]

                            if FLASH_ATTN_3_AVAILABLE:
                                attn, attn_prob = flash_attn_interface.flash_attn_func(
                                    q,
                                    kv_cache_keyboard["k"][:, max(0, local_end_index - max_attention_size):local_end_index].repeat(S, 1, 1, 1),
                                    kv_cache_keyboard["v"][:, max(0, local_end_index - max_attention_size):local_end_index].repeat(S, 1, 1, 1),
                                )
                            else:
                                attn = flash_attn_func(
                                    q,
                                    kv_cache_keyboard["k"][:, max(0, local_end_index - max_attention_size):local_end_index].repeat(S, 1, 1, 1),
                                    kv_cache_keyboard["v"][:, max(0, local_end_index - max_attention_size):local_end_index].repeat(S, 1, 1, 1),
                                )

                            kv_cache_keyboard["global_end_index"].fill_(current_end)
                            kv_cache_keyboard["local_end_index"].fill_(local_end_index)
                    else:
                        attn = flash_attn_func(
                                q, # 1, f*880, 16, 64
                                k, # 1, f, 16, 64
                                v, # 1, f, 16, 64
                                causal=False,
                            )
                    attn = rearrange(attn, '(B S) T H D -> B (T S) (H D)', S=S)
                else:
                    if is_causal:
                        if kv_cache_keyboard is None:
                        
                            padded_length = math.ceil(q.shape[1] / 32) * 32 - q.shape[1]
                            padded_q = torch.cat(
                                [q,
                                    torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                                device=q.device, dtype=v.dtype)],
                                dim=1
                            )
                            padded_k = torch.cat(
                                [k, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                                        device=k.device, dtype=v.dtype)],
                                dim=1
                            )
                            padded_v = torch.cat(
                                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                                device=v.device, dtype=v.dtype)],
                                dim=1
                            )
                            attn = flex_attention(
                                query=padded_q.transpose(2, 1), # after: B, HW, F, C
                                key=padded_k.transpose(2, 1),
                                value=padded_v.transpose(2, 1),
                                block_mask=block_mask_keyboard
                            )[:, :, :-padded_length].transpose(2, 1)
                        else:
                            current_start = start_frame
                            current_end = current_start + k.shape[1]
                            assert k.shape[1] == num_frame_per_block
                            sink_size = 0
                            local_attn_size = self.local_attn_size
                            max_attention_size = self.local_attn_size
                            sink_tokens = sink_size * 1
                            kv_cache_size = kv_cache_keyboard["k"].shape[1]
                            num_new_tokens = k.shape[1]


                            if (current_end > kv_cache_keyboard["global_end_index"].item()) and (
                                num_new_tokens + kv_cache_keyboard["local_end_index"].item() > kv_cache_size):
                                num_evicted_tokens = num_new_tokens + kv_cache_keyboard["local_end_index"].item() - kv_cache_size
                                num_rolled_tokens = kv_cache_keyboard["local_end_index"].item() - num_evicted_tokens - sink_tokens
                                kv_cache_keyboard["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                                    kv_cache_keyboard["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                                kv_cache_keyboard["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                                    kv_cache_keyboard["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                            # Insert the new keys/values at the end
                                local_end_index = kv_cache_keyboard["local_end_index"].item() + current_end - \
                                    kv_cache_keyboard["global_end_index"].item() - num_evicted_tokens
                                local_start_index = local_end_index - num_new_tokens

                            
                            else:
                                local_end_index = kv_cache_keyboard["local_end_index"].item() + current_end - kv_cache_keyboard["global_end_index"].item()
                                local_start_index = local_end_index - num_new_tokens
                            kv_cache_keyboard["k"][:, local_start_index:local_end_index] = k
                            kv_cache_keyboard["v"][:, local_start_index:local_end_index] = v
                            attn = flash_attn_func(
                                q,
                                kv_cache_keyboard["k"][:, max(0, local_end_index - max_attention_size):local_end_index],
                                kv_cache_keyboard["v"][:, max(0, local_end_index - max_attention_size):local_end_index],
                            # causal=is_causal
                            )
                            kv_cache_keyboard["global_end_index"].fill_(current_end)
                            kv_cache_keyboard["local_end_index"].fill_(local_end_index)
                    else:
                        attn = flash_attn_func(
                                q, # 1, f*880, 16, 64
                                k, # 1, f, 16, 64
                                v, # 1, f, 16, 64
                            # causal=is_causal,
                            )
                    attn = rearrange(attn, 'B L H D -> B L (H D)')
                attn = self.proj_keyboard(attn)
                hidden_states = hidden_states + attn
            return hidden_states