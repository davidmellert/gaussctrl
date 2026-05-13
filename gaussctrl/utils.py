import numpy as np
import torch
import gc
from torchvision.transforms import Resize, InterpolationMode
from einops import rearrange
import glob
from diffusers.utils import USE_PEFT_BACKEND

def read_depth2disparity(depth_dir):
    depth_paths = sorted(glob.glob(depth_dir + '/*.npy'))
    disparity_list = []
    for depth_path in depth_paths:
        depth = np.load(depth_path) # [512,512,1] 
        
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / np.max(disparity) # 0.00233~1
        # disparity_map = disparity_map.astype(np.uint8)[:,:,0]
        disparity_map = np.concatenate([disparity_map, disparity_map, disparity_map], axis=2)
        disparity_list.append(disparity_map[None]) 

    detected_maps = np.concatenate(disparity_list, axis=0)
    
    control = torch.from_numpy(detected_maps.copy()).float()
    return rearrange(control, 'f h w c -> f c h w')


def chunked_attention(attn_module, query, key, value, attention_mask=None, chunk_size=512):
    """Memory-efficient attention that avoids materialising the full (B, S, S) score matrix.

    Processes the query sequence in chunks of `chunk_size` tokens so that peak
    memory is O(B * chunk_size * S_k) rather than O(B * S_q * S_k).

    For a 64×64 spatial feature map (S = 4096) with batch 80 in fp16:
      • Standard path:  80 × 4096 × 4096 × 2 B ≈ 2.5 GB  (causes OOM on 11 GB GPUs)
      • This path (chunk_size=512):  80 × 512 × 4096 × 2 B ≈ 320 MB
    """
    dtype = query.dtype
    B, S_q, _ = query.shape
    D_v = value.shape[-1]
    out = torch.zeros(B, S_q, D_v, device=query.device, dtype=dtype)

    upcast = getattr(attn_module, 'upcast_attention', False)
    upcast_softmax = getattr(attn_module, 'upcast_softmax', True)
    scale = attn_module.scale

    for start in range(0, S_q, chunk_size):
        end = min(start + chunk_size, S_q)
        q_c = query[:, start:end]          # (B, chunk, D)

        if upcast:
            q_c_fp, k_fp = q_c.float(), key.float()
        else:
            q_c_fp, k_fp = q_c, key

        scores = torch.bmm(q_c_fp, k_fp.transpose(-1, -2)) * scale  # (B, chunk, S_k)

        if attention_mask is not None:
            # attention_mask after prepare_attention_mask is (B, 1, S_q, S_k) or (B, S_q, S_k).
            # After head_to_batch_dim it is (B*H, 1, S_k) or (B*H, S_q, S_k).
            if attention_mask.ndim == 3:
                scores = scores + attention_mask[:, start:end]
            else:
                scores = scores + attention_mask  # broadcast (B,1,S_k)

        if upcast_softmax:
            scores = scores.float()
        probs = scores.softmax(dim=-1).to(dtype)
        del scores, q_c_fp, k_fp

        v = value.float() if upcast else value
        out[:, start:end] = torch.bmm(probs, v).to(dtype)
        del probs, v

    return out


def compute_attn(attn, query, key, value, video_length, ref_frame_index, attention_mask):
    key_ref_cross = rearrange(key, "(b f) d c -> b f d c", f=video_length)
    key_ref_cross = key_ref_cross[:, ref_frame_index]
    key_ref_cross = rearrange(key_ref_cross, "b f d c -> (b f) d c")
    value_ref_cross = rearrange(value, "(b f) d c -> b f d c", f=video_length)
    value_ref_cross = value_ref_cross[:, ref_frame_index]
    value_ref_cross = rearrange(value_ref_cross, "b f d c -> (b f) d c")

    key_ref_cross = attn.head_to_batch_dim(key_ref_cross)
    value_ref_cross = attn.head_to_batch_dim(value_ref_cross)

    # Use chunked attention to avoid materialising the full attention matrix.
    hidden_states_ref_cross = chunked_attention(attn, query, key_ref_cross, value_ref_cross, attention_mask)

    # Eagerly free the per-ref key/value to keep peak memory low.
    del key_ref_cross, value_ref_cross
    return hidden_states_ref_cross


class CrossViewAttnProcessor:
    def __init__(self, self_attn_coeff, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size
        self.self_attn_coeff = self_attn_coeff

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,):

        residual = hidden_states
        
        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        query = attn.head_to_batch_dim(query)

        if not is_cross_attention:
            # ── Self-attention ──────────────────────────────────────────────────────
            # Use chunked_attention to avoid the O(S²) peak allocation that causes OOM
            # on GPUs with ≤ 12 GB when S = 4096 (64×64 spatial) and batch > ~32.
            key_self  = attn.head_to_batch_dim(key)
            value_self = attn.head_to_batch_dim(value)
            hidden_states_self = chunked_attention(attn, query, key_self, value_self, attention_mask)
            del key_self, value_self

            video_length = key.size()[0] // self.unet_chunk_size
            ref0_frame_index = [0] * video_length
            ref1_frame_index = [1] * video_length
            ref2_frame_index = [2] * video_length
            ref3_frame_index = [3] * video_length

            # ── Cross-view attention ────────────────────────────────────────────────
            # Compute each reference sequentially and accumulate into a running sum
            # instead of keeping all four hidden-state tensors alive simultaneously.
            # This cuts the number of large intermediate tensors from 4 → 1 at a time.
            cross_sum = torch.zeros_like(hidden_states_self)

            for ref_indices in [ref0_frame_index, ref1_frame_index, ref2_frame_index]:
                h = compute_attn(attn, query, key, value, video_length, ref_indices, attention_mask)
                cross_sum.add_(h)
                del h

            # ref3: rearrange key/value to ref3 frames (preserves original behaviour)
            key_r3 = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key_r3 = key_r3[:, ref3_frame_index]
            key_r3 = rearrange(key_r3, "b f d c -> (b f) d c")
            value_r3 = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value_r3 = value_r3[:, ref3_frame_index]
            value_r3 = rearrange(value_r3, "b f d c -> (b f) d c")

            key_r3   = attn.head_to_batch_dim(key_r3)
            value_r3 = attn.head_to_batch_dim(value_r3)

        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if not is_cross_attention:
            # Use chunked attention for ref3 as well
            hidden_states_ref3 = chunked_attention(attn, query, key_r3, value_r3, attention_mask)
            del key_r3, value_r3
            cross_sum.add_(hidden_states_ref3)
            del hidden_states_ref3

            cross_mean = cross_sum.div_(4.0)  # in-place to avoid extra allocation
            del cross_sum

            hidden_states = (
                self.self_attn_coeff * hidden_states_self
                + (1.0 - self.self_attn_coeff) * cross_mean
            )
            del hidden_states_self, cross_mean
        else:
            # Cross-attention: single chunked attention against encoder hidden states
            hidden_states = chunked_attention(attn, query, key, value, attention_mask)

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def free_cuda_memory():
    """Try to free CUDA memory: collect garbage and empty the CUDA cache.

    Call this after deleting large model or tensor references in the caller.
    It cannot delete caller names, so caller should `del` large objects first.
    """
    try:
        gc.collect()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass