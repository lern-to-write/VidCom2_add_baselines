"""
CDPruner adaptation for Qwen3-VL model.

CDPruner applies conditional DPP pruning to visual tokens using the current text
instruction for relevance. This implementation follows the official algorithm and
adapts the scoring to Qwen3-VL's vision features.

Reference: https://github.com/Theia-4869/CDPruner
"""

from typing import Optional, Union, List, Tuple
import os
import torch
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModelOutputWithPast,
    is_torchdynamo_compiling,
)


def _compute_text_embeds(
    input_ids: Tensor,
    inputs_embeds: Tensor,
    attention_mask: Optional[torch.Tensor],
    image_token_id: int,
    video_token_id: int,
    pad_token_id: Optional[int],
) -> Tensor:
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask is not None and not isinstance(attention_mask, dict):
        if attention_mask.dim() == 2 and attention_mask.shape[1] == input_ids.shape[1]:    
            
            mask &= attention_mask.bool()
    mask &= input_ids != image_token_id
    mask &= input_ids != video_token_id
    if pad_token_id is not None:
        mask &= input_ids != pad_token_id
    if mask.sum() == 0:
        mask = (input_ids != image_token_id) & (input_ids != video_token_id)
    mask_f = mask.unsqueeze(-1).to(inputs_embeds.dtype)
    pooled = (inputs_embeds * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
    return pooled


def _cdpruner_select_indices(
    image_features: Tensor,
    text_embed: Tensor,
    keep_tokens: int,
) -> Tensor:
    num_tokens = image_features.shape[0]
    if keep_tokens <= 0 or keep_tokens >= num_tokens:
        return torch.arange(num_tokens, device=image_features.device)

    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)
    image_normalized = image_normalized.float()
    similarity = torch.matmul(image_normalized, image_normalized.transpose(0, 1))

    image_embeds = image_features / image_features.norm(dim=-1, keepdim=True)
    image_embeds = image_embeds.float()
    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
    text_embed = text_embed.float()

    relevance = torch.matmul(image_embeds, text_embed.unsqueeze(-1)).squeeze(-1)
    relevance = (-relevance)
    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min() + 1e-6)

    kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0)

    cis = torch.zeros((keep_tokens, num_tokens), device=image_features.device)
    di2s = torch.diagonal(kernel, dim1=0, dim2=1).clone()
    select_idx = torch.empty((keep_tokens,), dtype=torch.long, device=image_features.device)
    for i in range(keep_tokens):
        j = torch.argmax(di2s)
        select_idx[i] = j
        if i == keep_tokens - 1:
            break
        if i == 0:
            eis = kernel[j] / torch.sqrt(di2s[j])
        else:
            eis = (kernel[j] - torch.einsum("t,tn->n", cis[:i, j], cis[:i])) / torch.sqrt(di2s[j])
        cis[i, :] = eis
        di2s -= torch.square(eis)
        di2s[j] = -float("inf")

    return torch.sort(select_idx).values


def Qwen3VLModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[tuple, Qwen3VLModelOutputWithPast]:
    """
    Patched forward with CDPruner token selection.

    Configuration via environment variables:
    - CDPRUNER_TOKENS: Number of visual tokens to keep per image (default: keep all)
    - COMPRESS_IMAGE: Set to "1" to enable image compression (default: "0")
    """

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None
    deepstack_image_embeds = None
    deepstack_video_embeds = None
    image_keep_indices = None

    text_embeds = None
    if input_ids is not None:
        text_embeds = _compute_text_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            pad_token_id=getattr(self.config, "pad_token_id", None),
        )

    image_compression_on = (
        pixel_values is not None
        and image_grid_thw is not None
        and os.getenv("COMPRESS_IMAGE", "0") == "1"
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
        and text_embeds is not None
    )

    if image_compression_on:
        batch_size = inputs_embeds.shape[0]
        if batch_size != 1:
            image_compression_on = False

    if pixel_values is not None:
        if image_compression_on:
            visual_token_num = int(os.getenv("CDPRUNER_TOKENS", "0"))
            image_embeds_list, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)

            split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
            deepstack_splits = (
                [list(torch.split(layer, split_sizes)) for layer in deepstack_image_embeds]
                if deepstack_image_embeds
                else []
            )
            kept_indices: List[Tensor] = []
            kept_image_chunks: List[Tensor] = []
            kept_deepstack: List[List[Tensor]] = [[] for _ in deepstack_splits]
            offset = 0

            total_original_tokens = 0
            total_kept_tokens = 0
            for image_idx, image_feat in enumerate(image_embeds_list):
                original_tokens = image_feat.shape[0]
                keep_tokens = visual_token_num if visual_token_num > 0 else original_tokens
                keep_local = _cdpruner_select_indices(image_feat, text_embeds[0], keep_tokens)
                kept_indices.append(keep_local + offset)
                kept_image_chunks.append(image_feat[keep_local])
                for layer_idx, layer_splits in enumerate(deepstack_splits):
                    layer_chunk = layer_splits[image_idx]
                    kept_deepstack[layer_idx].append(layer_chunk[keep_local])
                offset += original_tokens
                total_original_tokens += original_tokens
                total_kept_tokens += len(keep_local)
                
                # Print per-image compression info
                print(f"[CDPruner] Image {image_idx + 1} compression:")
                print(f"  - Original tokens: {original_tokens}")
                print(f"  - Kept tokens: {len(keep_local)}")
                print(f"  - Removed tokens: {original_tokens - len(keep_local)}")
                print(f"  - Retention ratio: {len(keep_local) / original_tokens * 100:.2f}%")

            image_keep_indices = torch.sort(torch.cat(kept_indices)).values
            image_embeds = torch.cat(kept_image_chunks, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            
            # Print total compression summary
            print(f"[CDPruner] Total Token Compression Summary:")
            print(f"  - Total images: {len(image_embeds_list)}")
            print(f"  - Total original tokens: {total_original_tokens}")
            print(f"  - Total kept tokens: {total_kept_tokens}")
            print(f"  - Total removed tokens: {total_original_tokens - total_kept_tokens}")
            print(f"  - Overall retention ratio: {total_kept_tokens / total_original_tokens * 100:.2f}%")
            if kept_deepstack:
                deepstack_image_embeds = [torch.cat(chunks, dim=0) for chunks in kept_deepstack]

            image_token_id = self.config.image_token_id
            image_mask = (input_ids == image_token_id)
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        else:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
        attention_mask_tensor = attention_mask if not isinstance(attention_mask, dict) else attention_mask.get(
            "full_attention", None
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )

        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    def _prune_attention(attn: Optional[torch.Tensor], indices: torch.Tensor) -> Optional[torch.Tensor]:
        if attn is None:
            return None
        if attn.dim() == 2:
            return attn[:, indices]
        if attn.dim() == 4:
            return attn[:, :, indices, :][:, :, :, indices]
        return attn

    if image_compression_on and image_keep_indices is not None:
        image_token_positions = image_mask[..., 0][0].nonzero(as_tuple=False).squeeze(-1)
        kept_image_positions = image_token_positions[image_keep_indices]

        all_positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        non_image_positions = all_positions[~image_mask[..., 0][0]]
        keep_token_indices = torch.cat((non_image_positions, kept_image_positions)).sort().values

        inputs_embeds = inputs_embeds[:, keep_token_indices, :]
        if input_ids is not None:
            input_ids = input_ids[:, keep_token_indices]
        attention_mask = (
            {k: _prune_attention(v, keep_token_indices) for k, v in attention_mask.items()}
            if isinstance(attention_mask, dict)
            else _prune_attention(attention_mask, keep_token_indices)
        )
        position_ids = position_ids[:, :, keep_token_indices]
        if cache_position is not None:
            cache_position = torch.arange(
                inputs_embeds.shape[1], 
                device=inputs_embeds.device, 
                dtype=cache_position.dtype
            )

        if image_mask is not None:
            image_mask = image_mask[:, keep_token_indices, :]
        if video_mask is not None:
            video_mask = video_mask[:, keep_token_indices, :]

        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.dtype))

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        image_mask_compact = image_mask[..., 0]
        video_mask_compact = video_mask[..., 0]
        visual_pos_masks = image_mask_compact | video_mask_compact
        deepstack_visual_embeds = []
        image_mask_joint = image_mask_compact[visual_pos_masks]
        video_mask_joint = video_mask_compact[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask_compact = image_mask[..., 0]
        visual_pos_masks = image_mask_compact
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask_compact = video_mask[..., 0]
        visual_pos_masks = video_mask_compact
        deepstack_visual_embeds = deepstack_video_embeds

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )
