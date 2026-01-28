"""
iLLaVA adaptation for Qwen3-VL model.

iLLaVA applies token merging INSIDE the Vision Transformer progressively
across multiple layers, not just after the ViT.

 Key features (following official implementation):
- Attention-based self-selection for token merging
- Progressive merging at multiple layers inside ViT
- Merge low-importance tokens using weighted averaging

Reference: https://github.com/hulianyuyy/iLLaVA
"""

from typing import Optional, Union, List, Tuple
import math
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModelOutputWithPast,
    is_torchdynamo_compiling,
    apply_rotary_pos_emb_vision,
)

def _vision_attention_with_weights(
    attn_module,
    hidden_states: Tensor,
    cu_seqlens: Tensor,
    position_embeddings: Tuple[Tensor, Tensor],
) -> Tuple[Tensor, List[Tensor]]:
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        attn_module.qkv(hidden_states)
        .reshape(seq_length, 3, attn_module.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    attn_outputs = []
    attn_weights_list = []
    offset = 0
    for seg_len in lengths:
        end = offset + seg_len
        q_seg = query_states[offset:end].transpose(0, 1)
        k_seg = key_states[offset:end].transpose(0, 1)
        v_seg = value_states[offset:end].transpose(0, 1)

        attn_weights = torch.matmul(q_seg, k_seg.transpose(1, 2)) * attn_module.scaling
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_seg.dtype)
        attn_output = torch.matmul(attn_weights, v_seg)

        attn_outputs.append(attn_output.transpose(0, 1))
        attn_weights_list.append(attn_weights)
        offset = end

    attn_output = torch.cat(attn_outputs, dim=0).reshape(seq_length, -1)
    attn_output = attn_module.proj(attn_output)
    return attn_output, attn_weights_list


def _downsample_features_to_target(
    features: Tensor,
    target_len: int,
) -> Tensor:
    """
    Downsample features to target length using uniform sampling.
    
    Args:
        features: Features to downsample [seq_len, hidden_dim]
        target_len: Target sequence length
        
    Returns:
        downsampled: Downsampled features [target_len, hidden_dim]
    """
    seq_len = features.shape[0]
    if seq_len == target_len:
        return features
    if seq_len < target_len:
        # If features are smaller than target, pad with last token
        padding = features[-1:].expand(target_len - seq_len, -1)
        return torch.cat([features, padding], dim=0)
    
    # Uniform sampling to downsample
    indices = torch.linspace(0, seq_len - 1, target_len, dtype=torch.long, device=features.device)
    return features[indices]


def _get_video_features_with_illava_compression(
    visual_model,
    pixel_values: Tensor,
    grid_thw: Tensor,
    merge_layers: List[int],
    merge_ratio_per_layer: float,
    final_retention_ratio: float,
) -> Tuple[Tensor, List[Tensor]]:
    """
    Extract video features with iLLaVA token merging applied INSIDE the ViT.
    
    Token merging is applied progressively at specified layers.
    
    Args:
        visual_model: Qwen3VLVisionModel instance
        pixel_values: Video pixel values
        grid_thw: Grid dimensions [num_videos, 3] (time, height, width)
        merge_layers: List of layer indices where to apply merging
        merge_ratio_per_layer: Ratio of tokens to merge at each layer
        final_retention_ratio: Target final retention ratio
        
    Returns:
        hidden_states: Compressed video features after all processing
        deepstack_features: List of deepstack features (all downsampled to final size)
    """
    # Patch embedding
    hidden_states = visual_model.patch_embed(pixel_values)
    
    # Position embedding interpolation
    pos_embeds = visual_model.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds
    
    # Rotary position embedding
    rotary_pos_emb = visual_model.rot_pos_emb(grid_thw)
    
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    
    # Compute cu_seqlens for variable length attention
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    
    # Track whether merging has been applied (affects deepstack feature collection)
    merging_applied = False
    
    # Process through vision blocks with progressive token merging
    deepstack_feature_lists = []
    num_blocks = len(visual_model.blocks)
    
    merge_unit = visual_model.spatial_merge_size ** 2

    for layer_num, blk in enumerate(visual_model.blocks):
        if layer_num in merge_layers:
            attn_output, attn_weights_list = _vision_attention_with_weights(
                blk.attn,
                blk.norm1(hidden_states),
                cu_seqlens,
                position_embeddings,
            )
            hidden_states = hidden_states + attn_output

            # Collect deepstack features BEFORE merging to preserve spatial structure
            if hasattr(visual_model, 'deepstack_visual_indexes') and \
               layer_num in visual_model.deepstack_visual_indexes and not merging_applied:
                deepstack_feature = visual_model.deepstack_merger_list[
                    visual_model.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

            new_hidden_states = []
            new_cos_list = []
            new_sin_list = []
            new_cu_seqlens = [0]
            offset = 0

            for attn_weights in attn_weights_list:
                image_token_length = attn_weights.shape[-1]
                attn_weights_mean = attn_weights.mean(0)
                if attn_weights_mean.ndim == 2:
                    attn_weights_mean = attn_weights_mean.mean(0)

                reduce_tokens_current = math.floor(merge_ratio_per_layer * image_token_length)
                if reduce_tokens_current % merge_unit != 0:
                    if reduce_tokens_current > merge_unit:
                        reduce_tokens_current = (reduce_tokens_current // merge_unit) * merge_unit
                    else:
                        reduce_tokens_current = 0
                reduce_tokens_current = min(reduce_tokens_current, max(image_token_length - 1, 0))

                if reduce_tokens_current > 0:
                    indice = attn_weights_mean.topk(reduce_tokens_current + 1, largest=False)[1]
                    start_index = offset
                    indice = indice + start_index
                    values = hidden_states[indice]
                    size = torch.arange(
                        reduce_tokens_current + 1,
                        0,
                        -1,
                        device=values.device,
                        dtype=values.dtype,
                    )
                    merged = (size.float() @ values.float()) / size.float().sum().to(values.device)
                    hidden_states[indice[-1]] = merged.to(values.dtype)

                    set_all = set(range(start_index, start_index + image_token_length))
                    set_excluded = set(indice[:-1].detach().cpu().numpy().tolist())
                    set_selected = sorted(set_all - set_excluded)
                else:
                    set_selected = list(range(offset, offset + image_token_length))

                new_hidden_states.append(hidden_states[set_selected])
                new_cos_list.append(position_embeddings[0][set_selected])
                new_sin_list.append(position_embeddings[1][set_selected])
                new_cu_seqlens.append(new_cu_seqlens[-1] + image_token_length - reduce_tokens_current)
                offset += image_token_length

            hidden_states = torch.cat(new_hidden_states, dim=0)
            position_embeddings = (torch.cat(new_cos_list, dim=0), torch.cat(new_sin_list, dim=0))
            cu_seqlens = torch.tensor(new_cu_seqlens, dtype=torch.int32, device=hidden_states.device)

            hidden_states = hidden_states + blk.mlp(blk.norm2(hidden_states))
            merging_applied = True
        else:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if hasattr(visual_model, 'deepstack_visual_indexes') and \
               layer_num in visual_model.deepstack_visual_indexes and not merging_applied:
                deepstack_feature = visual_model.deepstack_merger_list[
                    visual_model.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
    
    # Get final token count before merger (this is what LLM will see)
    final_token_count = hidden_states.shape[0]
    
    # Final merger for hidden states
    hidden_states = visual_model.merger(hidden_states)
    
    # Downsample all deepstack features to match final token count
    # This ensures deepstack features match the compressed sequence length
    if deepstack_feature_lists:
        # After merger, the final hidden_states length is what we need to match
        final_merged_len = hidden_states.shape[0]
        downsampled_deepstack = []
        for ds_feat in deepstack_feature_lists:
            downsampled_deepstack.append(
                _downsample_features_to_target(ds_feat, final_merged_len)
            )
        deepstack_feature_lists = downsampled_deepstack
    
    return hidden_states, deepstack_feature_lists


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
    Patched forward with iLLaVA token merging INSIDE the ViT.
    
    Configuration via environment variables:
    - R_RATIO: Token retention ratio (default: 0.25)
    - ILLAVA_MERGE_RATIO: Ratio of tokens to merge per layer (default: 0.5)
    - ILLAVA_LAYERS: Comma-separated layer indices for merging (default: auto)
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

    # Check if image compression should be applied
    image_compression_on = (
        pixel_values is not None
        and image_grid_thw is not None
        and os.getenv("COMPRESS_IMAGE", "0") == "1"
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
    )

    if image_compression_on:
        batch_size = inputs_embeds.shape[0]
        if batch_size != 1:
            image_compression_on = False

    if pixel_values is not None:
        if image_compression_on:
            # Apply iLLaVA merging INSIDE the ViT for images
            retention_ratio = float(os.getenv("R_RATIO", "0.25"))
            merge_ratio_per_layer = float(os.getenv("ILLAVA_MERGE_RATIO", "0.5"))
            
            # Determine which layers to apply merging
            num_blocks = len(self.visual.blocks)
            merge_layers_str = os.getenv("ILLAVA_LAYERS")
            if merge_layers_str:
                merge_layers = [int(x) for x in merge_layers_str.split(",")]
            else:
                # Default: apply merging at multiple layers to achieve target ratio
                merge_layers = [num_blocks // 3, 2 * num_blocks // 3]
            
            pixel_values_typed = pixel_values.type(self.visual.dtype)
            image_embeds_raw, deepstack_image_embeds = _get_video_features_with_illava_compression(
                self.visual,
                pixel_values_typed,
                image_grid_thw,
                merge_layers,
                merge_ratio_per_layer,
                retention_ratio
            )
            
            image_embeds = image_embeds_raw.to(inputs_embeds.device, inputs_embeds.dtype)
            
            # Compute image_mask manually for sequence pruning
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

    # Check if video compression should be applied
    compression_on = (
        pixel_values_videos is not None
        and video_grid_thw is not None
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
    )

    if compression_on:
        batch_size = inputs_embeds.shape[0]
        if batch_size != 1:
            compression_on = False

    if pixel_values_videos is not None:
        if compression_on:
            # Apply iLLaVA merging INSIDE the ViT
            retention_ratio = float(os.getenv("R_RATIO", "0.25"))
            merge_ratio_per_layer = float(os.getenv("ILLAVA_MERGE_RATIO", "0.5"))
            
            # Determine which layers to apply merging
            num_blocks = len(self.visual.blocks)
            merge_layers_str = os.getenv("ILLAVA_LAYERS")
            if merge_layers_str:
                merge_layers = [int(x) for x in merge_layers_str.split(",")]
            else:
                # Default: apply merging at multiple layers to achieve target ratio
                # For 25% retention with 50% merge per layer: need ~2 merge operations
                merge_layers = [num_blocks // 3, 2 * num_blocks // 3]
            
            pixel_values_videos_typed = pixel_values_videos.type(self.visual.dtype)
            video_embeds_raw, deepstack_video_embeds = _get_video_features_with_illava_compression(
                self.visual,
                pixel_values_videos_typed,
                video_grid_thw,
                merge_layers,
                merge_ratio_per_layer,
                retention_ratio
            )
            
            video_embeds = video_embeds_raw.to(inputs_embeds.device, inputs_embeds.dtype)
            
            # For iLLaVA compression, we cannot use get_placeholder_mask because
            # the number of video_embeds (compressed) doesn't match video tokens in input_ids.
            # We compute video_mask manually based on original video token positions.
            video_token_id = self.config.video_token_id
            video_mask = (input_ids == video_token_id)
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            # Note: inputs_embeds will be scattered later after sequence pruning
        else:
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

    # Helper function for attention mask pruning
    def _prune_attention(attn: Optional[torch.Tensor], indices: torch.Tensor) -> Optional[torch.Tensor]:
        if attn is None:
            return None
        if attn.dim() == 2:
            return attn[:, indices]
        if attn.dim() == 4:
            return attn[:, :, indices, :][:, :, :, indices]
        return attn

    if image_compression_on:
        # Prune sequence to match compressed image tokens
        image_token_positions = image_mask[..., 0][0].nonzero(as_tuple=False).squeeze(-1)
        num_compressed_image_tokens = len(image_embeds)
        
        all_positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        non_image_mask = ~image_mask[..., 0][0]
        non_image_positions = all_positions[non_image_mask]
        
        kept_image_positions = image_token_positions[:num_compressed_image_tokens]
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

    if compression_on:
        # Prune sequence to match compressed video tokens
        video_token_positions = video_mask[..., 0][0].nonzero(as_tuple=False).squeeze(-1)
        num_compressed_tokens = len(video_embeds)
        
        all_positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        non_video_mask = ~video_mask[..., 0][0]
        non_video_positions = all_positions[non_video_mask]
        
        kept_video_positions = video_token_positions[:num_compressed_tokens]
        keep_token_indices = torch.cat((non_video_positions, kept_video_positions)).sort().values

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

        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds.to(inputs_embeds.dtype))

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
