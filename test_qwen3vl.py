import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import os
from loguru import logger as eval_logger

# 模型名称 (保持你原本的输入)
model_id = "/mnt/data/public/back/huggingface/Qwen3-VL-4B-Instruct"

# 1. 加载模型
# 注意：多图场景强烈建议开启 flash_attention_2，否则显存容易溢出且速度慢
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # 建议指定 bf16
    attn_implementation="sdpa", # 强烈推荐开启
    device_map="auto",
)

#############################################################################

# Token compression integration based on COMPRESSOR environment variable
compressor = os.getenv("COMPRESSOR")
if compressor == "vidcom2":
    import types
    from token_compressor.vidcom2.models.qwen3_vl import Qwen3VLModel_forward
    model.model.forward = types.MethodType(Qwen3VLModel_forward, model.model)
    eval_logger.success("[VidCom2] Successfully integrated VidCom2 with Qwen3-VL.")
elif compressor == "fastv":
    import types
    from token_compressor.fastv.models.qwen3_vl import Qwen3VLModel_forward
    model.model.forward = types.MethodType(Qwen3VLModel_forward, model.model)
    fastv_k = os.getenv("FASTV_K", "2")
    r_ratio = os.getenv("R_RATIO", "0.5")
    eval_logger.success(f"[FastV] Successfully integrated FastV with Qwen3-VL. (K={fastv_k}, R_RATIO={r_ratio})")
elif compressor == "visionzip":
    import types
    from token_compressor.visionzip.models.qwen3_vl import Qwen3VLModel_forward
    model.model.forward = types.MethodType(Qwen3VLModel_forward, model.model)
    r_ratio = os.getenv("R_RATIO", "0.2")
    eval_logger.success(f"[VisionZip] Successfully integrated VisionZip with Qwen3-VL. (R_RATIO={r_ratio})")
elif compressor == "holitom":
    import types
    from token_compressor.holitom.models.qwen3_vl import Qwen3VLModel_forward
    model.model.forward = types.MethodType(Qwen3VLModel_forward, model.model)
    r_ratio = os.getenv("R_RATIO", "0.15")
    tau = os.getenv("HOLITOM_T", "0.8")
    eval_logger.success(f"[HoliTom] Successfully integrated HoliTom with Qwen3-VL. (R_RATIO={r_ratio}, T={tau})")
elif compressor == "ipcv":
    import types
    from token_compressor.ipcv.models.qwen3_vl import Qwen3VLModel_forward
    model.model.forward = types.MethodType(Qwen3VLModel_forward, model.model)
    r_ratio = os.getenv("R_RATIO", "0.25")
    num_blocks = len(model.model.visual.blocks)
    ipcv_layer = os.getenv("IPCV_LAYER", str(num_blocks // 2))
    as_layers = os.getenv("IPCV_AS_LAYERS", "4")
    top_k = os.getenv("IPCV_TOP_K", "5")
    eval_logger.success(f"[IPCV] Successfully integrated IPCV with Qwen3-VL. Multi-layer pruning INSIDE ViT starting at layer {ipcv_layer} with {as_layers} AS layers. (R_RATIO={r_ratio}, TOP_K={top_k})")
elif compressor == "illava":
    import types
    from token_compressor.illava.models.qwen3_vl import Qwen3VLModel_forward
    model.model.forward = types.MethodType(Qwen3VLModel_forward, model.model)
    r_ratio = os.getenv("R_RATIO", "0.25")
    merge_ratio = os.getenv("ILLAVA_MERGE_RATIO", "0.5")
    eval_logger.success(f"[iLLaVA] Successfully integrated iLLaVA with Qwen3-VL. Merging INSIDE ViT progressively. (R_RATIO={r_ratio}, MERGE_RATIO={merge_ratio})")
elif compressor == "cdpruner":
    import types
    from token_compressor.cdpruner.models.qwen3_vl import Qwen3VLModel_forward
    model.model.forward = types.MethodType(Qwen3VLModel_forward, model.model)
    visual_tokens = os.getenv("CDPRUNER_TOKENS", "all")
    eval_logger.success(f"[CDPruner] Successfully integrated CDPruner with Qwen3-VL. (CDPRUNER_TOKENS={visual_tokens})")
elif compressor == "tome":
    import types
    from token_compressor.tome.models.qwen3_vl import Qwen3VLModel_forward
    model.model.forward = types.MethodType(Qwen3VLModel_forward, model.model)
    r_ratio = os.getenv("R_RATIO", "0.25")
    tome_r = os.getenv("TOME_R", "auto")
    apply_every = os.getenv("TOME_APPLY_EVERY", "2")
    eval_logger.success(f"[ToMe] Successfully integrated ToMe with Qwen3-VL. Merging INSIDE ViT every {apply_every} layers. (R_RATIO={r_ratio}, TOME_R={tome_r})")
elif compressor == "pooling":
    import types
    from token_compressor.pooling.models.qwen3_vl import Qwen3VLModel_forward
    model.model.forward = types.MethodType(Qwen3VLModel_forward, model.model)
    r_ratio = os.getenv("R_RATIO", "0.25")
    pool_type = os.getenv("POOLING_TYPE", "avg")
    num_blocks = len(model.model.visual.blocks)
    pool_layer = os.getenv("POOLING_LAYER", str(num_blocks // 2))
    eval_logger.success(f"[Pooling] Successfully integrated Pooling with Qwen3-VL. Pooling INSIDE ViT at layer {pool_layer}. (R_RATIO={r_ratio}, TYPE={pool_type})")
elif compressor is not None:
    eval_logger.warning(f"[Warning] Unknown COMPRESSOR value: {compressor}. Supported values: vidcom2, fastv, visionzip, holitom, ipcv, illava, tome, pooling")
    
###############################################################################


processor = AutoProcessor.from_pretrained(model_id)

# 2. 构建多图消息
# 这里我们传入两张不同的图片，并让模型进行对比
image1_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image2_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" # 为了演示方便用了同一张，实际请换成不同链接
image3_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image4_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" # 为了演示方便用了同一张，实际请换成不同链接

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image1_url,
            },
            {
                "type": "image",
                "image": image2_url,
            },
            {
                "type": "image",
                "image": image3_url,
            },
            {
                "type": "image",
                "image": image4_url,
            },
            {
                "type": "text",
                "text": "Identify the similarities and differences between these two images."
            },
        ],
    }
]

# 3. 预处理 (Preparation for inference)
# Qwen 的 processor 会自动处理多图的 token 插入位置
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# 4. 推理 (Inference)
generated_ids = model.generate(**inputs, max_new_tokens=128)

# 5. 解码输出
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text[0])