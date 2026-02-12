import torch
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image
import os
from huggingface_hub import login, hf_hub_download
from safetensors.torch import load_file

# ===================== 1. Hugging Face 认证（核心修复） =====================
# 替换为你的 HF Token（Read 权限即可，从 https://huggingface.co/settings/tokens 获取）
HF_TOKEN = "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"
# 登录认证（解决 gated repo 访问限制）
login(token=HF_TOKEN, add_to_git_credential=False)

# 国内用户配置镜像加速（可选，避免下载慢）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ===================== 2. 基础配置 =====================
# 自定义模型的 HF 仓库信息
CUSTOM_REPO_ID = "xixiaogua/Qipao_XXG_Flux_BF16_V1"
CUSTOM_FILE_NAME = "Qipao_XXG_Flux_BF16_V1.safetensors"

# 设备与精度配置
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    device = "cpu"
    dtype = torch.float32


# 保存路径（自动创建，避免图片丢失）
save_dir = "demo_images"
os.makedirs(save_dir, exist_ok=True)

# ===================== 2. 定义多组提示词（核心修改） =====================
# 格式：(任务名, 正向提示词, 反向提示词)
prompt_list = [
    (
        "task1",
        "Photograph of an East Asian woman with long, straight black hair, wearing a tight, colorful cheongsam dress with floral patterns in red, orange, and blue. The dress has three-quarter sleeves, a high collar, and a slit on the left side revealing her leg.",
        "(worst quality:2), (low quality:2), jpeg artifacts, blurry, badhandv4, easynegative, missing fingers, extra fingers, bad anatomy, plastic skin, waxy skin, greasy skin, cartoon, anime, 3d render, cgi, overexposed, underexposed, cross-eye, cloned face"
    ),
    (
        "task2",
        "Photograph of a young Asian woman with fair skin and black hair in a traditional Chinese qipao dress, seated on a wooden chair with a blue floral cloth. The dress is white with black trim and delicate floral embroidery. She has a slender build with medium-sized breasts. The woman is smiling gently, with red lipstick and subtle makeup, exuding a calm and serene presence.",
        "(worst quality:2), (low quality:2), jpeg artifacts, blurry, badhandv4, easynegative, missing fingers, extra fingers, bad anatomy, plastic skin, waxy skin, greasy skin, cartoon, anime, 3d render, cgi, overexposed, underexposed, cross-eye, cloned face"
    ),
    (
        "task3",
        "Photograph of an Asian woman with fair skin and long black hair, wearing a silver, high-neck, sleeveless cheongsam dress with a keyhole cutout, and white, sheer, elbow-length gloves.",
        "(worst quality:2), (low quality:2), jpeg artifacts, blurry, badhandv4, easynegative, missing fingers, extra fingers, bad anatomy, plastic skin, waxy skin, greasy skin, cartoon, anime, 3d render, cgi, overexposed, underexposed, cross-eye, cloned face"
    ),
    (
        "task4",
        "Photograph of a young woman with a fair skin tone and short, straight brown hair with bangs. She is wearing a tight, sleeveless, high-neck Chinese cheongsam-style dress in white with blue floral patterns, which accentuates her slim, petite figure and plump chest. Her legs are crossed at the ankles, and she is wearing beige, pointed-toe flats with a subtle bow detail.",
        "(worst quality:2), (low quality:2), jpeg artifacts, blurry, badhandv4, easynegative, missing fingers, extra fingers, bad anatomy, plastic skin, waxy skin, greasy skin, cartoon, anime, 3d render, cgi, overexposed, underexposed, cross-eye, cloned face"
    ),
    (
        "task5",
        "Photograph of an East Asian woman with long, dark brown hair standing on a wooden bridge over a calm pond in a traditional chinese garden. She is wearing a form-fitting, beige, ribbed, short-sleeved cheongsam with blue decorative stitching along the sides and neckline. Her figure is hourglass-shaped, with plump chest and a slim waist. She has fair skin and a slight smile, with minimal makeup accentuating her natural beauty. Her right hand rests on her hip, and she gazes directly at the camera.",
        "(worst quality:2), (low quality:2), jpeg artifacts, blurry, badhandv4, easynegative, missing fingers, extra fingers, bad anatomy, plastic skin, waxy skin, greasy skin, cartoon, anime, 3d render, cgi, overexposed, underexposed, cross-eye, cloned face"
    ),
    (
        "task6",
        "Photograph of a young Asian woman with pale skin and black hair in a bob cut, wearing a sleeveless, form-fitting, green and white floral Chinese cheongsam dress. She has visible tattoos on her left arm and right upper arm. She is leaning against a stone railing with her right arm resting on it, looking directly at the camera with a neutral expression. The woman's dress has a high collar and green trim, and her makeup is subtle, enhancing her natural features.",
        "(worst quality:2), (low quality:2), jpeg artifacts, blurry, badhandv4, easynegative, missing fingers, extra fingers, bad anatomy, plastic skin, waxy skin, greasy skin, cartoon, anime, 3d render, cgi, overexposed, underexposed, cross-eye, cloned face"
    ),
    (
        "task7",
        "Photograph of an East Asian woman standing in a lush, green garden. She has fair skin, long black hair styled in a side braid, and wears a light green, traditional Chinese cheongsam dress with intricate patterns and gold buttons. She holds a decorative, white hand fan with floral designs in her right hand and a large green leaf in her left. The background features various potted plants, including large leaves and shrubs, with a blurred, natural garden setting. The lighting is soft, highlighting the greenery and her serene expression. The overall scene exudes a tranquil, traditional Asian aesthetic.",
        "(worst quality:2), (low quality:2), jpeg artifacts, blurry, badhandv4, easynegative, missing fingers, extra fingers, bad anatomy, plastic skin, waxy skin, greasy skin, cartoon, anime, 3d render, cgi, overexposed, underexposed, cross-eye, cloned face"
    ),
    (
        "task8",
        "Photograph of an East Asian woman standing in a lush garden. She has long, straight black hair partially tied to one side, and fair skin. She is wearing a light green, short-sleeved cheongsam with intricate floral patterns and a high collar adorned with pearls. She holds a small, round, beige fan in her right hand, raised slightly above her head. Her left hand gently touches her hair. The background features dense green foliage and blooming white flowers on a tree, with sunlight filtering through the leaves. The woman's expression is calm and composed, and she gazes directly at the camera. The overall scene evokes a sense of tranquility and elegance.",
        "(worst quality:2), (low quality:2), jpeg artifacts, blurry, badhandv4, easynegative, missing fingers, extra fingers, bad anatomy, plastic skin, waxy skin, greasy skin, cartoon, anime, 3d render, cgi, overexposed, underexposed, cross-eye, cloned face"
    )
]

# ===================== 3. 加载官方Flux管道 + 下载并替换HF自定义权重 =====================
# ===================== 3. 加载官方 FLUX 管道（已认证，可访问） =====================
print("加载官方 FLUX.1-schnell 基础管道（已认证）...")
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=dtype,
    device_map=None,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    token=HF_TOKEN  # 显式传入 token，确保认证生效
)

# ===================== 4. 下载并加载自定义 Qipao 权重 =====================
print(f"从 HF 仓库下载自定义权重 {CUSTOM_FILE_NAME}...")
custom_weight_path = hf_hub_download(
    repo_id=CUSTOM_REPO_ID,
    filename=CUSTOM_FILE_NAME,
    token=HF_TOKEN  # 确保访问自定义仓库（若有访问限制）
)

# 用 safetensors 安全加载单文件权重
weight_dict = safe_load_file(custom_weight_path, device="cpu")
# 注入到 FLUX 的 transformer 核心组件
pipe.transformer.load_state_dict(weight_dict, strict=False)

# 移到目标设备
pipe = pipe.to(device)

# 显存优化（GPU 环境）
if device == "cuda":
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing("max")

# ===================== 4. 批量推理参数 =====================
common_params = {
    "num_inference_steps": 4,          # schnell版固定4步
    "guidance_scale": 0.0,             # Flux.1无需引导，固定0
    "height": 768,                    # 图片尺寸（64倍数）
    "width": 1024,
    "max_sequence_length": 512,
    "generator": torch.Generator(device=device).manual_seed(42)  # 固定种子可复现
}

# ===================== 5. 批量推理 =====================
print(f"\n开始批量生成 {len(prompt_list)} 张图片（基于Qipao_XXG_Flux_BF16_V1模型）...")
for idx, (task_name, prompt, negative_prompt) in enumerate(prompt_list):
    print(f"\n生成第 {idx + 1}/{len(prompt_list)} 张：{task_name}")

    # 执行推理
    image: Image.Image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        **common_params
    ).images[0]

    # 保存图片
    save_path = os.path.join(save_dir, f"{task_name}.png")
    image.save(save_path)
    print(f"图片保存成功：{save_path}")

print(f"\n所有图片生成完成！保存目录：{os.path.abspath(save_dir)}")