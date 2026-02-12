import torch
import os
from diffusers import FluxPipeline, FluxTransformer2DModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# --------------------------
# 固定配置
# --------------------------
HF_TOKEN = "your_huggingface_token_here"
CUSTOM_REPO = "xixiaogua/Qipao_XXG_Flux_BF16_V1"
MODEL_FILE = "Qipao_XXG_Flux_BF16_V1.safetensors"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
device = "cuda"
dtype = torch.bfloat16

# 保存路径（自动创建，避免图片丢失）
save_dir = "demo_images"
os.makedirs(save_dir, exist_ok=True)

# --------------------------
# 1. 下载你的旗袍权重
# --------------------------
print("Loading Qipao_XXG_Flux_BF16_v1 model...")
weight_path = hf_hub_download(
    repo_id=CUSTOM_REPO,
    filename=MODEL_FILE,
    token=HF_TOKEN,
    resume_download=True
)

# --------------------------
# 2. 直接用你的权重初始化 Transformer（核心！）
# --------------------------
transformer = FluxTransformer2DModel.from_single_file(
    weight_path,
    torch_dtype=dtype
)

# --------------------------
# 3. 加载 FLUX.1-dev 管道（和你权重同架构）
# --------------------------
print("Loading FLUX dev base...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,  # 直接把你的模型塞进去
    torch_dtype=dtype,
    token=HF_TOKEN
)

# --------------------------
# 4. 移到 GPU，不开任何卸载（保证生效）
# --------------------------
pipe.to(device)

# --------------------------
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

# ===================== 批量推理 =====================
print(f"\n开始批量生成 {len(prompt_list)} 张图片（基于Qipao_XXG_Flux_BF16_V1模型）...")
for idx, (task_name, prompt, negative_prompt) in enumerate(prompt_list):
    print(f"\n生成第 {idx + 1}/{len(prompt_list)} 张：{task_name}")

    # 执行推理
    image  = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=20,    # ✅ 修正：冒号→等号
        guidance_scale=3.5,       # ✅ 修正：冒号→等号
        height=1024,               # ✅ 修正：冒号→等号
        width=768,                # ✅ 修正：冒号→等号
        generator=torch.Generator(device=device).manual_seed(666),
        max_sequence_length=512,  # 补充：FLUX.1-dev必需参数
    ).images[0]

    # 保存图片
    save_path = os.path.join(save_dir, f"{task_name}.png")
    image.save(save_path)
    print(f"图片保存成功：{save_path}")

print(f"\n所有图片生成完成！保存目录：{os.path.abspath(save_dir)}")