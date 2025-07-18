import json
import os
from tqdm import tqdm
import torch
from PIL import Image
import clip
from transformers import AutoImageProcessor, Dinov2Model
from sklearn.metrics.pairwise import cosine_similarity

def compute_clip_similarity(img1_path, img2_path, device, model, preprocess):
    image1 = preprocess(Image.open(img1_path).convert("RGB")).unsqueeze(0).to(device)
    image2 = preprocess(Image.open(img2_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb1 = model.encode_image(image1).cpu().numpy()
        emb2 = model.encode_image(image2).cpu().numpy()
    return float(cosine_similarity(emb1, emb2)[0][0])

def compute_dino_similarity(img1_path, img2_path, device, processor, model):
    image1 = Image.open(img1_path).convert("RGB")
    image2 = Image.open(img2_path).convert("RGB")
    inputs1 = processor(images=image1, return_tensors="pt").to(device)
    inputs2 = processor(images=image2, return_tensors="pt").to(device)
    with torch.no_grad():
        emb1 = model(**inputs1).last_hidden_state.mean(dim=1).cpu().numpy()
        emb2 = model(**inputs2).last_hidden_state.mean(dim=1).cpu().numpy()
    return float(cosine_similarity(emb1, emb2)[0][0])

def main():
    # 配置
    input_json = "llava_contrast_0_10.json"  # 按你的生成批次修改
    output_json = input_json.replace(".json", "_filtered.json")
    image_root = "/mnt/data/s-vco/image"
    contrast_root = image_root  # 和step2一致

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("加载CLIP模型...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    print("加载DINOv2模型...")
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device)

    with open(input_json, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    filtered = []
    for item in tqdm(pairs, desc="筛选中"):
        img1_path = os.path.join(image_root, item["chosen_images"][0])
        img2_path = os.path.join(contrast_root, item["rejected_images"][0])
        if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
            print(f"图片缺失：{img1_path} 或 {img2_path}")
            continue
        try:
            clip_sim = compute_clip_similarity(img1_path, img2_path, device, clip_model, clip_preprocess)
            dino_sim = compute_dino_similarity(img1_path, img2_path, device, dino_processor, dino_model)
        except Exception as e:
            print(f"特征提取失败：{e}")
            continue

        # 筛选
        if 0.3 < clip_sim < 0.9:
            if item["chosen_response"] == item["rejected_response"]:
                print("丢弃内容完全相同的答案")
                continue
            filtered.append(item)
        else:
            print(f"丢弃 pair {item['chosen_images'][0]} <-> {item['rejected_images'][0]} | CLIP={clip_sim:.2f}, DINO={dino_sim:.2f}")

    print(f"最终筛选保留 {len(filtered)} 条 / 原始 {len(pairs)} 条")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"已保存到 {output_json}")

if __name__ == "__main__":
    main()