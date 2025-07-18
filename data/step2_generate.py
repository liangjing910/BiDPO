import io

from PIL.Image import Image
from openai import OpenAI
import base64
import json
import os
from tqdm import tqdm
import argparse

# 工具函数
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_modification_example(contrast_type):
    if contrast_type == "position_flip":
        return "e.g., swap left and right positions, or move from the front to the back."
    elif contrast_type == "count_modification":
        return "e.g., increase or decrease the number of objects in the scene."
    elif contrast_type == "attribute_modification":
        return "e.g., change the color, size, or other visible attribute of the target object."
    elif contrast_type == "object_replacement":
        return "e.g., replace one object with another of a different category."
    else:
        return ""

def build_contrast_prompt(sample):
    example = get_modification_example(sample['contrast_type'])
    return f"""You are an assistant for constructing visual contrastive samples for multimodal instruction tuning.

Your task:
    1. Based on the original image and the question, generate a visually modified version of the image according to the specified contrast_type (here: {sample['contrast_type']}).
    2. The modification should be minimal but meaningful and correspond to the focus of the question ({example}).
    3. Generate a new image, and write the new answer according to this modified image.

Original Image: (see below)
Question: “{sample['question']}”
Contrast Type: “{sample['contrast_type']}”
"""

def main():
    # 路径配置
    input_json = "data/llava_labeled.json"

    # openai client 配置
    client = OpenAI(
        base_url='your_base_url',
        api_key='your_api_key'
    )


    with open(input_json, "r", encoding="utf-8") as f:
        samples = json.load(f)


    # argparse 支持 --start_idx --end_idx
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0, help="起始索引（包含）")
    parser.add_argument("--end_idx", type=int, default=-1, help="结束索引（不包含）")
    args = parser.parse_args()

    with open(input_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    start_idx = args.start_idx
    end_idx = args.end_idx
    if end_idx == -1:
        print("end_idx 不能为 -1，请显式指定结束索引。")
        exit(1)
    if start_idx >= end_idx:
        print("起始索引必须小于结束索引")
        exit(1)
    print(f"处理区间：start_idx={start_idx}, end_idx={end_idx}")

    output_json = f"data/llava_contrast_{start_idx}_{end_idx}.json"

    samples_to_process = [s for s in samples[start_idx:end_idx] if s.get("contrast_type", "unknown") != "unknown"]
    print(f"实际将生成 {len(samples_to_process)} 条（已跳过unknown）")

    image_root = "/Volumes/Lexar/datasets/data/coco/train2017"

    results = []
    for sample in tqdm(samples_to_process, desc="生成中"):
        # 1. 构建图片路径
        image_path = sample["image"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(image_root, image_path)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue


        prompt = build_contrast_prompt(sample)

        # 3. 调用图片编辑API生成对比图片
        with open(image_path, "rb") as image_file:
            result = client.images.edit(
                model="gpt-image-1",
                image=image_file,
                quality="low",
                prompt=prompt
            )
        image_base64 = result.data[0].b64_json


        gpt4o_prompt = f"""Given the following question and image, provide a answer according to the visual content in the image.

                    Question: “{sample['question']}”
                    Instructions:
                    1.Answer in one sentence.
                    2.Keep the format and tone consistent with:“{sample['answer']}”
                    """

        gpt4o_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": gpt4o_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ]
        )
        new_answer = gpt4o_response.choices[0].message.content.strip()


        contrast_image_name = f"contrast_{sample['id']}.jpg"
        contrast_image_path = os.path.join(os.path.dirname(image_path), contrast_image_name)
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.save(contrast_image_path, format="JPEG")


        results.append({
            "system_prompt": "",
            "chosen_images": [sample["image"]],
            "rejected_images": [contrast_image_name],
            "instruction_prompt": sample["question"],
            "chosen_response": sample["answer"],
            "rejected_response": new_answer
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"全部处理完成，已保存到 {output_json}")

if __name__ == "__main__":
    main()