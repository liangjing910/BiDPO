# step1_label.py
import json
import re
from tqdm import tqdm
from collections import Counter

def infer_contrast_type(question: str) -> str:
    q = question.lower()
    if re.search(r'(how many|number of|are there|total number|how much|amount of|any .* in the image|.*\bany\b.*\bin the\b.*)', q):
        return "count_modification"
    elif re.search(r'(color|colours|colors|wearing|clothing|hair|type|types|kind|kinds|style|styles|material|pattern|texture|brand|size|sizes|shape|shapes|current|status)', q):
        return "attribute_modification"
    elif re.search(r'(left|right|next to|beside|between|behind|in front of|above|below|top|bottom|middle|near|far from|close to|position|where is|located)', q):
        return "position_flip"
    elif re.search(r'(what|which|feature|describe|shown|who|whose|see|doing|else can be seen|can you describe|what is .+ doing|what kind|what type|what feature)', q):
        return "object_replacement"
    else:
        return "unknown"

def label_dataset(input_json, output_json):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = []
    unknown_count = 0  # 新增
    for sample in tqdm(data, desc="打标签中"):
        qa_pairs = []
        convs = sample['conversations']
        for i in range(len(convs) - 1):
            if convs[i]['from'] == 'human' and convs[i+1]['from'] == 'gpt':
                question = convs[i]['value'].replace('<image>', '').strip()
                answer = convs[i+1]['value'].strip()
                contrast_type = infer_contrast_type(question)
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "contrast_type": contrast_type
                })
        qa_selected = next((qa for qa in qa_pairs if qa['contrast_type'] != 'unknown'), qa_pairs[0] if qa_pairs else None)
        if qa_selected:
            if qa_selected["contrast_type"] == "unknown":
                unknown_count += 1  # 统计unknown
            results.append({
                "id": sample["id"],
                "question": qa_selected["question"],
                "image": sample["image"],
                "answer": qa_selected["answer"],
                "contrast_type": qa_selected["contrast_type"]
            })
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"已处理原始样本数: {len(data)}")
    print(f"成功打标签的样本数: {len(results)}，已保存到 {output_json}")
    print(f"其中 contrast_type=unknown 的样本数: {unknown_count}，占比 {unknown_count / len(results):.2%}")
    type_counter = Counter([item["contrast_type"] for item in results])
    print("各 contrast_type 样本数分布：")
    for k, v in type_counter.items():
        print(f"{k}: {v}，占比 {v / len(results):.2%}")

if __name__ == "__main__":
    label_dataset("llava_instruct_150k.json", "llava_labeled.json")