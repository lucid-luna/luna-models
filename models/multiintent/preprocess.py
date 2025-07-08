# model/intent/preprocess.py

import os
import json
import ast
from datasets import Dataset, DatasetDict
from utils.config import load_config

def main():
    """
    Multi-Intent Classification 전처리 스크립트
    """
    config = load_config("multiintent_config")
    
    json_path = os.path.join(config.data.raw_dir, "MixCLINC150.json")    
    save_dir = config.data.processed_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. JSONL 로드
    all_data = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line.strip())
            # 문자열 형태의 리스트 → 실제 리스트로 변환
            ex["intent"] = ast.literal_eval(ex["intent"])
            all_data.append(ex)

    # 2. 전체 라벨 수집
    all_labels = sorted(set(label for ex in all_data for label in ex["intent"]))
    label_to_idx = {label: i for i, label in enumerate(all_labels)}

    # 3. 멀티핫 벡터 생성 함수
    def encode_multi_hot(label_list):
        vec = [0] * len(label_to_idx)
        for l in label_list:
            vec[label_to_idx[l]] = 1
        return vec

    # 4. split별 정리
    split_data = {"train": [], "validation": [], "test": []}
    for ex in all_data:
        raw_split = ex["split"]
        split = "validation" if raw_split == "dev" else raw_split
        if split not in split_data:
            continue
        split_data[split].append({
            "text": ex["utterance"],
            "labels": encode_multi_hot(ex["intent"])
        })

    # 5. DatasetDict 생성 및 저장
    dataset_dict = DatasetDict({
        split: Dataset.from_list(data) for split, data in split_data.items()
    })

    for split in ["train", "validation", "test"]:
        if not split_data[split]:
            print(f"⚠️ {split} split이 비어 있어 저장 생략")
            continue
        out_path = os.path.join(save_dir, split)
        dataset_dict[split].save_to_disk(out_path)
        print(f"✅ {split} 저장 완료: {out_path}")

    # 6. label_list 저장
    label_list_path = os.path.join(save_dir, "label_list.json")
    with open(label_list_path, "w", encoding="utf-8") as f:
        json.dump(all_labels, f, ensure_ascii=False, indent=2)

    print(f"🎉 전체 완료: {save_dir} (라벨 수: {len(all_labels)})")

if __name__ == "__main__":
    main()