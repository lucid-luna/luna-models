# ====================================================================
#  File: models/multiintent/preprocess.py
# ====================================================================
"""
LunaMultiIntent 데이터 전처리 스크립트

실행 예시:
    python -m models.multiintent.preprocess
"""

import os
import json
import ast
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from utils.config import load_config

def main():
    config = load_config("multiintent_config")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=True)
    
    raw_path = os.path.join(config.data.raw_dir, "MixCLINC150.json")
    save_dir = config.data.processed_dir
    os.makedirs(save_dir, exist_ok=True)
    
    all_data = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line.strip())
            ex['intent'] = ast.literal_eval(ex['intent'])
            all_data.append(ex)

    all_labels = sorted({label for ex in all_data for label in ex['intent']})
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    def encode_multi_hot(intents):
        vec = [0] * len(label_to_idx)
        for intent in intents:
            vec[label_to_idx[intent]] = 1
        return vec

    split_data = {'train': [], 'validation': [], 'test': []}
    for ex in all_data:
        raw_split = 'validation' if ex.get('split') == 'dev' else ex.get('split')
        if raw_split not in split_data:
            continue

        enc = tokenizer(
            ex['utterance'],
            truncation=True,
            padding='max_length',
            max_length=config.max_length
        )
        split_data[raw_split].append({
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels': encode_multi_hot(ex['intent'])
        })

    dataset_dict = DatasetDict({
        split: Dataset.from_list(data)
        for split, data in split_data.items() if data
    })
    for split, ds in dataset_dict.items():
        out_path = os.path.join(save_dir, split)
        os.makedirs(out_path, exist_ok=True)
        ds.save_to_disk(out_path)
        print(f"[L.U.N.A] {split} split 저장 완료: {out_path}")

    label_list_path = os.path.join(save_dir, 'label_list.json')
    with open(label_list_path, 'w', encoding='utf-8') as f:
        json.dump(all_labels, f, ensure_ascii=False, indent=2)
    print(f"[L.U.N.A] 전체 전처리 완료. 레이블 수: {len(all_labels)}, 저장 경로: {save_dir}")

if __name__ == "__main__":
    main()