import os
import json
import ast
from datasets import Dataset, DatasetDict

def load_and_split_jsonl(path):
    splits = {
        "train": [],
        "validation": [],
        "test": []
    }

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            split = ex["split"]
            if split == "dev":
                split = "validation"
            text = ex["utterance"]
            try:
                labels = ast.literal_eval(ex["intent"])
            except:
                labels = []

            splits[split].append({
                "text": text,
                "labels": labels
            })

    return splits

def main():
    input_path = "data/raw/clinc_multi/MixCLINC150.json"
    output_base = "data/raw/clinc_multi"

    os.makedirs(output_base, exist_ok=True)
    splits = load_and_split_jsonl(input_path)

    for split_name, records in splits.items():
        ds = Dataset.from_list(records)
        save_path = os.path.join(output_base, split_name)
        ds.save_to_disk(save_path)
        print(f"{split_name} 저장 완료 → {save_path}")

if __name__ == "__main__":
    main()
