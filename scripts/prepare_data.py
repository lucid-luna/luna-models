# scripts/prepare_data.py
import os
from datasets import load_dataset

def main():
    base_dir = os.path.join("data", "raw", "go_emotions_simplified")
    for split in ("train", "validation", "test"):
        os.makedirs(os.path.join(base_dir, split), exist_ok=True)

    ds = load_dataset("google-research-datasets/go_emotions", "simplified")

    ds["train"].save_to_disk(os.path.join(base_dir, "train"))
    ds["validation"].save_to_disk(os.path.join(base_dir, "validation"))
    ds["test"].save_to_disk(os.path.join(base_dir, "test"))

    print("✅ 데이터셋이 ", base_dir, "에 저장되었습니다.")

if __name__ == "__main__":
    main()
