#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_inference.py

1) HTTP endpoint 벤치마크
2) Local VLMPipeline 벤치마크

Usage examples:
    # HTTP + Local benchmarks
    python test_inference.py \
      --model-dir "../../ModelTest/Phi-3.5-vision-instruct/fp16" \
      --image-path test_screenshot.jpg \
      --url http://localhost:8000/v1/vision \
      --device NPU

    # Only HTTP benchmark
    python test_inference.py --skip-local \
      --model-dir "../../ModelTest/Phi-3.5-vision-instruct/fp16" \
      --image-path test_screenshot.jpg \
      --url http://localhost:8000/v1/vision

    # Only Local benchmark
    python test_inference.py --skip-http \
      --model-dir "../../ModelTest/Phi-3.5-vision-instruct/fp16" \
      --image-path test_screenshot.jpg \
      --device NPU
"""

import argparse
import time
import requests
from PIL import Image

def benchmark_http(image_path, url, iterations, warmup):
    with open(image_path, "rb") as f:
        files = {"file": f}
        # Warm-up
        for _ in range(warmup):
            requests.post(url, files=files)
        f.seek(0)

        times = []
        for _ in range(iterations):
            start = time.time()
            _ = requests.post(url, files=files)
            elapsed = time.time() - start
            times.append(elapsed)
            f.seek(0)
        print(f"[HTTP] {iterations} runs: avg={sum(times)/len(times):.3f}s, min={min(times):.3f}s, max={max(times):.3f}s")


def benchmark_local(image_path, model_dir, device, iterations, warmup):
    import numpy as np
    from openvino import Tensor
    from openvino_genai import VLMPipeline, GenerationConfig
    from utils.config import load_config_dict

    # 설정 불러오기
    cfg = load_config_dict("vision_config")
    serve_cfg = cfg.get("serve", {})
    prompt_tpl = serve_cfg.get(
        "prompt_tpl",
        "<|user|>\n<|image_1|>\nDescribe.\n<|assistant|>\n"
    )
    max_new_tokens = serve_cfg.get("max_new_tokens", 32)

    # 모델 초기화
    pipe = VLMPipeline(model_dir, device=device)
    tokenizer = pipe.get_tokenizer()
    eos_id = getattr(tokenizer, "eos_token_id", tokenizer.get_eos_token_id())

    # 이미지 로드 및 Tensor 변환
    img = Image.open(image_path).convert("RGB")
    img_np = np.asarray(img, dtype=np.uint8)[None]      # shape [1, H, W, C]
    img_tensor = Tensor(img_np)

    # GenerationConfig 준비
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        do_sample=False,
        temperature=0.0
    )

    # 워밍업
    for _ in range(warmup):
        pipe.generate(prompt_tpl, images=[img_tensor], generation_config=gen_cfg)

    # 측정
    times = []
    for _ in range(iterations):
        start = time.time()
        pipe.generate(prompt_tpl, images=[img_tensor], generation_config=gen_cfg)
        times.append(time.time() - start)

    print(f"[Local] Device={device}, {iterations} runs: avg={sum(times)/len(times):.3f}s, min={min(times):.3f}s, max={max(times):.3f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark HTTP & Local VLMPipeline")
    parser.add_argument("--image-path", default="test_screenshot.jpg",
                        help="Path to the screenshot image")
    parser.add_argument("--url", default="http://localhost:8000/v1/vision",
                        help="HTTP endpoint URL")
    parser.add_argument("--model-dir", required=True,
                        help="Local OpenVINO IR directory (same as serve.py --model_dir)")
    parser.add_argument("--device", default="NPU",
                        help="Device for local pipeline (e.g., NPU, GPU, CPU)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Number of warm-up runs")
    parser.add_argument("--skip-http", action="store_true",
                        help="Skip HTTP benchmark")
    parser.add_argument("--skip-local", action="store_true",
                        help="Skip local pipeline benchmark")
    args = parser.parse_args()

    if not args.skip_http:
        print("== HTTP Endpoint Benchmark ==")
        benchmark_http(args.image_path, args.url, args.iterations, args.warmup)
        print()

    if not args.skip_local:
        print("== Local Pipeline Benchmark ==")
        benchmark_local(args.image_path, args.model_dir, args.device, args.iterations, args.warmup)


if __name__ == "__main__":
    main()
