# ====================================================================
#  File: models/vision/export_openvino.py
# ====================================================================
"""
LunaVision OpenVINO 모델 변환 스크립트

• fp16 모델(LoRA 적용 포함)을 OpenVINO IR(./fp16)로 변환
• --precision int4 옵션을 주면 POT(Post-Training Optimization) INT4 양자화도 수행

실행 예시:
    1) fp16 IR 변환
        python -m models.vision.export_openvino \
            --model_dir outputs/vision/fp16_lora \
            --out_dir  models/vision/ov/fp16 \
            --precision fp16
    2) int4 양자화
        python -m models.vision.export_openvino \
            --model_dir outputs/vision/fp16_lora \
            --out_dir  models/vision/ov/int4 \
            --precision int4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd: list[str]) -> None:
    """subprocess 호출 및 오류 처리"""
    print("⚙️ ", " ".join(cmd))
    result = subprocess.run(cmd, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.exit(f"[L.U.N.A.] 명령어 실행 실패 (exit={result.returncode})")

def export_openvino(model_dir: str, output_dir, precision: str) -> None:
    """ OpenVINO 모델 변환 함수

    Args:
        model_dir (str): 변환할 모델 디렉토리
        output_dir (str): 변환된 OpenVINO 모델 저장 디렉토리
        precision (str): 'fp16' 또는 'int4' 중 하나
    """
    model_dir = Path(model_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "optimum-cli",
        "export",
        "openvino",
        "--model", str(model_dir),
        "--task", "vlm-generation",
        "--output", str(output_dir),
    ]
    
    if precision.lower() == "int4":
        cmd += ["--quantization", "int4"]
    elif precision.lower() != "fp16":
        sys.exit("[L.U.N.A.] precision은 'fp16' 또는 'int4'만 지원합니다.")
        
    run_cmd(cmd)
    print(f"[L.U.N.A.] OpenVINO 모델이 {output_dir}에 성공적으로 저장되었습니다.")
    
# ----------
# CLI Entry Point
# ----------

def main():
    ap = argparse.ArgumentParser(description="LunaVision OpenVINO Exporter")
    ap.add_argument("--model_dir", required=True, help="LoRA 적용 fp16 모델 디렉터리")
    ap.add_argument("--output_dir", required=True, help="내보낼 IR 저장 경로")
    ap.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "int4"],
        help="fp16 또는 int4 (양자화) 중 선택 (기본값: fp16)",
    )
    args = ap.parse_args()

    export_openvino(args.model_dir, args.output_dir, args.precision)


if __name__ == "__main__":
    main()