# ====================================================================
#  File: models/vision/serve.py
# ====================================================================
"""
LunaVision 모델 서빙 엔드포인트

기본: openvino_genai.VLMPipeline 로 로컬 추론 (AUTO/NPU/GPU/CPU)
--ovms grpc://HOST:9000 옵션 → OVMS inference_service.Reply 호출

실행 예시:
    # 로컬 VLMPipeline (NPU)
    python -m models.vision.serve \
        --model_dir models/vision/ov/fp16 \
        --device NPU \
        --port 8000

    # OVMS 프록시
    python -m models.vision.serve \
        --ovms grpc://127.0.0.1:9000 \
        --port 8000
"""

from __future__ import annotations

import argparse, asyncio, io, logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from utils.config import load_config_dict as load_cfg

from openvino import Tensor
from openvino_genai import VLMPipeline, GenerationConfig

# ――― Optional OVMS gRPC deps ―――
try:
    import grpc
    from model_server import model_service_pb2 as pb2, model_service_pb2_grpc as pb2_grpc
except ImportError:
    grpc = pb2 = pb2_grpc = None

# ───────────────────────────── Logger ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger("luna.serve")

# ─────────────────── YAML 설정 로드 ────────────────────────────────
CFG   = load_cfg("vision_config")
SERVE = CFG.get("serve", {})

PROMPT_TPL   = SERVE.get("prompt_tpl", "<|user|>\n<|image_1|>\nDescribe.\n<|assistant|>\n")
CHAR_LIMIT   = SERVE.get("char_limit", 300)
MAX_NEW_TOK  = SERVE.get("max_new_tokens", 32)
TEMP         = SERVE.get("temperature", 0.0)
TOP_P        = SERVE.get("top_p", 1.0)

MAX_PROMPT   = 1024
RESERVE_TOK  = 64

# ─────────────────── Global 상태 ────────────────────────────────
pipe: Optional[VLMPipeline] = None
ovms_stub = None
tokenizer_eos_id: Optional[int] = None
tokenizer = None
semaphore: Optional[asyncio.Semaphore] = None
args = None

# ─────────────────── 인자 파싱 ────────────────────────────────
def parse_args():
    global args
    parser = argparse.ArgumentParser("LunaVision FastAPI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_dir", help="OpenVINO IR directory")
    group.add_argument("--ovms", help="OVMS gRPC address e.g. grpc://127.0.0.1:9000")
    parser.add_argument("--device", default="AUTO", help="OpenVINO device")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server")
    parser.add_argument("--max_concurrent", type=int, default=1, help="Max concurrent inferences")
    args = parser.parse_args()

# ─────────────────── 초기화 ────────────────────────────────

def init_local(model_dir: str, device: str):
    global pipe, tokenizer_eos_id, tokenizer
    pipe = VLMPipeline(model_dir, device=device)
    
    tokenizer = pipe.get_tokenizer()
    tokenizer_eos_id = getattr(tokenizer, "eos_token_id", None) or tokenizer.get_eos_token_id()

    if hasattr(tokenizer, "eos_token_id") and not callable(tokenizer.eos_token_id):
        tokenizer_eos_id = tokenizer.eos_token_id
    elif hasattr(tokenizer, "get_eos_token_id"):
        tokenizer_eos_id = tokenizer.get_eos_token_id()
    elif hasattr(tokenizer, "eos_token_id") and callable(tokenizer.eos_token_id):
        tokenizer_eos_id = tokenizer.eos_token_id()
    else:
        raise AttributeError(
            "[L.U.N.A.] Tokenizer에 eos_token_id가 정의되어 있지 않습니다."
        )
        
    log.info(f"Initialized local VLMPipeline on {device}, eos_token_id={tokenizer_eos_id}")

    print(f"[L.U.N.A.] VLMPipeline 초기화 완료 → {device} / eos={tokenizer_eos_id}")
    
def init_ovms(ovms_address: str):
    global ovms_stub
    if grpc is None:
        raise ImportError(
            "[L.U.N.A.] OVMS gRPC 모듈이 설치되어 있지 않습니다. "
            "pip install optimum-openvino[ovms] 를 실행하세요."
        )
    channel = grpc.insecure_channel(ovms_address.replace("grpc://", ""))
    ovms_stub = pb2_grpc.ModelServiceStub(channel)
    
    log.info(f"Initialized OVMS gRPC client for {ovms_address}")
    

# ----------
# Utility Functions
# ----------

def pil_to_tensor(img: Image.Image) -> Tensor:
    return Tensor(np.asarray(img, dtype=np.uint8)[None])

def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def ovms_generate(img: Image.Image, prompt: str) -> str:
    """OVMS (gRPC) 추론 함수"""
    if ovms_stub is None:
        raise RuntimeError("OVMS gRPC stub이 초기화되지 않았습니다.")
    
    req = pb2.PredictRequest()
    req.model_spec.name = "lunavision"
    req.inputs["prompt"].string_val.append(prompt)
    req.inputs["image"].bytes_val.append(pil_to_bytes(img))
    
    resp = ovms_stub.Predict(req, timeout=30.0)
    return resp.outputs["text"].string_val[0]

def local_generate(
    img: Image.Image,
    prompt: str,
    max_tokens: int = 48
) -> str:
    """로컬 VLMPipeline 추론 함수"""
    gcfg = GenerationConfig(
        max_new_tokens=max_tokens,
        eos_token_id=tokenizer_eos_id,
        pad_token_id=tokenizer_eos_id,
        do_sample=False,
        temperature=0.0,
    )
    
    return pipe.generate(
        prompt,
        images=[pil_to_tensor(img)],
        generation_config=gcfg,
    )

# API
@asynccontextmanager
async def lifespan(app: FastAPI):
    parse_args()
    if args.model_dir:
        init_local(args.model_dir, args.device)
    else:
        init_ovms(args.ovms)
    global semaphore
    semaphore = asyncio.Semaphore(args.max_concurrent)
    log.info("Startup complete. Ready to handle requests.")
    yield

app = FastAPI(title="LunaVision Model Serving API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model_version": app.version}

# ----------
# 엔드포인트
# ----------

@app.post("/v1/vision")
async def vision_qa(
    file: UploadFile = File(...),
):
    data = await file.read()
    
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        log.error("Failed to load image", exc_info=e)
        raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {str(e)}")
    
    prompt = PROMPT_TPL
    
    try:
        async with semaphore:
            if pipe:
                answer = await asyncio.to_thread(local_generate, img, prompt)
            else:
                answer = await asyncio.to_thread(ovms_generate, img, prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {e}")
    
    if not isinstance(answer, str):
        answer = getattr(answer, "text", str(answer))
    
    if '<|user|>' in answer:
        answer = answer.split('<|user|>')[0].rstrip()
        
    if len(answer) > CHAR_LIMIT:
        answer = answer[:CHAR_LIMIT].rsplit(" ", 1)[0] + " ..."

    return JSONResponse({"answer": answer})

def main():
    parse_args()
    import uvicorn
    uvicorn.run("models.vision.serve:app", host="0.0.0.0", port=args.port, log_level="info")

if __name__ == "__main__":
    main()
