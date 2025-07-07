<p align="center">
  <img src="https://github.com/lucid-luna/.github/blob/main/profile/assets/Logo.png" width="180" alt="L.U.N.A. Logo"/>
</p>

<h1 align="center">Luna Models</h1>
<h3 align="center">💫 Core Neural Architectures for the L.U.N.A. System</h3>

<p align="center">
  <b>PyTorch 기반의 시각 및 멀티모달 모델을 설계, 훈련, 최적화합니다.</b>
</p>

---

## 🧠 소개

`luna-models`는 [L.U.N.A. (Lucid Undulation Neural Aether)](https://github.com/lucid-luna) 프로젝트의 인공지능 비서 시스템에서 사용되는 모든 **맞춤형 신경망 모델**을 개발하고 관리하는 리포지토리입니다.

실시간 상호작용, 감성 인식, 멀티모달 입력 처리를 목표로 한 작고 빠른 모델들을 중심으로 구성되어 있습니다.

---

## 📦 구성 중인 모델

| 모델명 | 설명 | 상태 |
|--------|------|------|
| `LunaVision` | 이미지 기반 상황 인식 태거 (NPU 대응) | 🔧 개발 중 |
| `LunaMM` | 이미지 → 문장 생성 (BLIP-lite 구조) | ⏳ 계획 |
| `LunaEmbed` | 이미지/텍스트 공통 임베딩 매핑기 | ⏳ 계획 |
| `LunaTuner` | 캐릭터 LLM 튜닝용 LoRA Adapter | ⏳ 계획 |

---

## 🔨 기본 구조

luna-models/
├── src/
│ ├── models/ # 모델 정의 (e.g., LunaVision)
│ ├── training/ # 학습 루프, 데이터셋, 평가
│ └── utils/ # 전처리 및 기타 유틸
├── configs/ # 모델별 설정 파일
├── export/ # ONNX / INT4 변환 스크립트
├── quantize/ # 양자화 관련 코드
├── train.py # 메인 학습 진입점
└── README.md


---

## 🖼️ 예시: LunaVision

`LunaVision`은 실시간 이미지 입력을 받아,  
현재 장면을 설명하는 **다중 태그**를 출력하는 경량 모델입니다.

- ✅ PyTorch + timm 기반 MobileViT 구조
- ✅ Multi-label classification (`sigmoid + BCE`)
- ✅ 추론 속도 최적화 (NPU 양자화 대응)
- ✅ 입력: RGB 이미지 (224x224)
- ✅ 출력: `[person, smile, cat, window, sunlight, ...]`

---

## ⚙️ 향후 개발 계획

- [ ] LunaVision 모델 구성 및 학습 스크립트 작성
- [ ] 데이터셋 수집 및 태깅 체계 정립
- [ ] ONNX → INT4 양자화 변환 구조 구현
- [ ] LunaMM (멀티모달 Captioner) 아키텍처 설계
- [ ] Trainer 유틸 통합 및 CLI 구성

---

## 🪪 라이선스

Apache License 2.0 © lucid-luna

---

## ✨ 프로젝트 전체 보기

👉 [lucid-luna/luna-core](https://github.com/lucid-luna/luna-core)  
👉 [lucid-luna/luna-client](https://github.com/lucid-luna/luna-client)  
👉 [lucid-luna/luna-plugins](https://github.com/lucid-luna/luna-plugins)
