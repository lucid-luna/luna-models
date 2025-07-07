<p align="center">
  <img src="https://github.com/lucid-luna/.github/blob/main/profile/assets/Logo.png" width="180" alt="L.U.N.A. Logo"/>
</p>

<h1 align="center">🌙 Luna Models</h1>
<p align="center">
  <b>PyTorch 기반의 멀티모달 인공지능 모델 설계 & 추론 최적화</b><br/>
  <i>for real-time, low-latency AI companions ✨</i>
</p>

---

## 🧠 소개

`luna-models`는 [L.U.N.A. (Lucid Undulation Neural Aether)](https://github.com/lucid-luna) 프로젝트에 사용되는 모든 **신경망 모델**을 개발하고 관리하는 리포지토리입니다.

---

## 📦 구성 모델

| 모델명 | 설명 | 상태 |
|--------|------|------|
| `LunaVision` | 이미지 기반 상황 인식 태거 (NPU 대응) | 🔧 개발 중 |
| `LunaMM` | 이미지 → 문장 생성 | ⏳ 계획 |
| `LunaEmbed` | 이미지/텍스트 공통 임베딩 매핑기 | ⏳ 계획 |
| `LunaTuner` | 캐릭터 LLM 튜닝용 LoRA Adapter | ⏳ 계획 |

---

## 🔨 기본 구조

<pre>
luna-models/
└── README.md
</pre>

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
