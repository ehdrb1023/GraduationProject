# Decay-SAM: 사전학습 마지막 구간에만 SAM을 켜서 "덜 잊는" 언어모델 만들기

졸업 프로젝트 실험 저장소입니다.

**한 줄 요약**: 언어모델을 사전학습할 때 **마지막 20% 구간에만 SAM을 켜면**, 비용은 1.2배 정도만 늘고 나중에 다른 일(파이썬 코드)을 가르쳤을 때 원래 배운 것(영어)을 **20~27% 덜 잊습니다.** 이 효과는 AdamW와 Muon 두 옵티마이저에서 모두 나타났습니다.

---

## 1. 쉽게 설명하면

### 무슨 문제를 푸나?
- 언어모델은 보통 **사전학습**(대량의 글로 기본기 학습) → **추가 학습**(코드, 수학 같은 특정 일 학습) 순서로 만듭니다.
- 문제는 추가 학습을 하면 **원래 알던 것을 잊어버린다**는 점입니다 (catastrophic forgetting).

### SAM이 뭔가?
- 보통 학습은 "loss가 낮은 곳"을 찾아갑니다.
- **SAM**(Sharpness-Aware Minimization)은 "loss가 낮으면서 **주변도 평평한 곳**"을 찾아갑니다. 가중치를 일부러 살짝 흔들어 보고, 흔들어도 loss가 별로 안 오르는 방향으로 업데이트합니다.
- 평평한 곳에 있는 모델은 나중에 가중치가 좀 바뀌어도(=추가 학습) 원래 성능이 덜 무너집니다.
- 단점: 매 step 계산을 두 번 해서 **학습 시간이 약 2배**입니다.

### 우리 아이디어
- 학습률 스케줄 WSD(워밍업 → 유지 → **마지막에 감쇠**)에서, 모델이 최종 위치에 "자리 잡는" 구간은 마지막 감쇠 구간입니다.
- 그래서 **SAM을 마지막 20%에만** 켭니다 → 비용은 1.2~1.26배로 줄이고, 효과는 최대한 유지.

### 옵티마이저는 왜 두 개?
- **AdamW**: 지금 대부분의 LLM이 쓰는 표준 옵티마이저
- **Muon**: 최근 주목받는 옵티마이저. 가중치 행렬 전체를 한꺼번에 보고(직교화) 업데이트해서 같은 step에서 더 잘 배움
- 두 옵티마이저는 "거리"를 재는 방식(기하학)이 달라서, SAM의 흔드는 방식도 맞춰줘야 합니다
  - AdamW → **유클리드 SAM** (원래 SAM)
  - Muon → **스펙트럼 SAM** (행렬마다 직교화한 방향으로 흔듦, 2026년 7월 SpecSAM 논문 방식)

---

## 2. 핵심 결과

### 실험 구조
1. BabyLM 영어 데이터(1,570만 토큰)로 사전학습 (10 epoch)
2. 파이썬 코드로 추가 학습 1,000 step (약하게 / 강하게 두 가지)
3. **망각** = 추가 학습 후 영어 검증 loss가 얼마나 올랐나
   - 공정하게 비교하려고, 모든 모델이 **코드를 같은 만큼 배운 지점**에서 비교합니다 ("같은 지점 망각")

### 결과표 (같은 지점 망각, 약한 / 강한 추가 학습)

| 방법 | 사전학습 비용 | 사전학습 loss ↓ | 문법 점수(BLiMP) ↑ | 망각 ↓ | SAM 없음 대비 감소 |
|---|---|---|---|---|---|
| AdamW, SAM 없음 | 1.00배 | 3.288 | 67.1% | 0.643 / 1.409 | — |
| **AdamW + 감쇠 구간 SAM (ρ 0.2)** | **1.21배** | 3.267 | 67.2% | **0.467 / 1.115** | **27% / 21%** |
| Muon, SAM 없음 | 1.00배 | 3.380 | 66.2% | 0.862 / 1.571 | — |
| **Muon + 감쇠 구간 스펙트럼 SAM (ρ 0.05)** | **1.26배** | **3.243** | **68.3%** | 0.626 / 1.196 | **27% / 24%** |
| Muon + 전체 구간 스펙트럼 SAM (ρ 0.05) | 2.27배 | 3.273 | 68.3% | 0.618 / 1.274 | 28% / 19% |

- 모든 비교에서 추가 학습 seed 3개 중 3개가 같은 방향
- 새 태스크(코드)를 덜 배우는 손해는 없음 (감쇠 구간 SAM 기준)

### 결론 4가지

1. **감쇠 구간 SAM은 옵티마이저와 상관없이 통한다.** 흔드는 방식만 옵티마이저에 맞추면, AdamW와 Muon 모두 망각이 20~27% 줄고 비용은 1.2배 정도.
2. **옵티마이저마다 얻는 게 다르다.**
   - Muon + 스펙트럼: 사전학습 품질이 가장 좋음 (loss 가장 낮고 문법 점수 +2%p)
   - AdamW + 유클리드: 망각 절대량이 가장 적고, 실제 학습 시간도 더 빠름
3. **Muon에서는 감쇠 구간만으로 충분하다.** 전체 구간 SAM과 망각 보호는 비슷한데 시간은 약 절반. 오히려 사전학습 loss와 새 태스크 학습은 감쇠 구간 쪽이 더 좋음.
4. **덜 잊는 이유는 "덜 움직여서"가 아니라 "평평한 방향으로 움직여서".** SAM 모델은 추가 학습 때 가중치를 오히려 **더 많이** 움직였는데도 덜 잊었습니다. 움직인 방향의 곡률이 약 1/4로 낮았기 때문입니다.

### 다른 지표로 봐도 (Muon, 감쇠 구간 SAM vs SAM 없음)

| 지표 | 뜻 | SAM 없음 | 감쇠 구간 SAM |
|---|---|---|---|
| BLiMP 문법 정확도 | 영어 문법 이해 | 66.2% | **68.3%** |
| 4비트 양자화 후 loss 증가 | 모델을 1/8 크기로 압축해도 버티나 | +0.95 | **+0.33** |
| 가중치 노이즈 민감도 | 가중치를 살짝 흔들면 얼마나 무너지나 | +0.016 | **+0.007** |
| Hessian trace | 주변이 얼마나 뾰족한가 (평균 곡률) | 9,472 | **2,404** |
| 마지막 층 표현 보존(CKA) | 추가 학습 후 내부 표현이 얼마나 유지되나 (약) | 0.69 | **0.78** |

**잘 안 된 것**: BLiMP **하락폭** 자체는 못 줄였습니다. SAM 모델은 높은 점수에서 출발해서 끝나도 더 높게 남는 형태입니다.

### 시간
- 같은 **step 수** 기준으로는 Muon + 감쇠 구간 SAM이 가장 좋은 품질을 냅니다.
- 하지만 T4 GPU의 **실제 시간** 기준으로는 AdamW가 약 1.5배 빠릅니다. Muon의 직교화 계산 비용 때문입니다. 그래서 "AdamW보다 빠르다"는 주장은 하지 않습니다.

---

## 3. 연구가 흘러온 길

```
비전 (V1~V3)         이미지 모델(ResNet)에서는 효과 없음. 망각의 65%가 BatchNorm 통계 변화 탓
   ↓  BatchNorm 없는 Transformer 언어모델로 이동
LLM + AdamW (L1)     감쇠 구간 SAM(ρ 0.05)으로 망각 5~10% 감소
   ↓  옵티마이저를 Muon으로
LLM + Muon (L2~L3)   원래 SAM(유클리드)은 Muon에 거의 안 통함
                     대신 "사전학습과 같은 옵티마이저로 추가 학습"하면 20~25% 덜 잊음
   ↓  SpecSAM 논문: Muon에는 스펙트럼 방식이 맞다
스펙트럼 SAM (L5~L7) 사전학습 성능까지 크게 개선. 망각 보호의 손잡이는 구간 길이가 아니라 ρ
   ↓
공정 비교 (L7b)      감쇠 구간 ρ 0.05 ≈ 전체 구간 ρ 0.05의 망각 보호를 절반 시간에
   ↓
지표 확장 (E1)       문법·양자화·노이즈·곡률·CKA 대부분 개선. 메커니즘: 평평한 방향으로 움직임
시간 (T1)            step 기준 Muon 우위, 실제 시간은 AdamW 우위
AdamW ρ 스윕 (E2)    ρ를 0.2까지 키우니 AdamW도 Muon만큼 망각 감소 → "옵티마이저 무관" 결론
```

---

## 4. 노트북 목록

GitHub 파일 이름 = Kaggle 노트북 제목. 모든 노트북은 Kaggle T4 GPU에서 **Save & Run All** 한 번으로 실행됩니다.

| 번호 | 파일 | 하는 일 | 핵심 결과 | 소요 |
|---|---|---|---|---|
| V1 | `sam-x-adamw-dacay-cmu.ipynb` | ImageNet-100 사전학습 → 항공기 사진 추가 학습 | 추가 학습이 너무 세서 판단 불가 | 3.5시간 |
| V2 | `sam-x-adamw-dacay-cmu-finetuning.ipynb` | 약한 추가 학습 + BatchNorm 원인 분리 | 망각의 65%가 BN 통계 탓 → 비전 중단 | 3시간 |
| L1 | `babylm-python-adamw.ipynb` | AdamW 사전학습 → AdamW 추가 학습 | 망각 5~10% 감소 | 3시간 |
| L2 | `babylm-muon-adamw.ipynb` | Muon 사전학습 → AdamW 추가 학습 | 유클리드 SAM 효과 미미 | 4시간 |
| L2b | `L2b_Muon_lr-sweep.ipynb` | Muon 학습률 재스윕 | 짧은 스윕 최적값 ≠ 전체 길이 최적값 | 1.3시간 |
| L3 | `babylm-muon-muon.ipynb` | Muon 사전학습 → Muon 추가 학습 | 옵티마이저 일치 시 20~25% 덜 잊음 | 2시간 |
| L5A | `L5A_Muon005_pretrain_euc-spec-full.ipynb` | 스펙트럼 SAM 사전학습 | 사전학습 loss 크게 개선 | 9시간 |
| L5B | `L5B_Muon005-Muon_finetune.ipynb` | L5A 모델 추가 학습 | 전체 구간 SAM 망각 12~20% 감소 | 3시간 |
| L6A | `L6A_Muon005_pretrain_window-rho.ipynb` | SAM 구간 길이·ρ 축 사전학습 | 이득은 마지막 20%에서 포화 | 7.5시간 |
| L6B | `L6B_Muon005-Muon_finetune_window-rho.ipynb` | L6A 모델 추가 학습 | 망각 보호는 ρ가 결정 | 2.5시간 |
| L7 | `L7_Muon005_fullspec005-decay01.ipynb` | 큰 ρ 시험 | ρ가 너무 크면 새 태스크를 덜 배움 | 6.5시간 |
| L7b | `L7b_common-target_recompute.ipynb` | 비교 기준점 통일 (GPU 불필요) | 위 Muon 헤드라인의 출처 | — |
| E1 | `E1_Muon005_eval_metrics.ipynb` | Muon 모델 평가 지표 확장 | 대부분 지표 개선, 메커니즘 확인 | 3시간 |
| T1 | `T1_time-to-target.ipynb` | 목표 loss 도달 시간 (GPU 불필요) | step 기준 Muon, 실제 시간 AdamW | — |
| E2 | `E2_AdamW_eval_rho-sweep.ipynb` | AdamW ρ 스윕 + 지표 + 두 옵티마이저 통합 비교 | ρ 0.2에서 망각 21~27% 감소 | 5시간 |

**실행 순서(의존 관계)**: 뒤 노트북은 앞 노트북의 출력(모델 파일)을 Kaggle Input으로 받습니다.
```
V1 → V2
L1 → E2          L2 → L3
L5A → L5B
L6A → L6B ─┐
L6A → L7 ──┼→ L7b (L5B 포함)
L6A, L7 → E1
L1, L5A, L6A, L7 → T1
```

---

## 5. 설정

### 모든 언어모델 실험에서 같은 것
| 항목 | 값 |
|---|---|
| 모델 | GPT-2 스타일 6층·512차원·8헤드, 문맥 256, 4,480만 파라미터 |
| 토크나이저 | GPT-2 |
| 사전학습 | BabyLM-2026-Strict-Small (5%를 검증용으로 분리), 10 epoch = 19,140 step, 배치 32 |
| 학습률 스케줄 | WSD: warmup 2% → 유지 → 마지막 20% 선형 감쇠 |
| 기타 | weight decay 0.1, 그래디언트 클리핑 1.0, fp16, 사전학습 seed 0 |
| 추가 학습 | codeparrot 파이썬 파일 1만 개, 1,000 step, seed 0·1·2, 사전학습과 같은 옵티마이저, SAM 없음 |
| 평가 문장 | 모든 노트북에서 동일 (BabyLM 400문장, 코드 400문장) |

### 옵티마이저에 맞춰 다르게 둔 것
| 항목 | AdamW | Muon |
|---|---|---|
| 학습률 | 1e-3 (각자 스윕) | 0.005 (임베딩 등은 보조 AdamW 1e-3) |
| SAM 방식 | 유클리드 | 스펙트럼 |
| 시험한 ρ | 0.02, 0.05, 0.1, 0.2 | 0.0005, 0.002, 0.008, 0.02, 0.05, 0.1 |

각 노트북 Output의 `config.json`에 실제 사용한 설정이 저장되어 있습니다.

---

## 6. 한계

- **작은 규모**: 모델 4,480만 파라미터, 데이터 1,570만 토큰, 데이터 쌍 1개(영어 → 파이썬). 큰 모델에서의 효과는 CMU 논문(OLMo-2-1B, AdamW)이 보고한 결과에 기댑니다
- **사전학습 seed 1개** (추가 학습 seed는 3개). 같은 seed 재실행 간 사전학습 loss 변동은 약 0.01로, 주장하는 효과보다 10배 이상 작음
- **AdamW 최적 ρ 미확정**: 0.2가 시험 범위의 끝. 더 큰 ρ와 AdamW 전체 구간 SAM은 아직 미실행
- **Muon 학습률**: 주 실험의 0.005는 전체 길이 기준으로 0.01보다 약간 불리했을 수 있음 (SAM 없음 loss 0.04 차이). 0.01 기준으로 봐도 감쇠 구간 SAM의 사전학습 이득은 유지
- **문법 점수 하락폭**은 줄이지 못함
- **실제 시간**은 T4에서 AdamW가 더 빠름

---

## 7. 참고 문헌

- Foret et al., *Sharpness-Aware Minimization for Efficiently Improving Generalization*, ICLR 2021
- Watts et al., *Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting*, arXiv 2605.02105 (CMU) — 사전학습 후반 SAM이 망각을 줄임 (AdamW, OLMo-2-1B)
- *Optimizer-Model Consistency: Full Finetuning with the Same Optimizer as Pretraining Forgets Less*, arXiv 2605.06654
- Zhong, Milsom, Murray, *Sharpness-Aware Minimization and Muon: Robustness under the Spectral Norm*, arXiv 2607.26001 (SpecSAM)
- Andriushchenko et al., *Sharpness-Aware Minimization Efficiently Selects Flatter Minima Late in Training*, ICLR 2025
- Jordan et al., *Muon: An optimizer for hidden layers in neural networks*, 2024
- BabyLM Challenge; Warstadt et al., *BLiMP*, TACL 2020

자세한 숫자는 `docs/` 폴더의 실험 기록 참고.
