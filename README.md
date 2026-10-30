# 졸업프로젝트 코드 정리 — Decay-Phase SAM on AdamW and Muon

WSD 학습률 스케줄의 마지막 감쇠 구간(20%)에만 SAM을 켰을 때, 사전학습 성능과 파인튜닝 후 망각이 어떻게 달라지는지 AdamW(유클리드 SAM)와 Muon(스펙트럼 SAM)에서 비교한 실험 코드 모음입니다. 모든 실험은 Kaggle T4 GPU 1개로 돌렸습니다.

- 최종 보고서: `report/Investigating_the_Effects_of_Decay-Phase_SAM_on_AdamW_and_Muon.pdf`
- 모든 노트북은 실행 결과(출력)가 저장된 상태입니다. 다시 돌리지 않아도 숫자를 확인할 수 있습니다.

---

## 1. 폴더 구조

```
00_explore_CIFAR10/     H  7~8월 탐색: 베이스라인, Muon→SAM 전환, SAM Overlay, WSD 분기 구조, ViT 전이
01_CIFAR100/            C  9월: sam_lab 패키지로 3개 옵티마이저 공정 비교, seed 검증
02_vision_transfer/     V  9월 말: ImageNet-100 → Aircraft, 망각 측정
03_LLM/                 L  10월: BabyLM → 파이썬 코드, AdamW·Muon·스펙트럼 SAM
04_eval/                E·T  10월 말: 평가 지표 확장, 학습 시간, AdamW ρ 스윕, 2×3 격자
report/                 최종 보고서, 그림 생성 코드(figs.py)
```

이름 규칙: `{번호}_{데이터}_{사전학습옵티마이저-파인튜닝옵티마이저}_{내용}.ipynb`
번호는 실험 순서입니다. 같은 실험의 후속·변형은 `b`, `c`를 붙였습니다.

---

## 2. 연구 흐름 한눈에

```
H0a 베이스라인 ─ H1 옵티마이저 교체 → 모멘텀 리셋 문제 발견
   └ H2·H3 SAM Overlay (옵티마이저는 그대로, SAM은 ascent만)
       └ H4 WSD 공통 구간 → decay 구간만 SAM 끔/켬 분기 구조 확립
           ├ H5 사전학습 → 전이 시도 (ViT)
           └ C1~C3 CIFAR-100: 평평해지지만 정확도 이득은 노이즈 수준
               └ (CMU 논문: 효과는 정확도가 아니라 파인튜닝 후 망각에서 나타남)
                   └ V1·V2 비전 망각: 효과 작음, BatchNorm 통계가 망각의 대부분
                       └ L1~L3 LLM(LayerNorm): AdamW에서 망각 감소, Muon은 유클리드 SAM이 안 맞음
                           └ L5~L7 스펙트럼 SAM: Muon 망각 24~27% 감소, decay가 full과 비슷한 보호를 약 절반 시간에
                               └ E1~E3 평가 지표 확장 + AdamW ρ 스윕 + 2×3 격자 (최종 결과)
```

---

## 3. 노트북 목록 (실행 순서)

### 00_explore_CIFAR10 — 탐색 단계 (CIFAR-10 / Tiny-ImageNet)

| 파일 | 원래 이름 | 요약 |
|---|---|---|
| `H0a_CIFAR10_SGD-AdamW-SAM_baseline.ipynb` | Resnet18_CIFAR10 (1).ipynb | 첫 베이스라인. ResNet-18(CIFAR용 3×3 stem), 50ep. SGD 94.22%, AdamW 92.93%, SAM 94.69%. SAM 시간 약 1.8배 |
| `H0b_CIFAR10_Lion-Muon_baseline.ipynb` | Untitled2.ipynb | Lion·Muon을 추가하려던 노트북. Colab이 CPU로 잡혀 실패, 결과 없음 |
| `H1_CIFAR10_Muon-to-SGDSAM_switch-v1v2.ipynb` | Muon.ipynb | Muon으로 학습하다 SGD 기반 SAM으로 갈아타는 동적 스위칭. 84.43% / 85.10%. 갈아탄 직후 loss 급등 → **옵티마이저를 바꾸면 모멘텀이 리셋된다**는 문제 발견 |
| `H2_CIFAR10_Muon_native-SAM.ipynb` | Muon_SAM_CIFAR_10.ipynb | Muon 모멘텀을 유지한 채 ascent만 얹는 방식. Muon만 85.35% vs native SAM 85.73% |
| `H3_CIFAR10_Muon_overlay-ramp-v3.ipynb` | Muon_SAM_v3.ipynb | SAMAscent 클래스(옵티마이저와 분리된 SAM) 첫 정식 버전, ρ 램프. Muon만 86.20 / hybrid 86.07 / 처음부터 SAM 86.54%. sharpness는 0.50 → 0.11 |
| `H3b_CIFAR10_Muon_scope-test-v4.ipynb` | Muon_SAM_v4.ipynb | Muon 적용 범위·weight decay 변경 시험. 개선 없음 |
| `H4_CIFAR10_Lion_WSD-decaySAM-branch.ipynb` | wsd-x-sam-overlay.ipynb | **이후 모든 실험의 뼈대**: 공통 구간 체크포인트(모델+모멘텀) 저장 → decay 구간만 SAM 끔/켬 분기. 87.50 vs 87.64%, sharpness 0.204 → 0.094, 시간 1.9배 |
| `H5_TinyImageNet-CIFAR10_ViT-Lion_FT-decaySAM.ipynb` | notebook926aa3574b.ipynb | 첫 전이 실험. ViT를 Tiny-ImageNet 사전학습 → CIFAR-10 파인튜닝, SAM은 파인튜닝 decay에만. 85.37 vs 85.34% (차이 없음) |
| `H5b_CIFAR10_ViT-Lion_scratch-control.ipynb` | stage1 (1).ipynb | H5의 대조군(사전학습 없음). 코드만 있고 실행 결과 없음 |
| `H5c_TinyImageNet_ViT-Lion_scratch-decaySAM.ipynb` | stage2.ipynb | ViT로 Tiny-ImageNet 직접 학습, decay SAM 분기. 47.09 vs 47.53%, sharpness 0.172 → 0.132 |

※ H1~H3는 ImageNet용 7×7 stem을 써서 H0a보다 정확도가 약 8%p 낮습니다. H0a와 숫자를 직접 비교하면 안 됩니다.

### 01_CIFAR100 — sam_lab 패키지로 공정 비교

| 파일 | 원래 이름 | 요약 |
|---|---|---|
| `C1_CIFAR100_4opt_decaySAM-rho0.05.ipynb` | sam-overlay-4optimizers-kaggle-ipynb.ipynb | SGD·AdamW·Lion·Muon × SAM 끔/decay, ρ0.05, 30ep. 차이 −0.34~+0.02%p (노이즈 수준). 7×7 stem이라 정확도 55~59% |
| `C2_samlab_code_v1_0908.zip` | archive_1.zip | sam_lab 코드 v1 (config, schedule, optim, models, data, engine, train, 스모크 테스트) |
| `C2_samlab_code_v2_0908.zip` | archive_2.zip | v2: sweep.py 추가 |
| `C2_samlab_code_v3_0916.zip` | archive_3.zip | v3: analysis.py, plots.py, analyze.py 추가. C2b·C3가 쓴 최종 코드 |
| `C2a_CIFAR100_3opt_samlab-pilot-grid.ipynb` | notebookd68a24ce5b.ipynb | sam_lab 첫 실행. 스모크 테스트 통과 → 추정 lr로 3옵티마이저 × 끔/decay. 차이 −0.27~+0.34%p. SAM 비용 1.5~2배 |
| `C2b-draft_CIFAR100_3opt_samlab-pipeline.ipynb` | s0-pilot.ipynb | C2b의 이전 버전. 연구 질문과 "새 알고리즘이 아니라 측정 설계가 기여"라는 범위가 적혀 있음. 실험 셀 출력 없음 |
| `C2b_CIFAR100_3opt_lr-rho-sweep.ipynb` | notebook30cb78dff1.ipynb | 학습률 스윕(AdamW 0.003, Lion 3e-4, Muon 0.02) → ρ 스윕 18회(0.02~0.4). 차이 −0.29~+1.24%p |
| `C3_CIFAR100_3opt_seed-variance.ipynb` | seded-seperate-test.ipynb | ρ0 vs ρ0.4 × seed 3개. λmax는 세 seed 모두 크게 감소(AdamW 62→20, Lion 137→31, Muon 66→35). 정확도는 Muon만 +2.03%p로 일관. 가지치기 강건성은 seed 노이즈보다 작음 |

### 02_vision_transfer — ImageNet-100 → FGVC Aircraft

| 파일 | 원래 이름 | 요약 |
|---|---|---|
| `V1_ImageNet100-Aircraft_AdamW-AdamW_decaySAM.ipynb` | sam-x-adamw-dacay-cmu.ipynb | ResNet-18 AdamW 100ep, 마지막 20ep SAM(ρ0.1). 정확도 72.00 vs 71.88%, λmax 66.4 → 11.8. 파인튜닝이 너무 세서 ImageNet 72% → 8~10%로 붕괴 |
| `V2_ImageNet100-Aircraft_AdamW-AdamW_weakFT-sweep.ipynb` | sam-x-adamw-dacay-cmu-finetuning.ipynb | V1 모델로 약한 파인튜닝(lr 1e-4, 3e-5) × seed 3. 사전학습 SAM의 망각 감소는 방향은 일관되나 1%p 안팎 |

※ V3(BatchNorm 통계 재보정: 망각의 65%가 BN 통계) 노트북은 남아 있지 않습니다. 결과는 프로젝트 기록에만 있습니다.

### 03_LLM — BabyLM 사전학습 → 파이썬 코드 파인튜닝

공통 설정: GPT-2형 6층·512차원, 4,480만 파라미터, BabyLM 1,570만 토큰 × 10 epoch(19,140 step), WSD(warmup 2%, 감쇠 20%), 파인튜닝 1,000 step × 약/강 × seed 3. 망각 = 파인튜닝 후 BabyLM 검증 loss 증가.

| 파일 | 요약 | 입력 |
|---|---|---|
| `L1_BabyLM-Python_AdamW-AdamW.ipynb` (babylm-python-adamw) | AdamW lr 스윕 → 1e-3. decay 유클리드 SAM ρ0.05. λmax 113 → 67, 망각 5~10% 감소 (6/6) | — |
| `L2_BabyLM-Python_Muon-AdamW.ipynb` (babylm-muon-adamw) | Muon(lr 0.01) 사전학습 → AdamW 파인튜닝. Muon이 λmax 2.2배, 망각 65% 더 큼. 유클리드 SAM 효과 거의 없음 | — |
| `L2b_BabyLM_Muon_lr-resweep.ipynb` (muon-rho, 이름과 달리 lr 스윕) | Muon lr 0.0025/0.005/0.01 → 0.005 선택 (이후 실험에서 사용) | — |
| `L3_BabyLM-Python_Muon-Muon.ipynb` (babylm-muon-muon) | L2 모델을 Muon으로 파인튜닝. **같은 옵티마이저로 파인튜닝하면 20~25% 덜 잊음** (0.753 / 0.798) | L2 |
| `L5A_Muon005_pretrain_euc-spec-full.ipynb` | Muon 0.005 사전학습 6갈래: off / decay 유클리드 / decay 스펙트럼 ρ 5e-4·2e-3·8e-3 / full 스펙트럼 8e-3. loss 3.390 / 3.370 / … / 3.284 / 3.265 | — |
| `L5B_Muon005-Muon_finetune.ipynb` | L5A 6개 모델 Muon 파인튜닝 36회. full 스펙트럼 망각 −20%, decay ρ0.008은 −5% | L5A |
| `L6A_Muon005_pretrain_window-rho.ipynb` | SAM 구간 40/60% (ρ0.008) + decay ρ 0.02·0.05. off 재현(3.380) | — |
| `L6B_Muon005-Muon_finetune_window-rho.ipynb` | L6A 5개 모델 파인튜닝 30회. **decay ρ0.05 망각 −21% / −20%**. 구간을 늘리는 것보다 ρ를 키우는 게 효과적 | L6A |
| `L7_Muon005_fullspec005-decay01.ipynb` | decay ρ0.1(L6A 80%에서) + full 스펙트럼 ρ0.05(처음부터) 사전학습·파인튜닝. 큰 ρ는 새 태스크 학습을 해침 | L6A, L6B, L5B |
| `L7b_common-target_recompute.ipynb` | GPU 불필요. L5B·L6B·L7 곡선을 모아 **8개 모델 공통 기준점**에서 망각 재계산. decay ρ0.05가 full ρ0.05 보호의 96~126% | L5B, L6B, L7 |

### 04_eval — 평가 지표 확장과 최종 비교

| 파일 | 요약 | 입력 |
|---|---|---|
| `E1_Muon005_eval_metrics.ipynb` | Muon 4개 모델(off, decay 0.02·0.05, full 0.05)에 BLiMP, INT8/INT4 양자화, 가중치 노이즈, Hutchinson trace, λmax, CKA, 방향 곡률·2차 근사. 파인튜닝 24회 (L6B 결과 재현 확인) | L6A, L7 |
| `T1_time-to-target.ipynb` | GPU 불필요. 사전학습 로그로 목표 loss 도달 시간 비교. step 기준은 Muon 우위, 실제 시간은 AdamW가 약 1.5배 빠름 | L1, L5A, L6A, L7 |
| `E2_AdamW_eval_rho-sweep.ipynb` | L1 80% 체크포인트에서 AdamW decay ρ 0.02·0.1·0.2 추가, E1과 같은 지표, 파인튜닝 30회 (L1 결과 재현 확인). Muon과 통합 비교 | L1, E1 |
| `E3_AdamW_decay-rho-full.ipynb` | AdamW decay ρ 0.5·1.0 + full SAM ρ1.0(L1과 같은 초기값·배치 순서로 처음부터). **최종 2×3 격자**: AdamW decay ρ1.0 망각 −51/−43%, full −12/−16%. Muon decay −27/−24%, full −28/−19% | L1, E1, E2 |

### report

| 파일 | 요약 |
|---|---|
| `figs.py` | 보고서 그림 중 노트북에서 나오지 않은 3장(WSD 스케줄, AdamW ρ 그래프, Muon ρ 그래프). 숫자는 E2·E3, L6·L7b 결과를 옮겨 적음 |
| 최종 보고서 PDF | 교수님께 제출한 버전 |

---

## 4. 노트북 사이 데이터 흐름 (Kaggle Input 연결)

```
L1 (exp_llm_adamw) ──────────────┬─→ E2 (exp_e2) ──→ E3 (exp_e3)
                                 └──────────────────→ E3
L2 (exp_llm_muon) ──→ L3
L5A (exp_l5_pretrain) ──→ L5B (exp_l5_finetune) ─┐
L6A (exp_l6_pretrain) ──→ L6B (exp_l6_finetune) ─┼─→ L7 (exp_l7) ──→ L7b
                    └──→ L7, E1                  │
E1 (exp_e1) ──→ E2, E3                           │
L1, L5A, L6A, L7 로그 ──→ T1                      │
V1 (exp_adamw) ──→ V2
```

---

## 5. 보고서 숫자 ↔ 노트북

| 보고서 | 출처 |
|---|---|
| 1.1·1.2 CIFAR에서 정확도 이득 없음 | C1, C2a, C2b, C3 |
| 1.3 비전에서 BatchNorm 의심 | V1, V2 (V3 노트북 없음) |
| 1.3 Muon λmax 2.2배, 망각 65% 더 큼, SAM 후 λmax 3.5배 감소, 망각 1.5% 감소 | L1, L2 |
| 1.4 Muon loss 3.390 / 3.370 / 3.284 | L5A |
| 1.4 ρ0.05에서 망각 24~27% 감소 | L6A, L6B, L7b, E1 |
| 1.5 AdamW ρ0.05 망각 5~10% | L1 |
| 3.1 2×3 격자표, 그림 grid.png | E3 |
| 3.2 AdamW ρ 0.02~1.0 | E2, E3 |
| 3.2 Muon ρ 축, SAM 구간 20/40/60% | L5A~L7, L7b |
| 3.3 BLiMP·INT4·노이즈·CKA (그림 e1_panel.png) | E1 (Muon), E2·E3 (AdamW) |
| 3.4 방향 곡률, 2차 근사 | E1, E2, E3 |
| 3.5 학습 시간 (그림 t1_time.png) | T1 |
| 옵티마이저 일관성 20~25% | L3 |

---

## 6. 설정 일관성 점검 결과 (2026-10-30)

노트북 코드를 직접 파싱해 함수 정의와 설정값을 노트북끼리 대조했습니다.

**일치 확인**
- L1~E3의 모델 구조, 데이터 처리, 평가 문장(BabyLM seed 11, 코드 seed 12), 학습률 스케줄, 파인튜닝 배치 순서, 망각 계산(baby_at)이 모두 같은 코드입니다.
- 평가 지표 함수(BLiMP, 양자화, 노이즈, Hutchinson, CKA, 방향 곡률)는 E1·E2·E3에서 같은 코드입니다.
- AdamW 갈래(L1, E2, E3)는 모두 L1의 같은 80% 체크포인트에서 분기합니다. E3 full SAM은 L1과 같은 초기값·같은 배치 순서로 처음부터 학습합니다.
- Muon 갈래는 각자의 80% 체크포인트(L5A, L6A)에서 분기합니다. L7의 decay ρ0.1은 L6A 체크포인트에서 출발합니다.
- E1은 L6B 망각을, E2는 L1 망각을 소수 넷째 자리까지 재현합니다.

**의도적으로 다른 것**: 옵티마이저별 학습률(AdamW 1e-3, Muon 0.005), SAM 방식(유클리드 / 스펙트럼), ρ 범위, 파인튜닝 옵티마이저(사전학습과 같은 것)

**알고 있는 한계**
1. Muon "SAM 없음" 기준선이 두 개(L5A 3.390, L6A 3.380). 같은 코드·seed로 두 번 돌린 값이고, 각 비교는 자기 계열 기준선과 짝지음
2. SAM 구간 길이 축의 사전학습 loss는 L5A(20·100%)와 L6A(40·60%) 실행이 섞여 있어, 정밀한 비율은 의미가 약함
3. 비용 배수: Muon은 실제 실행 시간 비율, AdamW(E2·E3)는 같은 세션에서 step 시간을 재서 계산
4. 실행 환경: L1·L2는 torch 2.10, L5A 이후는 2.11. 시간 비교에만 영향(약 10%)
5. Muon lr 0.005는 짧은 스윕 최적값. 사전학습 seed 1개, ρ를 같은 데이터 결과로 선택
6. sam_lab의 sweep.py에 AdamW lr 0.01이 남아 있었으나 C2b는 0.003으로 돌았고, C3에서 0.003으로 맞춤
7. 기준점이 다른 두 감소율: AdamW끼리 비교(E3 셀 7)는 49.4/32.7%, AdamW·Muon 6개 공통 지점(E3 셀 9)은 51/42.5%

---

## 7. 실행 방법

1. Kaggle 노트북 설정: Accelerator **GPU T4**, Internet **On**
2. 입력이 필요한 노트북은 Add Input에서 위 데이터 흐름의 이전 노트북 출력을 붙입니다. 노트북이 `find_input()`으로 폴더를 자동으로 찾습니다.
3. 위에서 아래로 순서대로 실행합니다. 대부분 체크포인트를 저장하므로 세션이 끊겨도 다시 실행하면 이어서 돕니다.
4. 대략적인 시간(T4): L1 3시간, L5A 9시간, L6A 7.5시간, L7·E1·E2·E3 각 2~5시간
