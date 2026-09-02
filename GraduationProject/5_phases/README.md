# 5_phases — 신규 실험 (2026-09~)

`1_baseline` ~ `4_wsd` 는 기록용 아카이브다. **새 실험은 전부 여기서 돈다.**

노트북 4개는 **자체완결형**이다. 외부 `.py` 없이 각자 혼자 돈다.
로직·설정·결과가 한 파일 안에 있어서, 팀원이 노트북만 열면 무엇을 어떻게 돌렸고
결과가 어땠는지 그대로 보인다.

## 파일

| 노트북 | 셀 | 역할 | GPU |
|---|---|---|---|
| `0_analysis.ipynb` | 6 | 기존 로그 재분석 — 궤적·추세제거·산점도 | 불필요 |
| `1_wsd_grid.ipynb` | 16 | **★ 메인 그리드** + Gate A/B/C 판정 | T4 |
| `2_vit.ipynb` | 13 | ViT-Tiny 사전학습 → 파인튜닝 → 분기 | T4 |
| `3_muon.ipynb` | 12 | Muon + SAM 짝 비교 | T4 |

## 실행 순서

```
0.  0_analysis.ipynb    기존 로그 분석 (즉시)
1.  1_wsd_grid.ipynb    ★ 메인. SEEDS=[42] 로 검증 → 5시드로 확장
2.  (게이트 판정)        1_wsd_grid.ipynb 12번 셀
3.  3_muon.ipynb        자투리 시간 (1시간)
4.  2_vit.ipynb         Phase 1 이 게이트를 통과한 뒤에
```

Phase 2 를 뒤로 두는 이유: ResNet 결과가 애매하면 "두 아키텍처에서 재현"이라는
논지 자체를 다시 써야 한다. 크로스-아키텍처 확장 전에 메인 축을 확정하는 게 안전하다.

## 노트북 구조 (전부 동일)

```
1. 셋업        2. 재현성      3. 데이터     4. 모델/옵티마이저
5. 스케줄·sharpness          6. 실시간 그래프
7. 학습 루프   8. Stage 함수  9. 설정       10. 실행
11. 결과표     12. 게이트     13. 그림      14. ★ 회수
```

**설정은 9번 셀 하나에만 있다.** 시드·rho·에폭을 바꾸려면 거기만 고치면 된다.

**Stage 2 는 함수 호출이지 복사한 셀이 아니다.**
아카이브 노트북의 가장 큰 결함(§6.1 — 두 arm 이 다른 데이터 순서를 봄)이
"셀을 복사해 플래그만 바꾸는" 구조에서 나왔기 때문에, `stage2(seed, rho, ...)`
함수로 만들어 재발을 구조적으로 막았다.

## 실행 방법

### A. Colab 웹 (실시간으로 보고 싶을 때 — 권장)
노트북을 업로드하고 런타임을 **T4** 로 설정한 뒤 위에서부터 실행.
매 에폭 accuracy / sharpness(단측·대칭) / grad norm 그래프가 갱신된다.
거슬리면 6번 셀에서 `LIVE['every'] = 5` 또는 `LIVE['on'] = False`.

### B. CLI (걸어두고 다른 일 할 때)
`colab exec` 는 `.ipynb` 를 그대로 실행하고 결과를 `<이름>_output.ipynb` 로 저장한다.

```bash
colab new  -s wsd --gpu T4
colab exec -s wsd -f GraduationProject/5_phases/1_wsd_grid.ipynb --timeout 25000
colab download -s wsd /content/out/phase1_results.tar.gz ./results/
colab stop -s wsd
```

⚠️ `--timeout` 기본값이 **30초**다. 반드시 명시할 것.

## ★ 결과 회수

**Colab VM 은 임시다.** Kaggle 에서 이미 한 번 전부 날렸다(`docs/PROGRESS.md` §8).
각 노트북 **마지막 셀**이 `tar.gz` 를 만들고 브라우저 다운로드를 띄운다.
**세션을 끄기 전에 반드시 실행할 것.**

`2_vit.ipynb` 의 `vit_pretrain_tinyimagenet.pt` 는 특히 중요하다 —
재학습에 36분이 들고, 다시 만들면 backbone 이 달라져 기존 ViT 수치와 비교가 끊긴다.

## 알려진 성질

- **decay 마지막 1에폭은 lr 이 정확히 0 이다.** WSD 스케줄이 `peak*(1 − k/D)` 라
  `k=D` 에서 0 이 된다. 그 에폭은 가중치가 안 움직이고 BatchNorm 통계만 갱신된다
  (ViT 는 BN 이 없어 지표가 완전히 동일하게 나온다).
  20에폭 중 1에폭이라 5% 손해지만, 고치면 기존 v1/v2 결과와 비교가 끊기므로 그대로 둔다.
- `1_wsd_grid` 의 Gate B 는 **시드 3개 미만이면 "판정 불가"** 를 낸다.
  부호검정은 5시드에서야 p=0.031 로 0.05 를 넘는다.
