# 졸업프로젝트 — 옵티마이저 기하와 일반화

SAM 계열 평탄화(flatness)가 **언제** 실제로 일반화에 기여하는지를 조사한다.
CIFAR-10 / ResNet-18 을 고정 벤치마크로 두고, WSD 학습률 스케줄의 decay 구간에서만
SAM 을 켠 arm 과 끈 arm 을 같은 체크포인트에서 분기시켜 비교한다.

현재까지의 관찰: **SAM 은 decay 구간에서 sharpness 를 확실히 낮추지만,
그 감소가 test accuracy 개선으로 이어지지 않는다.** ResNet-18 과 ViT-Tiny 에서 재현됐다.

## 문서

| 문서 | 내용 |
|---|---|
| [docs/PROGRESS.md](docs/PROGRESS.md) | **진행 상황 · 실험 결과 · 다음 단계** — 여기부터 |
| [docs/colab-cli.md](docs/colab-cli.md) | Colab CLI 셋업, 함정, 그리드 실행법 |
| [docs/COLAB_SKILL.md](docs/COLAB_SKILL.md) | CLI 동봉 에이전트 가이드 |
| [results/README.md](results/README.md) | 산출물 설명 |

## 실행

신규 실험은 [`GraduationProject/5_phases/`](GraduationProject/5_phases/) 의 노트북으로 돈다.
Colab 에 올려 런타임을 T4 로 두고 위에서부터 실행하면 된다.

```
0_analysis.ipynb   기존 로그 재분석 (GPU 불필요)
1_wsd_grid.ipynb   ★ 메인 그리드 + Gate A/B/C 판정
2_vit.ipynb        ViT-Tiny 재현
3_muon.ipynb       Muon + SAM
```

CLI 로 걸어둘 수도 있다 (`colab exec` 는 `.ipynb` 를 그대로 실행한다):

```bash
colab new  -s wsd --gpu T4
colab exec -s wsd -f GraduationProject/5_phases/1_wsd_grid.ipynb --timeout 25000
colab download -s wsd /content/out/phase1_results.tar.gz ./results/
colab stop -s wsd
```

## 구조

```
docs/                     문서
scripts/                  보조 도구
GraduationProject/
  1_baseline/ 2_muon/     과거 실험 아카이브
  3_flatness/ 4_wsd/      폴더=주제, 파일 앞 번호=진행 순서(01→11)
  5_phases/               ★ 신규 실험 — 자체완결형 노트북 4개
results/                  산출물 · 파싱된 로그
```

> `1_baseline`~`4_wsd` 는 기록용 아카이브다. Stage 2 시드 미고정 등 결함이 있으므로
> 새 실험은 반드시 `5_phases/` 의 노트북을 쓴다.
