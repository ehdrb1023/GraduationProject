#!/usr/bin/env python3
"""
5_phases/*.ipynb 4개를 하나로 합친다 → 5_phases/ALL_phases.ipynb

목적은 실행이 아니라 **통째로 복사해 외부에 질문하기** 위한 단일 파일이다.
원본 4개는 그대로 두고 여기서 생성만 한다. 원본을 고치면 이 스크립트를 다시 돌린다.
(합친 파일을 손으로 고치지 말 것 — 다음 생성 때 덮어쓴다.)

각 노트북은 자체완결형이라 정의가 겹친다. 단순히 이어붙이면 뒤 섹션의 정의가
앞 섹션 것을 덮어쓴다. 다만 **각 섹션이 실행 직전에 필요한 것을 전부 재정의**하므로
위에서 아래로 순서대로 실행하면 각 섹션은 자기 정의로 돈다.

주의 하나: 0_analysis 는 리포 루트를 찾으려고 os.chdir('..') 을 한다.
합친 파일에서는 그 셀이 뒤 섹션의 경로를 흔들 수 있어 맨 뒤로 보낸다.

사용: python3 scripts/merge_notebooks.py
"""
import json
import os

SRC = 'GraduationProject/5_phases'
OUT = os.path.join(SRC, 'ALL_phases.ipynb')

# 순서 = 실제 실행 권장 순서. 0_analysis 는 cwd 를 바꾸므로 맨 뒤.
ORDER = [
    ('1_wsd_grid.ipynb', 'Phase 1 — WSD × SAM 메인 그리드',
     'GPU 필요. 프로젝트의 핵심 실험. 시드당 약 65분.'),
    ('3_muon.ipynb', 'Phase 3 — Muon 라인 마무리',
     'GPU 필요. 약 1시간. 우선순위 낮음.'),
    ('2_vit.ipynb', 'Phase 2 — ViT-Tiny 재현',
     'GPU 필요. Stage P 36분 + 시드당 30분. Phase 1 게이트 통과 후에.'),
    ('0_analysis.ipynb', 'Phase 0 — 기존 로그 재분석',
     'GPU 불필요. **이 섹션은 cwd 를 리포 루트로 바꾸므로 맨 뒤에 둔다.**'),
]

HEAD = '''# 졸업프로젝트 — 전체 실험 노트북 (통합본)

> **이 파일은 `scripts/merge_notebooks.py` 가 생성한다. 직접 고치지 말 것.**
> 원본은 같은 폴더의 `0_analysis` / `1_wsd_grid` / `2_vit` / `3_muon` 이다.
> 실제 실행은 원본 4개를 따로 쓰는 편이 낫다 (Colab 세션이 약 80분마다 끊긴다).

## 주제

SAM 계열 평탄화(flatness)가 **언제 실제로 일반화에 기여하는가**.

CIFAR-10 / ResNet-18 에서 WSD 학습률 스케줄의 decay 구간에만 SAM 을 켜고,
같은 Stage 1 체크포인트에서 ρ ∈ {0, 0.02, 0.05, 0.1} 로 분기시켜 비교한다.

## 지금까지의 결과 (완주 arm 14개 / seed 42~45)

| ρ | n | 곡률 감소 | 정확도 증가 | Δacc |
|---|---|---|---|---|
| 0.02 | 4 | **4/4** | 1/4 | +0.02 %p |
| 0.05 | 3 | **3/3** | 1/3 | +0.09 %p |
| 0.10 | 3 | **3/3** | **3/3** | +0.26 %p |

- **곡률 감소는 10/10 arm 에서 예외 없다.** 평균이 ρ 에 완벽히 단조 (0.088 → 0.020).
- 정확도는 ρ=0.10 에서 3/3 개선되나 p=0.125 로 아직 유의하지 않다. 5시드 필요.

## 구조

각 섹션은 **자체완결형**이다. 정의가 겹치지만 각 섹션이 실행 직전에
필요한 것을 전부 재정의하므로, 위에서 아래로 실행하면 각자 자기 정의로 돈다.

## 이 코드가 아카이브 노트북과 다른 점

`GraduationProject/1_baseline`~`4_wsd` 의 옛 노트북에는 결함 세 가지가 있고
여기서 전부 고쳤다 (자세한 내용은 `docs/PROGRESS.md` §6).

1. **Stage 2 에 시드가 없었다** — 두 arm 이 서로 다른 데이터 순서를 봤다.
   "차이는 오직 SAM 유무"가 성립하지 않았다.
2. **probe 가 전역 RNG 에 의존** — 세션마다 다른 이미지로 sharpness 를 쟀다.
3. **측정식의 1차항이 grad norm** — `L(w+ε)−L(w) ≈ ρ‖g‖ + ½ρ²(gᵀHg)/‖g‖²`.
   대칭 차분 `½(L(w+ε)+L(w−ε))−L(w)` 로 1차항을 상쇄해 함께 기록한다.
'''


def cell(src, kind='markdown'):
    lines = src.rstrip('\n').split('\n')
    d = {'cell_type': kind, 'metadata': {},
         'source': [l + '\n' for l in lines[:-1]] + [lines[-1]]}
    if kind == 'code':
        d['execution_count'] = None
        d['outputs'] = []
    return d


def main():
    cells = [cell(HEAD)]
    for i, (fn, title, note) in enumerate(ORDER, 1):
        path = os.path.join(SRC, fn)
        if not os.path.exists(path):
            print(f'  건너뜀 (없음): {fn}')
            continue
        nb = json.load(open(path, encoding='utf-8'))
        bar = '=' * 78
        cells.append(cell(
            f'---\n\n# {bar}\n# 섹션 {i} — {title}\n# {bar}\n\n{note}\n\n'
            f'원본: `{fn}`  ·  셀 {len(nb["cells"])}개'))
        for c in nb['cells']:
            c = json.loads(json.dumps(c))     # 원본 보호
            c.pop('id', None)
            if c['cell_type'] == 'code':
                c['execution_count'] = None
                c['outputs'] = []             # 출력은 싣지 않는다 (붙여넣기용)
            cells.append(c)
        print(f'  + {fn}: {len(nb["cells"])}셀')

    json.dump({
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                           'name': 'python3'},
            'language_info': {'name': 'python'},
            'accelerator': 'GPU',
            'colab': {'provenance': [], 'gpuType': 'T4'},
        },
        'nbformat': 4, 'nbformat_minor': 0,
    }, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    size = os.path.getsize(OUT) / 1024
    print(f'\n생성: {OUT}  ({len(cells)}셀, {size:.0f} KB)')


if __name__ == '__main__':
    main()
