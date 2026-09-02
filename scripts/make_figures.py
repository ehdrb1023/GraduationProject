#!/usr/bin/env python3
"""
발표용 그림 생성.

입력은 두 종류다.
  1) results/grid/seed*_run_log.txt   — 신규 실험 (5_phases/1_wsd_grid.ipynb 출력)
  2) results/parsed/*.csv             — 아카이브 노트북에서 복구한 로그

로그 텍스트만으로 전부 그리도록 만들었다. JSON 이 없어도 동작한다
(Colab VM 이 임시라 원본을 잃은 전례가 있다 — docs/PROGRESS.md §8).

사용: python3 scripts/make_figures.py
출력: results/figures/*.png  (+ figure_data.json)
"""
import csv
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict

OUT = 'results/figures'
GRID = 'results/grid'
PARSED = 'results/parsed'
RHO_PROBE = 0.05      # 노트북 CFG 의 sharpness_rho

# 신규 실험 로그 한 줄 형식
#   [S2-seed42_sam0.05_d20|decay ] Ep [41/60] | 49.1s | Loss 0.31 | Train 88.97% |
#   Test 84.61% | lr 0.00019 | rho 0.0167 | gn 1.234 | Sharp1 0.16 | SharpSym 0.048 | pg 1.48
RX = re.compile(
    r'\[(?P<tag>[^\]|]+)\|\s*(?P<phase>\w+)\s*\] Ep \[(?P<ep>\d+)/(?P<tot>\d+)\] \| '
    r'(?P<sec>[\d.]+)s \| Loss (?P<loss>[\d.]+) \| Train (?P<train>[\d.]+)% \| '
    r'Test (?P<test>[\d.]+)% \| lr (?P<lr>[\d.e+-]+) \| rho (?P<rho>[\d.]+)'
    r'(?: \| gn (?P<gn>[\d.]+))?'
    r'(?: \| Sharp1 (?P<one>[\d.]+) \| SharpSym (?P<sym>[\d.e+-]+))?'
    r'(?: \| pg (?P<pg>[\d.]+))?')


# ------------------------------------------------------------------ 적재
def load_runs():
    """신규 실험 로그를 arm 단위로 읽는다."""
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(GRID, 'seed*_run_log.txt'))):
        for m in RX.finditer(open(path, encoding='utf-8', errors='replace').read()):
            d = m.groupdict()
            runs[d['tag'].strip()].append(d)
    return runs


def arm_meta(tag):
    """'S2-seed42_sam0.05_d20' → (42, 0.05)   'S2-seed42_samoff_d20' → (42, 0.0)"""
    m = re.search(r'seed(\d+)_(samoff|sam([\d.]+))', tag)
    if not m:
        return None, None
    return int(m.group(1)), (0.0 if m.group(2) == 'samoff' else float(m.group(3)))


def final_table(runs):
    """arm 별 최종값. decay 가 끝난 arm 만."""
    rows = []
    for tag, rs in runs.items():
        if not tag.startswith('S2-'):
            continue
        seed, rho = arm_meta(tag)
        if seed is None or rs[-1]['ep'] != rs[-1]['tot']:
            continue
        sh = [r for r in rs if r['sym']]
        if not sh:
            continue
        rows.append(dict(tag=tag, seed=seed, rho=rho,
                         acc0=float(rs[0]['test']), acc=float(rs[-1]['test']),
                         one=float(sh[-1]['one']), sym=float(sh[-1]['sym']),
                         pg=float(sh[-1]['pg']) if sh[-1]['pg'] else None))
    return sorted(rows, key=lambda r: (r['seed'], r['rho']))


def sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0


# ------------------------------------------------------------------ 그림
def fig_rho_sweep(rows, plt):
    """★ 메인 그림 — rho 에 대해 평탄도와 일반화가 어떻게 반응하는가."""
    by = defaultdict(lambda: {'sym': [], 'acc': [], 'one': []})
    for r in rows:
        by[r['rho']]['sym'].append(r['sym'])
        by[r['rho']]['acc'].append(r['acc'])
        by[r['rho']]['one'].append(r['one'])
    rhos = sorted(by)
    if len(rhos) < 2:
        return None
    n = len(by[rhos[0]]['sym'])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].errorbar(rhos, [st.mean(by[r]['sym']) for r in rhos],
                   yerr=[sd(by[r]['sym']) for r in rhos],
                   marker='o', ms=7, capsize=5, lw=2, color='#7048b6')
    ax[0].set_xlabel('SAM radius  $\\rho$'); ax[0].set_ylabel('Curvature (symmetric sharpness)')
    ax[0].set_title('Flatness responds to $\\rho$')
    ax[0].grid(alpha=.3)

    ax[1].errorbar(rhos, [st.mean(by[r]['acc']) for r in rhos],
                   yerr=[sd(by[r]['acc']) for r in rhos],
                   marker='s', ms=7, capsize=5, lw=2, color='#1f77b4')
    ax[1].set_xlabel('SAM radius  $\\rho$'); ax[1].set_ylabel('Test accuracy (%)')
    ax[1].set_title('Generalization — does it follow?')
    ax[1].grid(alpha=.3)

    fig.suptitle(f'Decay-phase SAM: $\\rho$ sweep  (n={n} seed{"s" if n>1 else ""})',
                 fontsize=12)
    plt.tight_layout()
    return save(fig, 'fig1_rho_sweep', plt)


def fig_trajectories(runs, plt):
    """decay 구간에서 accuracy 는 오르는데 sharpness 는 어떻게 움직이는가."""
    arms = {t: rs for t, rs in runs.items() if t.startswith('S2-')}
    if not arms:
        return None
    cmap = {0.0: '#d62728', 0.02: '#ff7f0e', 0.05: '#2ca02c', 0.1: '#1f77b4'}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    seen = set()
    for tag, rs in sorted(arms.items()):
        seed, rho = arm_meta(tag)
        c = cmap.get(rho, '#888')
        lab = f'$\\rho$={rho:g}' if rho not in seen else None
        seen.add(rho)
        ep = [int(r['ep']) for r in rs]
        ax[0].plot(ep, [float(r['test']) for r in rs], color=c, lw=1.4, alpha=.8, label=lab)
        sh = [r for r in rs if r['sym']]
        ax[1].plot([int(r['ep']) for r in sh], [float(r['sym']) for r in sh],
                   color=c, lw=1.4, marker='o', ms=3, alpha=.8, label=lab)
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Test accuracy (%)')
    ax[0].set_title('Accuracy rises through the decay phase')
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Curvature (symmetric)')
    ax[1].set_title('Curvature separates by $\\rho$')
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=8)
    plt.tight_layout()
    return save(fig, 'fig2_decay_trajectories', plt)


def fig_metric_problem(rows, runs, plt):
    """측정식 문제 — 단측 sharpness 의 1차항이 grad norm 이다.
    pg(probe grad norm)가 기록된 arm 만 쓴다. seed 42 는 pg 추가 이전이라 제외된다."""
    rows = [r for r in rows if r['pg'] is not None]
    if len(rows) < 2:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) 단측 = 1차항 + 곡률 분해
    labs = [f"s{r['seed']}\n$\\rho$={r['rho']:g}" for r in rows]
    x = range(len(rows))
    first = [r['one'] - r['sym'] for r in rows]
    ax[0].bar(x, first, .6, label='first-order term  $\\approx \\rho\\|g\\|$', color='#bbb')
    ax[0].bar(x, [r['sym'] for r in rows], .6, bottom=first,
              label='curvature (symmetric)', color='#7048b6')
    ax[0].set_xticks(list(x)); ax[0].set_xticklabels(labs, fontsize=7)
    ax[0].set_ylabel('One-sided sharpness')
    ax[0].set_title('One-sided sharpness is mostly gradient norm')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis='y')

    # (b) 1차항이 이론값 rho*|g| 를 따라가는가
    # 점들이 y=x 아래에 놓이는 것은 정상이다. 테일러 전개는 rho→0 근사인데
    # rho=0.05 는 무한소가 아니어서 고차항이 남는다. 관심사는 기울기 1 이 아니라
    # "1차항이 |g| 에 선형으로 붙어 있는가" 이므로 상관계수를 함께 표시한다.
    pred = [RHO_PROBE * r['pg'] for r in rows]
    ax[1].scatter(pred, first, s=55, color='#d62728', zorder=3)
    lim = [0, max(max(pred), max(first)) * 1.1]
    ax[1].plot(lim, lim, '--', color='#888', lw=1, label='y = x  (exact 1st order)')
    if len(rows) >= 3:
        mp, mf = st.mean(pred), st.mean(first)
        sp = sum((a-mp)**2 for a in pred)**.5; sf = sum((b-mf)**2 for b in first)**.5
        r = sum((a-mp)*(b-mf) for a, b in zip(pred, first))/(sp*sf) if sp and sf else float('nan')
        ax[1].annotate(f'r = {r:+.3f}', xy=(.05, .88), xycoords='axes fraction', fontsize=10)
    ax[1].set_xlabel('$\\rho \\cdot \\|g\\|$  (first-order prediction)')
    ax[1].set_ylabel('one-sided $-$ symmetric  (measured)')
    ax[1].set_title('Measured first-order term tracks $\\rho\\|g\\|$\n(below $y{=}x$: higher-order terms at $\\rho{=}0.05$)',
                    fontsize=10)
    ax[1].legend(fontsize=8, loc='lower right'); ax[1].grid(alpha=.3)
    plt.tight_layout()
    return save(fig, 'fig3_metric_decomposition', plt)


def fig_detrend(plt):
    """아카이브 로그 — raw 상관이 시간 추세 아티팩트임을 보인다."""
    ARMS = [('ResNet v1 off', '09_wsd_sam_overlay_v1__c8.csv'),
            ('ResNet v1 on',  '09_wsd_sam_overlay_v1__c9.csv'),
            ('ResNet v2 off', '10_wsd_sam_overlay_v2__c8.csv'),
            ('ResNet v2 on',  '10_wsd_sam_overlay_v2__c9.csv'),
            ('ViT off',       '11_tiny_imagenet_wsd_sam__c7.csv'),
            ('ViT on',        '11_tiny_imagenet_wsd_sam__c8.csv')]

    def pear(a, b):
        n = len(a)
        if n < 3: return float('nan')
        ma, mb = st.mean(a), st.mean(b)
        sa = sum((x-ma)**2 for x in a)**.5; sb = sum((y-mb)**2 for y in b)**.5
        if sa == 0 or sb == 0: return float('nan')
        return sum((x-ma)*(y-mb) for x, y in zip(a, b))/(sa*sb)

    names, raw, det = [], [], []
    for lab, fn in ARMS:
        p = os.path.join(PARSED, fn)
        if not os.path.exists(p): continue
        rs = [r for r in csv.DictReader(open(p)) if r['sharpness'] and r['test_acc']]
        if len(rs) < 4: continue
        s = [float(r['sharpness']) for r in rs]; a = [float(r['test_acc']) for r in rs]
        ds = [s[i+1]-s[i] for i in range(len(s)-1)]
        da = [a[i+1]-a[i] for i in range(len(a)-1)]
        names.append(lab); raw.append(pear(s, a)); det.append(pear(ds, da))
    if not names: return None

    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(names))
    ax.bar([i-.2 for i in x], raw, .4, label='raw correlation', color='#999')
    ax.bar([i+.2 for i in x], det, .4, label='detrended (first difference)', color='#7048b6')
    ax.axhline(0, color='k', lw=.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('corr(sharpness, accuracy)')
    ax.set_title('The apparent sharpness–accuracy link is a time trend')
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis='y')
    plt.tight_layout()
    return save(fig, 'fig4_detrended_correlation', plt)


def save(fig, name, plt):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + '.png')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('  ', path)
    return path


# ------------------------------------------------------------------ main
def main():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                             'axes.spines.right': False})
    except ImportError:
        print('matplotlib 이 없다. 설치:  uv tool install --with matplotlib ...')
        print('또는 Colab 에서 실행할 것.')
        return

    runs = load_runs()
    rows = final_table(runs)
    print(f'신규 실험 arm {len(rows)}개 / 아카이브 CSV {len(glob.glob(PARSED+"/*.csv"))}개')

    if rows:
        print('\n최종 표')
        print(f'{"seed":>5}{"rho":>7}{"acc":>8}{"one":>9}{"sym":>11}{"|g|":>8}')
        for r in rows:
            pg = f'{r["pg"]:.3f}' if r['pg'] else '  —  '
            print(f'{r["seed"]:>5}{r["rho"]:>7.2f}{r["acc"]:>8.2f}'
                  f'{r["one"]:>9.4f}{r["sym"]:>11.6f}{pg:>8}')

    print('\n그림 생성')
    made = [f for f in [fig_rho_sweep(rows, plt), fig_trajectories(runs, plt),
                        fig_metric_problem(rows, runs, plt), fig_detrend(plt)] if f]
    os.makedirs(OUT, exist_ok=True)
    json.dump({'rows': rows, 'figures': made},
              open(os.path.join(OUT, 'figure_data.json'), 'w'), indent=1)
    print(f'\n{len(made)}개 생성 → {OUT}/')


if __name__ == '__main__':
    main()
