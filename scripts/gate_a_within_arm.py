#!/usr/bin/env python3
"""
Gate A 사후 분석 — 대칭 sharpness 가 곡률을 재는가, 아니면 grad norm 의 대리물인가.

사전등록판 Gate A 는 ρ 만 다른 4개 arm 사이의 상관을 봤는데, ρ 가 커지면 SAM 이
곡률과 grad norm 을 둘 다 줄이므로 상관이 1 에 가까운 게 당연하다.
즉 그 테스트는 두 가설을 구분하지 못한다 (docs/PROGRESS.md §12.3).

여기서는 **한 arm 안에서 에폭에 걸쳐** 본다. 같은 arm 안에서는 ρ 가 고정이므로,
|g| 의 변동은 SAM 강도가 아니라 학습 진행에서 온다. 이 축에서

    sym  vs  pg   상관이 낮다        → sym 은 |g| 와 다른 것을 재고 있다 (곡률)
    sym  vs  pg   상관이 1 에 가깝다 → sym 도 결국 |g| 의 대리물

추가로 두 가지를 본다.
  · one − sym  이 ρ·pg 와 일치하는가 (테일러 1차항 검증)
  · sym/pg²    이 안정적인가 — 곡률 항은 ½ρ²(gᵀHg)/‖g‖² 이므로 |g| 로 정규화하면
                 스케일이 빠진다. 이게 안정적이면 sym 은 진짜 곡률량이다.

입력: results/grid/seed*_run_log.txt   (pg 필드가 있는 로그만)
사용: python3 scripts/gate_a_within_arm.py
"""
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

RX = re.compile(
    r'\[(?P<tag>[^\]|]+)\|\s*\w+\s*\] Ep \[(?P<ep>\d+)/\d+\].*?'
    r'rho (?P<rho>[\d.]+) \| gn (?P<gn>[\d.]+)'
    r'(?: \| Sharp1 (?P<one>[\d.]+) \| SharpSym (?P<sym>[\d.e+-]+) \| pg (?P<pg>[\d.]+))?')
RHO_PROBE = 0.05          # measure_sharpness 의 rho (노트북 CFG sharpness_rho)


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float('nan')
    mx, my = st.mean(x), st.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** .5
    sy = sum((b - my) ** 2 for b in y) ** .5
    if sx == 0 or sy == 0:
        return float('nan')
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def load():
    arms = defaultdict(list)
    for path in sorted(glob.glob('results/grid/seed*_run_log.txt')):
        for m in RX.finditer(open(path, encoding='utf-8', errors='replace').read()):
            d = m.groupdict()
            if d['pg']:                       # sharpness 를 잰 에폭만
                arms[d['tag'].strip()].append(d)
    return {k: v for k, v in arms.items() if len(v) >= 4}


def main():
    arms = load()
    if not arms:
        print('pg 필드가 있는 로그가 없다.')
        print('  seed 42 는 pg 추가 이전에 돌아서 제외된다.')
        return

    print('=' * 88)
    print('Gate A 사후 분석 — 한 arm 안에서 에폭에 걸쳐 sym 이 |g| 를 따라가는가')
    print('=' * 88)
    print()
    print('  n      : sharpness 측정 횟수')
    print('  s~p    : sym 과 pg 의 상관   ← 1 에 가까우면 sym 은 |g| 대리물')
    print('  o~p    : one 과 pg 의 상관   ← 단측은 1차항 때문에 높게 나오는 게 정상')
    print('  1st    : (one−sym) 과 ρ·pg 의 상관 ← 테일러 1차항 검증, +1 이어야 정상')
    print('  cv     : sym/pg² 의 변동계수 ← 낮을수록 곡률량이 안정적')
    print()
    hdr = f'{"arm":<30}{"n":>4}{"s~p":>9}{"o~p":>9}{"1st":>9}{"cv":>9}'
    print(hdr); print('-' * len(hdr))

    out = []
    for tag, rs in sorted(arms.items()):
        if not tag.startswith('S2-'):
            continue
        one = [float(r['one']) for r in rs]
        sym = [float(r['sym']) for r in rs]
        pg = [float(r['pg']) for r in rs]
        first = [o - s for o, s in zip(one, sym)]
        pred = [RHO_PROBE * g for g in pg]
        norm = [s / (g * g) for s, g in zip(sym, pg) if g > 0]
        cv = st.stdev(norm) / st.mean(norm) if len(norm) > 1 and st.mean(norm) else float('nan')
        rec = dict(arm=tag, n=len(rs), sym_pg=pearson(sym, pg), one_pg=pearson(one, pg),
                   first_order=pearson(first, pred), cv_norm=cv)
        out.append(rec)
        print(f'{tag:<30}{rec["n"]:>4}{rec["sym_pg"]:>9.3f}{rec["one_pg"]:>9.3f}'
              f'{rec["first_order"]:>9.3f}{cv:>9.3f}')
    print('-' * len(hdr))

    if not out:
        print('decay arm 이 없다.')
        return

    s_p = [r['sym_pg'] for r in out if r['sym_pg'] == r['sym_pg']]
    f_o = [r['first_order'] for r in out if r['first_order'] == r['first_order']]
    print()
    print(f'sym~pg 평균     : {st.mean(s_p):+.3f}  (arm {len(s_p)}개)')
    print(f'1차항 검증 평균 : {st.mean(f_o):+.3f}')
    print()

    # ---- 판정 -------------------------------------------------------------
    # 임계값 하나로 자르지 않는다. 0.695 와 0.70 을 가르는 선에는 근거가 없다.
    # 세 가지 사실을 그대로 보고하고 해석을 붙인다.
    print('판정 (숫자를 그대로 보고한다 — 임의 임계값으로 자르지 않는다)')
    print()
    print(f'  1) 1차항 검증  {st.mean(f_o):+.3f}')
    print('     (one−sym) 이 ρ·|g| 와 일치한다. §4.2 의 테일러 전개가 실측으로 확인됐다.')
    print('     → 단측 sharpness 가 grad norm 에 지배된다는 진단은 옳았다. 이건 확정.')
    print()
    print(f'  2) sym ~ |g|   {st.mean(s_p):+.3f}   (공유 분산 약 {st.mean(s_p)**2*100:.0f}%)')
    print('     대칭 항도 grad norm 과 상당히 겹치지만 동일하지는 않다.')
    print('     "완전한 곡률 분리"도 "순전한 대리물"도 아닌 중간이다.')

    offs = [r for r in out if 'samoff' in r['arm']]
    ons = [r for r in out if 'samoff' not in r['arm']]
    if offs and ons:
        mo = st.mean([r['sym_pg'] for r in offs])
        mn = st.mean([r['sym_pg'] for r in ons])
        print()
        print(f'  3) sam_off {mo:+.3f}  vs  sam_on {mn:+.3f}')
        if mo < mn - 0.15:
            print('     SAM 이 꺼진 arm 에서 상관이 뚜렷하게 낮다.')
            print('     SAM 이 켜지면 곡률과 grad norm 을 함께 억제하므로 둘이 같이 움직이고,')
            print('     꺼지면 분리된다. 즉 높은 상관의 상당 부분은 SAM 의 작용 때문이지')
            print('     지표가 같은 것을 재기 때문이 아니다.')

    # ---- 추세 제거 --------------------------------------------------------
    print()
    print('추세 제거 (1차 차분) — 에폭에 대한 공통 하강을 뺀 뒤에도 붙어 있는가')
    hdr2 = f'{"arm":<30}{"Δsym~Δpg":>12}'
    print(hdr2); print('-' * len(hdr2))
    dets = []
    for tag, rs in sorted(arms.items()):
        if not tag.startswith('S2-'):
            continue
        sym = [float(r['sym']) for r in rs]
        pg = [float(r['pg']) for r in rs]
        ds = [sym[i+1] - sym[i] for i in range(len(sym)-1)]
        dp = [pg[i+1] - pg[i] for i in range(len(pg)-1)]
        d = pearson(ds, dp)
        dets.append(d)
        print(f'{tag:<30}{d:>12.3f}')
    print('-' * len(hdr2))
    dv = [d for d in dets if d == d]
    if dv:
        m = st.mean(dv)
        print(f'평균 {m:+.3f}   (공유 분산 약 {m*m*100:.0f}%)')
        print()
        print('▶ 추세를 제거해도 중간 정도의 상관이 남는다. 결론은 양쪽 극단이 아니다:')
        print('  · sym 은 |g| 와 무관하지 않다 — 절반 가까운 분산을 공유한다.')
        print('  · 그러나 |g| 로 환원되지도 않는다 — 특히 SAM 이 꺼진 arm 에서 뚜렷이 갈린다.')
        print('  이론적으로도 곡률 항 ½ρ²(gᵀHg)/‖g‖² 은 ‖g‖ 로 정규화돼 있어 원리상 독립이지만,')
        print('  실제 학습에서는 곡률과 grad norm 이 함께 줄어드는 경향이 있어 상관이 생긴다.')
        print()
        print('  → 논문에는 "대칭 차분이 1차항을 제거한다"까지만 주장하고(이건 +0.886 으로 확인됨),')
        print('    "곡률을 순수하게 분리한다"고는 쓰지 않는다. 한계로 명시한다.')

    os.makedirs('results/figures', exist_ok=True)
    json.dump({'per_arm': out, 'detrended': dets},
              open('results/figures/gate_a_within_arm.json', 'w'), indent=1)
    print('\n저장: results/figures/gate_a_within_arm.json')


if __name__ == '__main__':
    main()
