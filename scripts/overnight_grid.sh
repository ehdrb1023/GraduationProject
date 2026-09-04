#!/usr/bin/env bash
# 야간 자동 실행 — 남은 시드를 순차로 돌린다.
#
#   · 세션이 없으면 생길 때까지 기다린다 (Colab 이 고아 VM 을 회수할 때까지)
#   · 세션이 중간에 죽으면 다시 얻어서 같은 시드를 재시도한다
#   · 이미 끝난 시드(에폭 50줄 이상)는 건너뛴다
#   · 시드마다 회수 · 그림 · 게이트 분석을 갱신한다
#
# 사용: nohup scripts/overnight_grid.sh > /tmp/overnight.out 2>&1 &
export PATH="$HOME/.local/bin:$PATH"
cd /home/martin1023/Graduation_project || exit 1

SESSION=wsd_d2
LOGDIR=results/grid
SEEDS="45 46"
MAX_RETRY=2
mkdir -p "$LOGDIR" results/figures

# grep -c 는 0건일 때 exit 1 을 내므로 `|| echo 0` 을 쓰면 값이 "0\n0" 으로 오염된다.
# (이 버그로 건전성 게이트가 통째로 무력화된 적이 있다 — PROGRESS §12.8)
count_epochs() {
  local f="$1"
  [ -f "$f" ] || { echo 0; return; }
  local n
  n=$(grep -c "Ep \[" "$f" 2>/dev/null) || n=0
  echo "${n:-0}"
}

# 완주한 arm 수. "에폭 줄 수"로 판정하면 중간에 끊긴 시드를 완료로 오판한다
# (seed 45 가 91에폭 = arm 2개만 완주한 채 끊겼는데 50줄 기준을 넘었다).
count_done_arms() {
  local f="$1"
  [ -f "$f" ] || { echo 0; return; }
  local n
  n=$(grep "Ep \[60/60\]" "$f" 2>/dev/null | grep -c "S2-") || n=0
  echo "${n:-0}"
}

# colab status 는 세션이 없어도 exit 0 을 낸다. 출력 내용으로 판정해야 한다.
has_session() {
  local out
  out=$(colab status -s "$SESSION" 2>&1)
  echo "$out" | grep -qi "not found" && return 1
  echo "$out" | grep -q "Hardware:" && return 0
  return 1
}

ensure_session() {
  local i
  for i in $(seq 1 72); do          # 최대 6시간 대기 (5분 간격)
    if has_session; then return 0; fi
    if colab new -s "$SESSION" --gpu T4 >/tmp/night_new.log 2>&1 && has_session; then
      echo "[$(date +%H:%M)] 세션 확보"
      return 0
    fi
    if [ $((i % 6)) -eq 1 ]; then
      echo "[$(date +%H:%M)] 세션 대기 ($(tr -d '\n' < /tmp/night_new.log | tail -c 70))"
    fi
    sleep 300
  done
  return 1
}

write_notebook() {
  python3 - "$1" <<'PY'
import json, re, sys
seed = sys.argv[1]
nb = json.load(open('GraduationProject/5_phases/1_wsd_grid.ipynb', encoding='utf-8'))
for c in nb['cells']:
    s = ''.join(c.get('source', []))
    if s.startswith('# ══ 9. 그리드 설정'):
        s = re.sub(r'SEEDS = \[[^\]]*\]', f'SEEDS = [{seed}]', s)
        c['source'] = [l + '\n' for l in s.split('\n')[:-1]] + [s.split('\n')[-1]]
json.dump(nb, open(f'/tmp/grid_seed{seed}.ipynb', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
PY
}

for SEED in $SEEDS; do
  LOG="$LOGDIR/seed${SEED}_run_log.txt"

  if [ "$(count_done_arms "$LOG")" -ge 4 ]; then
    echo "[$(date +%H:%M)] seed $SEED 이미 완료 (arm 4개) — 건너뜀"
    continue
  fi

  attempt=0
  while [ "$attempt" -le "$MAX_RETRY" ]; do
    attempt=$((attempt + 1))

    if ! ensure_session; then
      echo "[$(date +%H:%M)] 세션 확보 실패 — 중단"
      exit 1
    fi

    write_notebook "$SEED"
    echo "[$(date +%H:%M)] seed $SEED 실행 (시도 $attempt)"
    colab exec -s "$SESSION" -f "/tmp/grid_seed${SEED}.ipynb" --timeout 25000 > "$LOG" 2>&1

    N=$(count_epochs "$LOG"); A=$(count_done_arms "$LOG")
    echo "[$(date +%H:%M)] seed $SEED 종료 — 에폭 $N 줄 / 완주 arm $A 개"

    if [ "$A" -ge 4 ]; then
      colab download -s "$SESSION" /content/out/phase1_results.tar.gz \
          "$LOGDIR/seed${SEED}_results.tar.gz" >/dev/null 2>&1
      cp "/tmp/grid_seed${SEED}_output.ipynb" "$LOGDIR/" 2>/dev/null
      /tmp/plotenv/bin/python scripts/make_figures.py > results/figures/_last_run.txt 2>&1
      python3 scripts/gate_a_within_arm.py > results/figures/_gate_a.txt 2>&1
      echo "[$(date +%H:%M)] seed $SEED 회수·그림·게이트 갱신 완료"
      break
    fi

    echo "  실패 원인: $(grep -m1 -iE 'lost|error|404|401|NameError' "$LOG" | head -c 90)"
    # 세션이 죽었으면 로컬 기록을 지워 다음 ensure_session 이 새로 만들게 한다
    colab stop -s "$SESSION" >/dev/null 2>&1
    mv "$LOG" "${LOG%.txt}_failed${attempt}.txt" 2>/dev/null
  done
done

colab stop -s "$SESSION" >/dev/null 2>&1
echo "[$(date +%H:%M)] === 야간 실행 종료 ==="
