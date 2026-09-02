#!/usr/bin/env bash
# Kaggle 노트북 5개의 저장된 출력을 전부 results/ 로 회수한다.
# 사전조건: ~/.kaggle/kaggle.json (chmod 600)
set -u
cd "$(dirname "$0")/.."

echo "=== 내 노트북 목록 ==="
kaggle kernels list --mine || { echo "인증 실패 — ~/.kaggle/kaggle.json 확인"; exit 1; }

echo
echo "=== 각 노트북 출력 회수 시도 ==="
kaggle kernels list --mine --csv 2>/dev/null | tail -n +2 | cut -d, -f1 | while read -r ref; do
  [ -z "$ref" ] && continue
  slug="${ref##*/}"
  echo "--- $ref"
  mkdir -p "results/$slug"
  if kaggle kernels output "$ref" -p "results/$slug" 2>&1 | sed 's/^/    /'; then
    n=$(find "results/$slug" -type f | wc -l)
    if [ "$n" -eq 0 ]; then
      echo "    (출력 없음 — draft이거나 저장된 버전에 산출물이 없음)"
      rmdir "results/$slug" 2>/dev/null
    else
      echo "    ✅ 파일 $n개"
    fi
  fi
done

echo
echo "=== 회수 결과 ==="
find results -type f \( -name '*.json' -o -name '*.pt' -o -name '*.png' -o -name '*.csv' \) \
  -printf '%10s  %p\n' 2>/dev/null | sort -k2 || echo "(없음)"
