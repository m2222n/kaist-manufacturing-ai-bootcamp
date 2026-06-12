#!/bin/bash
# v3 배치 — 6/11 미팅 지시 반영: 부품 회색 단색 + 배경 2종(투명 플라스틱 박스 / 흰색 책상).
# scene마다 깨끗한 프로세스로 gen_one_v3.py 호출 (상태누수 함정 회피).
OUT=${1:-/data/jtm/synth_out/dataset_v3}
N=${2:-2000}
BP=/data/jtm/blenderproc_venv/bin/blenderproc
LOG=${OUT}.log
mkdir -p "$OUT"
echo "[RUNNER v3] start $(date) N=$N OUT=$OUT" >> "$LOG"
for ((i=0; i<N; i++)); do
  [ -f "$OUT/$i.hdf5" ] && continue   # resume
  $BP run /home/jtm/kaist_project/synth/gen_one_v3.py \
     --blender-install-path /data/jtm/blender -- "$i" "$OUT" \
     >> "$LOG" 2>&1
done
echo "[RUNNER v3] ALL DONE $N scenes $(date)" >> "$LOG"
