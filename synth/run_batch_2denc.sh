#!/bin/bash
# 2D 인코더 학습용 numpy 데이터셋 배치 (6/16 미팅 지시).
# scene마다 깨끗한 프로세스로 gen_one_2denc.py 호출 (상태누수 함정 회피, v3 러너와 동일 구조).
# scene npz = npz/scene_NNNNN.npz 존재 여부로 resume.
OUT=${1:-/data/jtm/synth_out/dataset_2denc}
N=${2:-1000}
BP=/data/jtm/blenderproc_venv/bin/blenderproc
LOG=${OUT}.log
mkdir -p "$OUT/npz" "$OUT/crops"
echo "[RUNNER 2denc] start $(date) N=$N OUT=$OUT" >> "$LOG"
for ((i=0; i<N; i++)); do
  printf -v PADDED "%05d" "$i"
  [ -f "$OUT/npz/scene_${PADDED}.npz" ] && continue   # resume
  $BP run /home/jtm/kaist_project/synth/gen_one_2denc.py \
     --blender-install-path /data/jtm/blender -- "$i" "$OUT" \
     >> "$LOG" 2>&1
done
NPZ=$(ls "$OUT/npz/" 2>/dev/null | wc -l)
CROP=$(ls "$OUT/crops/" 2>/dev/null | wc -l)
echo "[RUNNER 2denc] ALL DONE $(date) scene=$NPZ crop=$CROP" >> "$LOG"
