#!/bin/bash
# v2 배치 — 도메인 랜덤화(배경 텍스처/재질/조명). scene마다 깨끗한 프로세스로 gen_one_v2.py 호출.
OUT=/data/jtm/synth_out/dataset_v2
CC=/data/jtm/cc_textures
N=300
BP=/data/jtm/blenderproc_venv/bin/blenderproc
LOG=/data/jtm/synth_out/batch_v2.log
mkdir -p "$OUT"
for ((i=0; i<N; i++)); do
  [ -f "$OUT/$i.hdf5" ] && continue   # resume
  $BP run /home/jtm/kaist_project/synth/gen_one_v2.py \
     --blender-install-path /data/jtm/blender -- "$i" "$OUT" "$CC" \
     >> "$LOG" 2>&1
done
echo "[RUNNER] ALL DONE $N scenes" >> "$LOG"
