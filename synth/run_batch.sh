#!/bin/bash
# scene마다 깨끗한 프로세스로 gen_one.py 호출 (루프-in-스크립트 상태누수 회피)
OUT=/data/jtm/synth_out/dataset_v1
N=2000
BP=/data/jtm/blenderproc_venv/bin/blenderproc
mkdir -p "$OUT"
for ((i=0; i<N; i++)); do
  # 이미 있으면 건너뜀(resume)
  [ -f "$OUT/$i.hdf5" ] && continue
  $BP run /home/jtm/kaist_project/synth/gen_one.py \
     --blender-install-path /data/jtm/blender -- "$i" "$OUT" \
     >> /data/jtm/synth_out/batch.log 2>&1
done
echo "[RUNNER] ALL DONE $N scenes" >> /data/jtm/synth_out/batch.log
