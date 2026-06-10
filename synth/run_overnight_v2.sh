#!/bin/bash
# 밤샘 배치: v2를 2000장까지 생성(기존 300장 resume) → 끝나면 depth_noise 후처리 자동 연결.
#   1단계: gen_one_v2.py로 0~1999 (이미 있으면 skip = 301~1999만 새로 렌더)
#   2단계: depth_noise.py로 dataset_v2 전체 → dataset_v2_noisy (depth_noisy 채널)
# (set -e 안 씀: 한 장 실패해도 배치는 계속 진행)
OUT=/data/jtm/synth_out/dataset_v2
NOISY=/data/jtm/synth_out/dataset_v2_noisy
CC=/data/jtm/cc_textures
N=2000
BP=/data/jtm/blenderproc_venv/bin/blenderproc
PY=/data/jtm/blenderproc_venv/bin/python
LOG=/data/jtm/synth_out/overnight_v2.log

echo "[OVERNIGHT] 시작 $(cat /proc/uptime | cut -d' ' -f1)s uptime, 목표 ${N}장" >> "$LOG"
mkdir -p "$OUT"
for ((i=0; i<N; i++)); do
  [ -f "$OUT/$i.hdf5" ] && continue   # resume: 기존 300장 + 이미 만든 것 skip
  $BP run /home/jtm/kaist_project/synth/gen_one_v2.py \
     --blender-install-path /data/jtm/blender -- "$i" "$OUT" "$CC" \
     >> "$LOG" 2>&1
done
echo "[OVERNIGHT] 1단계 생성 완료: $(ls $OUT/*.hdf5 | wc -l)장" >> "$LOG"

# 2단계: depth 노이즈 후처리 (기존 noisy 폴더 갈아엎고 2000장 전체 재생성)
rm -rf "$NOISY"
$PY /home/jtm/kaist_project/synth/depth_noise.py "$OUT" "$NOISY" >> "$LOG" 2>&1
echo "[OVERNIGHT] 2단계 depth_noise 완료: $(ls $NOISY/*.hdf5 | wc -l)장" >> "$LOG"
echo "[OVERNIGHT] ALL DONE" >> "$LOG"
