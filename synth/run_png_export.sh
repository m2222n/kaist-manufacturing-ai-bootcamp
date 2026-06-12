#!/bin/bash
# 기존 3데이터셋(v1/v2/v2_noisy) 각 2000장을 4패널 PNG로 추출.
# Drive 업로드용. v2_noisy는 depth 자리에 depth_noisy 사용.
PY=/data/jtm/blenderproc_venv/bin/python
EX=/home/jtm/kaist_project/synth/extract_all.py
BASE=/data/jtm/synth_out/png_export
mkdir -p "$BASE"

echo "[1/3] v1 (plain gray)  $(date)"
$PY $EX /data/jtm/synth_out/dataset_v1 "$BASE/v1_plain_gray_2000"

echo "[2/3] v2 (domain rand)  $(date)"
$PY $EX /data/jtm/synth_out/dataset_v2 "$BASE/v2_domainrand_2000"

echo "[3/3] v2_noisy (depth noise)  $(date)"
$PY $EX --depth-key depth_noisy /data/jtm/synth_out/dataset_v2_noisy "$BASE/v2_domainrand_depthnoise_2000"

echo "[PNG EXPORT] ALL DONE  $(date)"
