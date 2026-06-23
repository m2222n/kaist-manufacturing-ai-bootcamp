# CADENCE Command README (조교 첨부, 2026-06-23 수신)
# STL → point cloud → 3D CAD encoder → 2D depth VQ detector → split eval → inference
# 전체 명령 가이드. 실행 위치 = model/ (조교 repo 기준 repo root).
# 우리 환경 적용: A100 /workspace/cadence/Mentoring 에서 --num_workers 0 으로 실행.
# 데이터 root = /workspace/cadence/data/2d_dataset, STL = /workspace/cadence/data/stl_folder
# 상세 단계별 명령은 INTEGRATION.md + 조교 원본 README 3종(3d/pointcloud/2d) 참고.
