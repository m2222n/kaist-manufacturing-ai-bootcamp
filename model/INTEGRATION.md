# 모델 코드 통합 메모 (CADENCE 파이프라인)

> 이 `model/` 폴더는 **조교(임학수) 제공 코드**를 우리 KAIST 프로젝트에 통합한 것이다.
> 출처: `github.com/LimHaksoo/Mentoring` (원본 커밋 `9972f2a`, 2026-06-23 수신).
> 모델 설계·학습 = 조교 / 데이터 구축·비교 실험 = 팀6 (6/19 미팅 역할 분담).

## 무엇인가 (한 줄)

CAD STL만으로 합성 데이터를 만들어, **Depth 한 장 → 부품 검출(box) + 분할(mask) + 분류(class) + CAD 식별(어느 부품인지)** 까지 하는 파이프라인.

## 파이프라인 (STEP 순서)

```
STL 27종 ─(stl_to_pointcloud_dataset.py)→ point cloud(8192점, xyz+normal)
         ─(train_cad_encoder.py: PointNet++)→ 3D CAD encoder
         ─(build_cad_memory_bank.py)→ CAD memory bank (27개 형상 임베딩 codebook)
                                                  │ frozen
합성 scene depth ─(train_depth_vq_detector.py)───┘
   2D Depth VQ Detector (DETR/Mask2Former 계열)
     stage=det  : class+box+mask 먼저 (warmup)
     stage=joint: + CAD VQ alignment (codebook과 cosine)
         ─(eval_depth_vq_detector.py)→ test 정량 평가
         ─(infer_depth_vq_detector.py)→ depth-only 추론 + 시각화
```

## 우리 데이터 연결 (중요)

- 학습/평가는 **6000이 아니라 A100**(`<A100_HOST>:<PORT>`)에서 수행. 환경 = `/opt/conda/bin/python` (torch 2.1 / CUDA 11.8).
- A100 작업 루트: `/workspace/cadence/`
  - `data/2d_dataset/`  = 우리 2D 인코더 1000장 (`npz/`, `crops/`) — 원본은 6000 `/data/jtm/synth_out/dataset_2denc/`
  - `data/stl_folder/`  = STL 27종 — 원본은 6000 `~/kaist_render/stl/`
  - `data/pc_dataset/`  = STL→pointcloud 산출(27개 npz/ply + manifest)
  - `data/2d_dataset/splits/` = scene 단위 800/100/100 (train/val/test)
  - `runs/`             = 학습 산출 (cad_pointnet2 / depth_detector_warmup_split / depth_vq_detector_split)
  - `opendata/`         = BOP ITODD + IC-BIN (비교 실험용)
- ⚠️ `/dev/shm`이 64MB라 detector 학습 시 **`--num_workers 0`** 필수 (안 그러면 DataLoader OOM). → `reference_aica_a100.md`
- 데이터 포맷 정합: 조교 코드의 scene npz 기대 키(`depth/inst_id/category_id/meta`)가 우리 1000장과 **완전 일치**. 조교가 우리 5장 샘플 포맷대로 코드를 작성함.

## 핵심 인자 (우리 데이터 기준)

```
--num_classes 27       # 부품 27종
--label_offset 1       # category_id 1..27 → 모델 라벨 0..26
--input_mode zv        # 정규화 depth + valid mask (배경 NaN→valid=False)
--num_workers 0        # A100 shm 제약
```

## 실행 순서 = `README_COMMANDS.md` 참고 (조교 첨부 전체 명령 가이드)

- 3D: `README_3d_encoder.md`
- pointcloud: `README_pointcloud_pipeline.md`
- 2D detector: `README_2d_vq_detector.md`

## 통합 시 우리가 한 일 (6/23)

1. 전체 파이프라인을 A100에서 처음부터 끝까지 직접 구동 (split→pc→3D enc→memory bank→detector warmup→VQ joint→eval).
2. 코드 리뷰 → `docs/조교_코드리뷰_0623.md`.
3. 우리 데이터 포맷이 코드와 정합함을 inspect 도구로 검증.
4. ⚠️ **성능 최적화(별도, 원본 미반영)**: `depth_vq_detector/dataset.py`의 `_boxes_from_mask`가 `torch.where`로 박스 추출 → 부품 많은 장면(10~14개)에서 장면당 5.8초 병목. 운영 시 **numpy 행/열 투영**(`np.any`)으로 교체하면 0.005초(결과 동일). 여기 `model/`엔 **원본 보존**(조교 코드 훼손 X), 최적화 diff는 멘토께 별도 공유 예정.
