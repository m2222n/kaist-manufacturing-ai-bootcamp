# Depth-only 2D VQ Detector — provided NPZ format

이 코드는 제공된 합성 데이터셋 포맷에 맞춘 **full-scene depth-only query detector + CAD VQ head**입니다.

핵심 원칙:

```text
training input  = scene-level depth only
training target = inst_id / category_id / meta.stl / mask derived from inst_id
inference input = scene-level depth only
inference output = bbox, predicted mask, class_id, cad_id
```

GT mask는 모델 입력으로 들어가지 않습니다. `inst_id`에서 instance mask를 만들고, 그 mask는 Hungarian matching과 mask loss에만 사용합니다.

---

## 1. Supported dataset layout

제공된 포맷을 바로 읽습니다.

```text
dataset_root/
 ├─ npz/
 │   ├─ scene_00000.npz
 │   ├─ scene_00001.npz
 │   └─ ...
 ├─ crops/
 │   ├─ scene00000_inst01_cat03.npz
 │   └─ ...
 └─ vis/
     └─ ...
```

`npz/scene_XXXXX.npz` keys:

```text
depth       : (512,512) float32, meter, background NaN
inst_id     : (512,512) int32, instance id 1..n, background 0
category_id : (512,512) int32, class id 1..27, background 0
meta        : JSON string
```

`meta.instances[inst_id]` is used for:

```text
category_id
stl
quat_wxyz
euler_zyx_deg
```

The main detector trains on **full scene npz** because inference must detect multiple objects in one depth map. Crop npz files are not used as detector input. They are useful for auxiliary crop-level CAD retrieval pretraining or debugging.

---

## 2. Architecture

```text
Scene depth
  ↓
depth-only input channels
  ↓
Depth CNN backbone
  ↓
FPN pixel decoder
  ├─ mask feature map
  └─ transformer memory feature
  ↓
Object query decoder
  ↓
K object tokens
  ├─ class head
  ├─ box head
  ├─ mask head
  └─ CAD VQ head
```

CAD VQ head:

```text
object token h_i
  ↓ MLP + L2 normalize
2D object embedding e_i
  ↓ cosine similarity against frozen CAD codebook
cad_logits_i = scale * e_i @ Z_cad.T
```

The CAD codebook comes from the pretrained 3D CAD encoder memory bank:

```text
cad_memory_bank.npz
 ├─ embeddings : (num_cads, embed_dim)
 └─ cad_ids    : (num_cads,)
```

The dataset maps `meta.instances[*].stl` to `cad_ids`. Matching is robust to these variants:

```text
03_sol_block_front.stl
03_sol_block_front
/path/to/03_sol_block_front.stl
```

---

## 3. Install

```bash
cd depth_vq_detector_format_code
pip install -r requirements.txt
```

---

## 4. Inspect provided data

```bash
python tools/inspect_dataset_format.py \
  --root /path/to/dataset_root \
  --max 3
```

This prints scene keys, NaN counts, visible instance ids, and crop keys.

---

## 5. Training

Detector + CAD VQ joint training:

```bash
python train_depth_vq_detector.py \
  --data_root /path/to/dataset_root \
  --cad_memory /path/to/cad_memory_bank.npz \
  --out_dir ./runs/depth_vq_detector \
  --num_classes 27 \
  --input_mode zv \
  --epochs 100 \
  --batch_size 4 \
  --stage joint
```

Detector-only warmup:

```bash
python train_depth_vq_detector.py \
  --data_root /path/to/dataset_root \
  --out_dir ./runs/depth_detector_warmup \
  --num_classes 27 \
  --input_mode zv \
  --stage det \
  --epochs 30
```

Debug on one scene:

```bash
python train_depth_vq_detector.py \
  --scene_npz /path/to/dataset_root/npz/scene_00000.npz \
  --cad_memory /path/to/cad_memory_bank.npz \
  --out_dir ./runs/debug_one_scene \
  --num_classes 27 \
  --input_mode zv \
  --image_size 128,128 \
  --epochs 1 \
  --batch_size 1 \
  --num_queries 16 \
  --hidden_dim 32 \
  --backbone_dim 16 \
  --decoder_layers 1 \
  --nheads 4 \
  --stage joint
```

Important defaults for the provided dataset:

```text
--num_classes 27
--label_offset 1   # category_id is 1..27; model labels are 0..26
--input_mode zv    # normalized depth + valid-depth mask
```

---

## 6. Inference

```bash
python infer_depth_vq_detector.py \
  --checkpoint ./runs/depth_vq_detector/best.pt \
  --scene_npz /path/to/dataset_root/npz/scene_00000.npz \
  --out_dir ./pred_scene_00000 \
  --score_thresh 0.25
```

Output:

```text
pred_scene_00000/
 ├─ predictions.json
 └─ predicted_masks.npz
```

`predictions.json` contains:

```json
{
  "scene_id": "scene_00000",
  "predictions": [
    {
      "query_index": 3,
      "score": 0.91,
      "class_id": 10,
      "bbox_xyxy": [x1, y1, x2, y2],
      "mask_area": 1234,
      "cad_index": 9,
      "cad_score": 0.88,
      "cad_id": "13_x2_bcf8ccb4.stl"
    }
  ]
}
```

`class_id` is converted back to the original dataset convention by adding `label_offset`, so the output class ids are 1..27 by default.

---

## 7. Input modes

```text
z     : normalized depth only                                      C=1
zv    : normalized depth + valid mask                              C=2
xyzv  : pseudo/camera XYZ + valid mask                             C=4
xyznv : pseudo/camera XYZ + surface normal XYZ + valid mask         C=7
```

For the provided dataset, `depth` uses NaN background. The code converts this into:

```text
valid_mask = isfinite(depth) & depth > 0
invalid depth channel value = 0
```

Start with `zv`. Move to `xyzv` or `xyznv` only if camera intrinsics are available or the pseudo-XYZ representation proves useful.

---

## 8. Losses

```text
L =
    L_class
  + L_box
  + L_giou
  + L_mask_BCE
  + L_mask_Dice
  + L_cad_CE
  + L_cad_alignment
```

CAD losses are applied only to matched query/GT pairs whose `stl` is found in the CAD memory bank.

---

## 9. Files

```text
depth_vq_detector/
 ├─ dataset.py            # provided npz format loader
 ├─ depth_preprocess.py   # NaN-safe depth handling and input channels
 ├─ model.py              # depth query detector + CAD VQ head
 ├─ matcher.py            # Hungarian matching
 ├─ losses.py             # DETR/Mask-style losses + CAD VQ loss
 ├─ postprocess.py        # inference post-processing
 └─ geometry.py

tools/
 ├─ build_scene_manifest.py
 └─ inspect_dataset_format.py

train_depth_vq_detector.py
infer_depth_vq_detector.py
```
