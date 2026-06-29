#!/usr/bin/env python3
"""labelme JSON(실증 100장 라벨링 결과) → 합성 dataset_2denc 포맷으로 변환.

입력:
  - labelme JSON들 (label_png 위에서 라벨링, polygon. label = STL 이름)
  - 원본 npy (848x480 uint16 mm)  ← raw depth 원천
  - 그룹 정답지 JSON (category_id 매핑)
출력:
  - scene npz: depth(512x512 float32 m, 배경 NaN) / inst_id / category_id / meta
    (합성 dataset_2denc/npz/ 포맷과 동일)

⚠️ 조교 포맷 컨펌 대기 항목 = 아래 CONFIG의 빈칸:
  - TARGET_RES: 합성은 512. 조교 답변으로 확정 (512 가정).
  - DEPTH_UNIT: 합성은 meter. (mm→m = /1000 가정).
  - BG_VALUE: 합성은 NaN. (배경 NaN 가정).
  ↑ 셋 다 합성과 동일하게 가정해뒀음. 조교가 다른 값 주면 여기만 수정.

⚠️ 좌표 스케일: label_png(1696x960) = npy(848x480) x2 → polygon 좌표 /2 해서 npy 매핑.
"""
import json, glob, os, argparse
import numpy as np

# ---- CONFIG (조교 컨펌 시 여기만 수정) -------------------------------
TARGET_RES = 512        # 합성과 동일 (조교 컨펌 대기)
DEPTH_DIV  = 1000.0     # mm → m
BG_VALUE   = np.nan     # 배경 = NaN (합성과 동일)
PNG_SCALE  = 2.0        # label_png = npy x2 → polygon 좌표 /PNG_SCALE
# --------------------------------------------------------------------

def poly_to_mask(points, h, w, scale):
    """labelme polygon(이미지 좌표) → npy 해상도 mask. (PIL ImageDraw 사용)"""
    from PIL import Image, ImageDraw
    img = Image.new("L", (w, h), 0)
    pts = [(x / scale, y / scale) for x, y in points]
    ImageDraw.Draw(img).polygon(pts, outline=1, fill=1)
    return np.array(img, dtype=bool)

def convert_one(json_path, npy_dir, name2cid):
    base = os.path.splitext(os.path.basename(json_path))[0]  # shot_XXX_gN
    npy_path = os.path.join(npy_dir, base + ".npy")
    if not os.path.exists(npy_path):
        print(f"  ⚠️ npy 없음: {base}"); return None
    raw = np.load(npy_path).astype(np.float32)   # (480,848) mm
    h, w = raw.shape
    lj = json.load(open(json_path, encoding="utf-8"))

    inst_id = np.zeros((h, w), np.int32)
    cat_id  = np.zeros((h, w), np.int32)
    instances = {}
    for i, shape in enumerate(lj.get("shapes", []), start=1):
        label = shape["label"].replace(".stl", "")
        if label not in name2cid:
            print(f"  ⚠️ {base}: 미지 라벨 '{label}' 건너뜀"); continue
        m = poly_to_mask(shape["points"], h, w, PNG_SCALE)
        # 유효 depth(부품)만 — polygon 안이라도 depth 0(배경 dropout)은 제외
        m = m & (raw > 0)
        inst_id[m] = i
        cat_id[m]  = name2cid[label]
        instances[str(i)] = {"category_id": name2cid[label], "stl": label + ".stl"}

    # depth: 부품 픽셀만 m 단위, 나머지 배경 NaN
    depth = np.full((h, w), BG_VALUE, np.float32)
    part = inst_id > 0
    depth[part] = raw[part] / DEPTH_DIV

    # 해상도 변환 (848x480 → 512x512). 합성과 맞추려면 resize 필요.
    # ⚠️ depth/label은 nearest로 (보간 시 라벨 섞임 방지). aspect 다름(848:480 vs 512:512) → 조교와 협의 필요.
    # 일단 nearest resize. 조교가 crop/pad 원하면 여기 수정.
    depth_r, inst_r, cat_r = resize_nn(depth, inst_id, cat_id, TARGET_RES)

    meta = {
        "shot": base, "group": base.split("_")[-1],
        "source": "real_capture (Blaze), labelme polygon",
        "orig_shape": [h, w], "target_res": TARGET_RES,
        "depth_units": "meters", "background": "NaN",
        "instances": instances,
        "note": "픽셀↔부품 = labelme 수동 라벨링. 6DoF pose 없음(실측 자세 미측정).",
    }
    return {"depth": depth_r, "inst_id": inst_r, "category_id": cat_r,
            "meta": json.dumps(meta, ensure_ascii=False)}

def resize_nn(depth, inst, cat, res):
    """nearest-neighbor resize to (res,res). depth NaN 보존."""
    from PIL import Image
    h, w = depth.shape
    # inst/cat = nearest. depth = NaN을 0으로 치환 후 nearest, 다시 마스크로 NaN 복원
    inst_r = np.array(Image.fromarray(inst).resize((res, res), Image.NEAREST))
    cat_r  = np.array(Image.fromarray(cat).resize((res, res), Image.NEAREST))
    d0 = np.where(np.isnan(depth), 0, depth).astype(np.float32)
    d_r = np.array(Image.fromarray(d0).resize((res, res), Image.NEAREST))
    d_r = np.where(inst_r > 0, d_r, np.nan).astype(np.float32)
    return d_r, inst_r.astype(np.int32), cat_r.astype(np.int32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="labelme JSON 폴더")
    ap.add_argument("--npy_dir", default="/data/jtm/synth_out/real_capture100/npy")
    ap.add_argument("--labels", default="/home/jtm/kaist_project/docs/real_capture100_group_labels.json")
    ap.add_argument("--out", default="/data/jtm/synth_out/real_capture100/synthformat")
    args = ap.parse_args()

    name2cid = json.load(open(args.labels, encoding="utf-8"))["category_id"]
    os.makedirs(args.out, exist_ok=True)
    jsons = sorted(glob.glob(os.path.join(args.json_dir, "*.json")))
    print(f"라벨 JSON {len(jsons)}개 변환 시작 (target {TARGET_RES}px, depth/{DEPTH_DIV:.0f}, bg NaN)")
    ok = 0
    for jp in jsons:
        r = convert_one(jp, args.npy_dir, name2cid)
        if r is None: continue
        base = os.path.splitext(os.path.basename(jp))[0]
        np.savez_compressed(os.path.join(args.out, base + ".npz"), **r)
        ok += 1
    print(f"✅ {ok}/{len(jsons)} 변환 완료 → {args.out}")

if __name__ == "__main__":
    main()
