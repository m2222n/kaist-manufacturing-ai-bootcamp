import blenderproc as bproc
# KAIST 3주차 — 2D 인코더 학습용 numpy 데이터 생성 (6/16 미팅 지시 반영)
#
# 6/16 KAIST 미팅 (조교 모델 아키텍처 확정):
#   조교님이 3D 인코더(STL→pointcloud 패치 관계성) + 2D 인코더(DepthMap+label+기울기 메타)를
#   latent space에서 복원하는 모델을 직접 개발. 우리는 2D 인코더 학습용 raw 데이터를 준비.
#
#   조교 요구 (이 스크립트가 충족):
#     - 이미지로 뽑기 전 numpy(raw)로 저장
#     - label별로 마스크가 따로 저장
#     - 한 장 = 마스크 + 메타데이터(CAD 도면상 어느 정도 기울었는지) + label
#     - 배경은 없애고 Depth Map으로
#
#   6/16 사용자 확정 저장 스펙:
#     - 저장 단위 = 둘 다: ① scene 전체 npz  ② 부품별 crop npz
#     - 기울기 메타 = 쿼터니언(w,x,y,z) + 오일러(deg) 둘 다 (STL canonical→현재 자세 회전)
#     - 배경 = 부품 픽셀만 남기고 나머지 전부 NaN
#     - 물리 적재(박스/책상)는 v3 그대로 유지(occlusion 분포 보존), 저장 단계에서 마스킹
#
# 물리/카메라/원점/조명은 gen_one_v3.py 검증본 그대로. 신규 = per-instance 자세 추출 + numpy 저장.
#
# 호출: blenderproc run gen_one_2denc.py --blender-install-path /data/jtm/blender -- <scene_idx> <out_dir>
import numpy as np
import glob, os, sys, math, json

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
scene_idx = int(argv[0])
OUT_DIR   = argv[1] if len(argv) > 1 else "/data/jtm/synth_out/dataset_2denc"

STL_DIR = "/home/jtm/kaist_render/stl"
BIN_HALF, WALL_H, WALL_T = 0.15, 0.15, 0.03
DROP_R = 0.05
SEED = 1234

PART_GRAY = [0.5, 0.5, 0.5, 1.0]
PART_ROUGHNESS = 0.5

NPZ_DIR  = os.path.join(OUT_DIR, "npz")          # scene 전체 npz
CROP_DIR = os.path.join(OUT_DIR, "crops")        # 부품별 crop npz
os.makedirs(NPZ_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)


def rotmat_to_quat(R):
    """3x3 회전행렬 -> 쿼터니언 (w,x,y,z). numpy만 사용 (scipy 없음)."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def rotmat_to_euler_zyx_deg(R):
    """3x3 -> 오일러 ZYX (yaw-pitch-roll) deg. 사람이 읽기용."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z]).astype(np.float64)


bproc.init()
rng = np.random.default_rng(SEED + scene_idx)
all_stl = sorted(glob.glob(os.path.join(STL_DIR, "*.stl")))

bg_kind = int(rng.integers(0, 2))

gray_mat = bproc.material.create("part_gray")
gray_mat.set_principled_shader_value("Base Color", PART_GRAY)
gray_mat.set_principled_shader_value("Roughness", PART_ROUGHNESS)
gray_mat.set_principled_shader_value("Metallic", 0.0)

if bg_kind == 0:
    g = bproc.object.create_primitive("CUBE", scale=[BIN_HALF, BIN_HALF, WALL_T])
    g.set_location([0, 0, -WALL_T])
    g.enable_rigidbody(active=False, collision_shape="BOX")
    g.set_cp("category_id", 0)
    clear_mat = bproc.material.create("clear_plastic")
    clear_mat.set_principled_shader_value("Base Color", [0.85, 0.92, 0.92, 1.0])
    clear_mat.set_principled_shader_value("Roughness", float(rng.uniform(0.15, 0.35)))
    clear_mat.set_principled_shader_value("Transmission Weight", 1.0)
    clear_mat.set_principled_shader_value("IOR", float(rng.uniform(1.40, 1.50)))
    clear_mat.set_principled_shader_value("Metallic", 0.0)
    g.replace_materials(clear_mat)
    thin_t = 0.006
    for x, y, span in [(BIN_HALF, 0, "y"), (-BIN_HALF, 0, "y"), (0, BIN_HALF, "x"), (0, -BIN_HALF, "x")]:
        sc = [thin_t, BIN_HALF, WALL_H / 2] if span == "y" else [BIN_HALF, thin_t, WALL_H / 2]
        w = bproc.object.create_primitive("CUBE", scale=sc)
        w.set_location([x, y, WALL_H / 2])
        w.enable_rigidbody(active=False, collision_shape="BOX")
        w.set_cp("category_id", 0)
        w.replace_materials(clear_mat)
    min_p, max_p = 8, 15
    drop_r = DROP_R
    z0, z_step = 0.05, 0.022
else:
    desk_half = BIN_HALF * 5
    g = bproc.object.create_primitive("CUBE", scale=[desk_half, desk_half, WALL_T])
    g.set_location([0, 0, -WALL_T])
    g.enable_rigidbody(active=False, collision_shape="BOX")
    g.set_cp("category_id", 0)
    white_mat = bproc.material.create("white_desk")
    v = float(rng.uniform(0.82, 0.95))
    white_mat.set_principled_shader_value("Base Color", [v, v, v, 1.0])
    white_mat.set_principled_shader_value("Roughness", float(rng.uniform(0.4, 0.8)))
    white_mat.set_principled_shader_value("Metallic", 0.0)
    g.replace_materials(white_mat)
    min_p, max_p = 4, 8
    drop_r = DROP_R * 1.2
    z0, z_step = 0.03, 0.005

# --- 부품 랜덤 적재 (회색 단색). instance_id <-> (category_id, stl명) 매핑 추적 ---
# instance_segmaps는 BlenderProc가 객체별 고유 id를 자동 부여(보통 등장 순서). 우리는
# 물리 후 각 부품의 world matrix(회전)로 "CAD canonical -> 현재 자세" 회전을 읽는다.
# STL 로드 직후 회전 적용 전 자세 = canonical(축정렬). 따라서 물리 후 R_3x3 = canonical->current.
n_parts = int(rng.integers(min_p, max_p + 1))
parts = []   # (obj, category_id, stl_name)
for k in range(n_parts):
    cid = int(rng.integers(0, len(all_stl)))
    stl_name = os.path.basename(all_stl[cid])
    for obj in bproc.loader.load_obj(all_stl[cid]):
        obj.set_scale([0.001, 0.001, 0.001])
        obj.set_origin(mode="CENTER_OF_VOLUME")
        obj.set_location([float(rng.uniform(-drop_r, drop_r)),
                          float(rng.uniform(-drop_r, drop_r)),
                          float(z0 + k * z_step)])
        obj.set_rotation_euler(rng.uniform(0, 2 * np.pi, size=3).tolist())
        obj.enable_rigidbody(active=True, collision_shape="CONVEX_HULL")
        obj.set_cp("category_id", cid + 1)
        obj.replace_materials(gray_mat)
        parts.append((obj, cid + 1, stl_name))

bproc.object.simulate_physics_and_fix_final_poses(
    min_simulation_time=2, max_simulation_time=10, check_object_interval=1)

# --- 물리 후 per-instance 자세 추출 (canonical CAD -> 현재) ---
# BlenderProc instance_segmaps id 부여: render 후 instance_attribute_maps에 idx<->obj 매핑이 나온다.
# 안전하게: 각 obj의 custom prop으로 우리만의 inst_id를 박아 render 후 instance map과 대조.
for i, (obj, cid, stl_name) in enumerate(parts):
    obj.set_cp("inst_id", i + 1)

# --- 조명 ---
n_lights = int(rng.integers(1, 4))
e_lo, e_hi = (30, 90) if bg_kind == 0 else (25, 70)
for _ in range(n_lights):
    light = bproc.types.Light(); light.set_type("POINT")
    light.set_location([float(rng.uniform(-0.25, 0.25)), float(rng.uniform(-0.25, 0.25)), float(rng.uniform(0.3, 0.6))])
    light.set_energy(float(rng.uniform(e_lo, e_hi)))

# --- 카메라 (v1 검증 수직 down-view) ---
cam_h = float(rng.uniform(0.36, 0.44))
RES = 512
bproc.camera.set_resolution(RES, RES)
fov = 2 * math.atan((BIN_HALF * 1.4) / cam_h)
bproc.camera.set_intrinsics_from_blender_params(lens=fov, lens_unit="FOV")
bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(
    [float(rng.uniform(-0.015, 0.015)), float(rng.uniform(-0.015, 0.015)), cam_h], [0, 0, 0]))

# --- 멀티모달 렌더 (depth + instance/category seg + inst_id seg) ---
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(
    map_by=["category_id", "instance", "inst_id"],
    default_values={"category_id": 0, "inst_id": 0})
data = bproc.renderer.render()

depth = np.array(data["depth"][0]).astype(np.float32)            # (H,W) meters
cat = np.array(data["category_id_segmaps"][0]).astype(np.int32)  # 부품 cid+1, 배경 0
inst_id = np.array(data["inst_id_segmaps"][0]).astype(np.int32)  # 우리가 박은 inst_id(1..n), 배경 0

# 부품별 자세 메타 (inst_id -> {category_id, stl, quat, euler})
inst_meta = {}
for i, (obj, cid, stl_name) in enumerate(parts):
    iid = i + 1
    R = np.array(obj.get_local2world_mat())[:3, :3]
    # 스케일 제거 (0.001 스케일이 행렬에 섞여 있으면 정규화)
    for c in range(3):
        n = np.linalg.norm(R[:, c])
        if n > 1e-9:
            R[:, c] = R[:, c] / n
    q = rotmat_to_quat(R)
    e = rotmat_to_euler_zyx_deg(R)
    inst_meta[iid] = {
        "category_id": int(cid),
        "stl": stl_name,
        "quat_wxyz": [float(v) for v in q],
        "euler_zyx_deg": [float(v) for v in e],
    }

# --- 배경 제거: 부품 픽셀(inst_id>0)만 남기고 나머지 depth = NaN ---
fg = inst_id > 0
depth_masked = np.where(fg, depth, np.nan).astype(np.float32)

# 화면에 실제로 보이는 인스턴스만 (occlusion으로 안 보이는 부품은 제외)
visible_ids = sorted(int(v) for v in np.unique(inst_id) if v > 0)

# --- ① scene 전체 npz ---
bg_name = "clear_box" if bg_kind == 0 else "white_desk"
scene_meta = {
    "scene_idx": scene_idx,
    "bg_kind": bg_name,
    "resolution": [RES, RES],
    "n_parts_dropped": n_parts,
    "visible_inst_ids": visible_ids,
    "instances": {str(k): inst_meta[k] for k in visible_ids},  # 보이는 것만
    "euler_convention": "ZYX_deg_intrinsic",
    "quat_convention": "wxyz",
    "rotation_meaning": "STL canonical(axis-aligned at load) -> post-physics current pose",
    "depth_units": "meters",
    "background": "non-part pixels set to NaN",
}
scene_path = os.path.join(NPZ_DIR, f"scene_{scene_idx:05d}.npz")
np.savez_compressed(
    scene_path,
    depth=depth_masked,            # (H,W) float32, 배경 NaN
    inst_id=inst_id,               # (H,W) int32, 부품 inst_id (1..n), 배경 0
    category_id=cat,               # (H,W) int32, 부품 cid+1, 배경 0
    meta=json.dumps(scene_meta),   # str
)

# --- ② 부품별 crop npz (보이는 인스턴스마다 1파일) ---
for iid in visible_ids:
    ys, xs = np.where(inst_id == iid)
    if ys.size == 0:
        continue
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop_mask = (inst_id[y0:y1, x0:x1] == iid)
    crop_depth = np.where(crop_mask, depth[y0:y1, x0:x1], np.nan).astype(np.float32)
    m = inst_meta[iid]
    crop_path = os.path.join(CROP_DIR, f"scene{scene_idx:05d}_inst{iid:02d}_cat{m['category_id']:02d}.npz")
    np.savez_compressed(
        crop_path,
        depth=crop_depth,                                  # (h,w) float32, 부품 픽셀만, 배경 NaN
        mask=crop_mask,                                    # (h,w) bool
        label=np.int32(m["category_id"]),                 # 부품 클래스
        quat_wxyz=np.array(m["quat_wxyz"], np.float64),    # 기울기 (CAD canonical -> 현재)
        euler_zyx_deg=np.array(m["euler_zyx_deg"], np.float64),
        bbox_yxyx=np.array([y0, x0, y1, x1], np.int32),
        stl=m["stl"],
    )

print(f"[2DENC] scene {scene_idx}: bg={bg_name} dropped={n_parts} visible={len(visible_ids)} "
      f"crops={len(visible_ids)} -> {scene_path}", flush=True)
