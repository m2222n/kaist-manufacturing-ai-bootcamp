import blenderproc as bproc
# KAIST 2주차 — 합성데이터 "1 scene" 생성 v2 (배경/재질 다양성 보강)
# v1(gen_one.py) 대비 추가:
#   - 빈 바닥/벽에 CC0 PBR 텍스처 랜덤 적용 (단색 -> 실사 배경, 교수님 "배경 실사" 피드백)
#   - 부품 재질 색/거칠기/메탈릭 랜덤 (회색 단일 -> 다양, 조교 "질감 후보정" 피드백)
#   - 조명 다광원 + 강도 랜덤
# 물리/카메라/원점 처리는 v1에서 검증된 것 그대로 유지 (함정 3개 회피).
#
# 호출: blenderproc run gen_one_v2.py --blender-install-path /data/jtm/blender -- <scene_idx> <out_dir> [cc_dir]
import numpy as np
import glob, os, sys, math

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
scene_idx = int(argv[0])
OUT_DIR   = argv[1] if len(argv) > 1 else "/data/jtm/synth_out/dataset_v2"
CC_DIR    = argv[2] if len(argv) > 2 else "/data/jtm/cc_textures"

STL_DIR = "/home/jtm/kaist_render/stl"
MIN_PARTS, MAX_PARTS = 5, 15
BIN_HALF, WALL_H, WALL_T = 0.15, 0.15, 0.03
DROP_R = 0.05
SEED = 1234

bproc.init()
rng = np.random.default_rng(SEED + scene_idx)
all_stl = sorted(glob.glob(os.path.join(STL_DIR, "*.stl")))
os.makedirs(OUT_DIR, exist_ok=True)

# --- CC0 PBR 텍스처 로드 (있으면) ---
cc_mats = []
if os.path.isdir(CC_DIR) and len(os.listdir(CC_DIR)) > 0:
    try:
        cc_mats = bproc.loader.load_ccmaterials(CC_DIR)
    except Exception as e:
        print(f"[V2] cc_materials load 실패(무시): {e}", flush=True)

def random_cc_mat():
    return cc_mats[int(rng.integers(0, len(cc_mats)))] if cc_mats else None

# --- 빈(두꺼운 CUBE 바닥+4벽), 바닥/벽에 텍스처 ---
bin_objs = []
g = bproc.object.create_primitive("CUBE", scale=[BIN_HALF, BIN_HALF, WALL_T])
g.set_location([0, 0, -WALL_T]); g.enable_rigidbody(active=False, collision_shape="BOX")
bin_objs.append(g)
for x, y, span in [(BIN_HALF,0,"y"),(-BIN_HALF,0,"y"),(0,BIN_HALF,"x"),(0,-BIN_HALF,"x")]:
    sc = [WALL_T, BIN_HALF, WALL_H/2] if span == "y" else [BIN_HALF, WALL_T, WALL_H/2]
    w = bproc.object.create_primitive("CUBE", scale=sc)
    w.set_location([x, y, WALL_H/2]); w.enable_rigidbody(active=False, collision_shape="BOX")
    bin_objs.append(w)
if cc_mats:
    floor_mat = random_cc_mat()        # 바닥은 한 재질로 통일(자연스러움)
    for o in bin_objs:
        o.replace_materials(floor_mat)

# --- 부품 5~15개 랜덤 적재 + 재질 랜덤 ---
n_parts = int(rng.integers(MIN_PARTS, MAX_PARTS + 1))
for k in range(n_parts):
    cid = int(rng.integers(0, len(all_stl)))
    for obj in bproc.loader.load_obj(all_stl[cid]):
        obj.set_scale([0.001, 0.001, 0.001])
        obj.set_origin(mode="CENTER_OF_VOLUME")
        obj.set_location([float(rng.uniform(-DROP_R, DROP_R)),
                          float(rng.uniform(-DROP_R, DROP_R)),
                          float(0.06 + k * 0.025)])
        obj.set_rotation_euler(rng.uniform(0, 2*np.pi, size=3).tolist())
        obj.enable_rigidbody(active=True, collision_shape="CONVEX_HULL")
        obj.set_cp("category_id", cid + 1)
        # 재질 랜덤: 50%는 CC 텍스처, 50%는 색/거칠기/메탈릭 랜덤
        mat = bproc.material.create(f"part_{k}")
        if cc_mats and rng.random() < 0.4:
            obj.replace_materials(random_cc_mat())
        else:
            base = rng.uniform(0.05, 0.9, size=3).tolist() + [1.0]
            mat.set_principled_shader_value("Base Color", base)
            mat.set_principled_shader_value("Roughness", float(rng.uniform(0.1, 0.9)))
            mat.set_principled_shader_value("Metallic", float(rng.choice([0.0, 0.0, 1.0])))
            obj.replace_materials(mat)

bproc.object.simulate_physics_and_fix_final_poses(
    min_simulation_time=2, max_simulation_time=10, check_object_interval=1)

# --- 조명: 다광원 + 강도 랜덤 ---
n_lights = int(rng.integers(1, 4))
for _ in range(n_lights):
    light = bproc.types.Light(); light.set_type("POINT")
    light.set_location([float(rng.uniform(-0.25,0.25)), float(rng.uniform(-0.25,0.25)), float(rng.uniform(0.3,0.6))])
    light.set_energy(float(rng.uniform(30, 90)))

# --- 카메라: v1 검증된 수직 down-view 유지 ---
cam_h = float(rng.uniform(0.36, 0.44))
bproc.camera.set_resolution(512, 512)
fov = 2 * math.atan((BIN_HALF * 1.4) / cam_h)
bproc.camera.set_intrinsics_from_blender_params(lens=fov, lens_unit="FOV")
bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(
    [float(rng.uniform(-0.015,0.015)), float(rng.uniform(-0.015,0.015)), cam_h], [0, 0, 0]))

# --- 멀티모달 렌더 + 저장 ---
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"],
                                          default_values={"category_id": 0})
data = bproc.renderer.render()
bproc.writer.write_hdf5(OUT_DIR, data, append_to_existing_output=True)
import numpy as _np
seg = _np.array(data['category_id_segmaps'][0])
print(f"[V2] scene {scene_idx}: parts={n_parts} visible={len(_np.unique(seg))-1} "
      f"cc_mats={len(cc_mats)} lights={n_lights}", flush=True)
