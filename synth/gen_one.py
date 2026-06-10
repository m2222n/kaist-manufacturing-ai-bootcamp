import blenderproc as bproc
# KAIST 2주차 — 합성데이터 "1 scene" 생성 (외부 bash 루프가 N번 호출)
# 루프-in-스크립트 방식(gen_batch.py)이 delete_multiple/reset_keyframes 상태누수로
# 2번째 장면부터 빈 장면이 되는 문제 -> scene마다 깨끗한 프로세스로 분리하는 게 확실.
#
# 호출: blenderproc run gen_one.py --blender-install-path /data/jtm/blender -- <scene_idx> <out_dir>
import numpy as np
import glob, os, sys, math

# "--" 뒤 인자: scene_idx, out_dir
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
scene_idx = int(argv[0])
OUT_DIR   = argv[1] if len(argv) > 1 else "/data/jtm/synth_out/dataset_v1"

STL_DIR = "/home/jtm/kaist_render/stl"
MIN_PARTS, MAX_PARTS = 5, 15
BIN_HALF, WALL_H, WALL_T = 0.15, 0.15, 0.03
DROP_R = 0.05
SEED = 1234

bproc.init()
rng = np.random.default_rng(SEED + scene_idx)   # scene마다 다른 시드
all_stl = sorted(glob.glob(os.path.join(STL_DIR, "*.stl")))
os.makedirs(OUT_DIR, exist_ok=True)

# --- 빈(두꺼운 CUBE 바닥+4벽) ---
g = bproc.object.create_primitive("CUBE", scale=[BIN_HALF, BIN_HALF, WALL_T])
g.set_location([0, 0, -WALL_T]); g.enable_rigidbody(active=False, collision_shape="BOX")
for x, y, span in [(BIN_HALF,0,"y"),(-BIN_HALF,0,"y"),(0,BIN_HALF,"x"),(0,-BIN_HALF,"x")]:
    sc = [WALL_T, BIN_HALF, WALL_H/2] if span == "y" else [BIN_HALF, WALL_T, WALL_H/2]
    w = bproc.object.create_primitive("CUBE", scale=sc)
    w.set_location([x, y, WALL_H/2]); w.enable_rigidbody(active=False, collision_shape="BOX")

# --- 부품 5~15개 랜덤 적재 ---
n_parts = int(rng.integers(MIN_PARTS, MAX_PARTS + 1))
for k in range(n_parts):
    cid = int(rng.integers(0, len(all_stl)))
    for obj in bproc.loader.load_obj(all_stl[cid]):
        obj.set_scale([0.001, 0.001, 0.001])
        obj.set_origin(mode="CENTER_OF_VOLUME")   # STL 원점 offset 제거 (안하면 허공 무한낙하)
        obj.set_location([float(rng.uniform(-DROP_R, DROP_R)),
                          float(rng.uniform(-DROP_R, DROP_R)),
                          float(0.06 + k * 0.025)])
        obj.set_rotation_euler(rng.uniform(0, 2*np.pi, size=3).tolist())
        obj.enable_rigidbody(active=True, collision_shape="CONVEX_HULL")
        obj.set_cp("category_id", cid + 1)

bproc.object.simulate_physics_and_fix_final_poses(
    min_simulation_time=2, max_simulation_time=10, check_object_interval=1)

# --- 조명 랜덤화 ---
light = bproc.types.Light(); light.set_type("POINT")
light.set_location([float(rng.uniform(-0.2,0.2)), float(rng.uniform(-0.2,0.2)), float(rng.uniform(0.3,0.6))])
light.set_energy(float(rng.uniform(40, 100)))

# --- 카메라: 수직 down-view 기준 + xy 위치/높이만 약간 랜덤 (회전은 검증된 [0,0,0]) ---
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
print(f"[ONE] scene {scene_idx}: parts={n_parts} visible={len(_np.unique(seg))-1}", flush=True)
