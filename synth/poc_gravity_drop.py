import blenderproc as bproc
# KAIST 2주차 PoC — 디지털 트윈 합성데이터 (BlenderProc)
# 우리 STL 부품을 빈(bin)에 중력 낙하 적재 -> RGB + depth + instance segmentation 출력.
#
# KAIST 6/9 피드백 직결:
#   - 합성데이터(디지털 트윈/도메인 랜덤화)로 학습셋 생성
#   - Blaze=depth / ACE2=seg 모달리티 분담 -> depth + seg 둘 다 렌더
#   - 적재형 빈피킹(occlusion) 대비 = 중력 낙하 적재
#
# 실행:
#   /data/jtm/blenderproc_venv/bin/blenderproc run \
#       /home/jtm/kaist_project/synth/poc_gravity_drop.py \
#       --blender-install-path /data/jtm/blender
import numpy as np
import glob
import os
import math

STL_DIR = "/home/jtm/kaist_render/stl"
OUT_DIR = "/data/jtm/synth_out/poc_drop"   # /data/jtm/ 내 개인 폴더 (루트 디스크 보호)
NUM_PARTS = 8                              # 빈 크기 대비 적정 (튕김 방지)
BIN_HALF = 0.15                            # 빈 내부 반폭 (전체 0.30m x 0.30m)
WALL_H = 0.15                              # 빈 벽 높이
WALL_T = 0.01                              # 벽 두께 (CUBE = 관통 방지, PLANE은 빠른 물체 통과)
DROP_R = 0.05                              # 흩뿌림 반경
SEED = 42                                  # 재현성 (Math.random 회피, 고정 시드)

bproc.init()
rng = np.random.default_rng(SEED)

# --- 1. 빈(bin) = 두꺼운 바닥 + 4벽 (CUBE, 단위 m. PLANE은 빠른 물체 관통하므로 박스) ---
# CUBE primitive 기본 크기 2m(±1) -> scale=절반치수
ground = bproc.object.create_primitive("CUBE", scale=[BIN_HALF, BIN_HALF, WALL_T])
ground.set_name("bin_floor")
ground.set_location([0, 0, -WALL_T])  # 윗면이 z=0
ground.enable_rigidbody(active=False, collision_shape="BOX")

# 4벽: 바닥 가장자리에 세운 두꺼운 박스
walls = []
wall_defs = [( BIN_HALF, 0, "y"), (-BIN_HALF, 0, "y"),
             (0,  BIN_HALF, "x"), (0, -BIN_HALF, "x")]
for i, (x, y, span) in enumerate(wall_defs):
    if span == "y":   # x벽: y방향으로 길게
        sc = [WALL_T, BIN_HALF, WALL_H/2]
    else:             # y벽: x방향으로 길게
        sc = [BIN_HALF, WALL_T, WALL_H/2]
    w = bproc.object.create_primitive("CUBE", scale=sc)
    w.set_location([x, y, WALL_H/2])
    w.enable_rigidbody(active=False, collision_shape="BOX")
    walls.append(w)

# --- 2. 우리 STL 부품 로드 + 중력 활성화 (빈 가운데 위 공중에서 떨어뜨림) ---
# NUM_PARTS개를 STL 28개에서 순환 샘플 (occlusion 적재 위해 같은 부품 반복 허용)
all_stl = sorted(glob.glob(os.path.join(STL_DIR, "*.stl")))
parts = []
for idx in range(NUM_PARTS):
    stl = all_stl[idx % len(all_stl)]
    cat_id = (idx % len(all_stl)) + 1
    objs = bproc.loader.load_obj(stl)
    for obj in objs:
        obj.set_scale([0.001, 0.001, 0.001])  # mm -> m
        # 빈 중앙 위쪽 흩뿌림 + 높이 시차(0.04m) -> 순차 낙하해 빈 안에 겹쳐 적재
        obj.set_location([float(rng.uniform(-DROP_R, DROP_R)),
                          float(rng.uniform(-DROP_R, DROP_R)),
                          float(0.08 + idx * 0.04)])
        obj.set_rotation_euler(rng.uniform(0, 2*np.pi, size=3).tolist())
        # 오목 부품 정확 충돌 = convex decomposition (V-HACD 내장)
        obj.enable_rigidbody(active=True, collision_shape="CONVEX_HULL")
        obj.set_cp("category_id", cat_id)      # instance seg 라벨
        parts.append(obj)

# --- 3. 물리 시뮬 (중력 낙하 → 안정될 때까지). 부품 많아 시간 넉넉히 ---
bproc.object.simulate_physics_and_fix_final_poses(
    min_simulation_time=3, max_simulation_time=12, check_object_interval=1
)

# --- 4. 조명 + 카메라 (빈 위에서 내려다봄 = eye-in-hand 근사) ---
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([0.15, -0.15, 0.4])
light.set_energy(60)

# 빈(0.24m)을 0.4m 높이에서 수직으로 내려다봄. FOV를 빈+여유 덮도록 명시
CAM_H = 0.4
bproc.camera.set_resolution(512, 512)
# 수평 시야 = 2*atan((BIN_HALF*1.4)/CAM_H) -> 빈 가장자리 + 40% 여유
fov = 2 * math.atan((BIN_HALF * 1.4) / CAM_H)
bproc.camera.set_intrinsics_from_blender_params(lens=fov, lens_unit="FOV")
cam_pose = bproc.math.build_transformation_mat(
    [0, 0, CAM_H], [0, 0, 0]   # 빈 바로 위에서 수직 down-view (광축 -Z)
)
bproc.camera.add_camera_pose(cam_pose)

# --- 5. 멀티모달 렌더: RGB + depth + instance seg ---
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"],
                                          default_values={"category_id": 0})  # bin/벽 = 배경 0
data = bproc.renderer.render()

# --- 6. 저장 (hdf5: colors/depth/category_id_segmaps) ---
os.makedirs(OUT_DIR, exist_ok=True)
bproc.writer.write_hdf5(OUT_DIR, data)
print(f"[PoC] done. parts={len(parts)} -> {OUT_DIR}")
