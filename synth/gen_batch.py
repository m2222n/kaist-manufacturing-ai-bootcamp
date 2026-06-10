import blenderproc as bproc
# KAIST 2주차 — 합성데이터 대량 생성 배치 (BlenderProc, 6000 CPU 밤샘)
# 적재형 빈피킹: 우리 STL 28부품을 빈에 중력 낙하 적재 -> RGB+depth+seg+pose 자동 라벨.
#
# 랜덤화 방침 (6/9 조교/교수 피드백 절충):
#   - 물리(부품 조합·수·자세·위치) + 조명(위치·세기) + 카메라(높이·기울기) = 넣음 (도메인 다양성 ROI 높음)
#   - 배경/질감 실사 합성 = 보류 (조교 "조명·질감 완벽 안 해도 됨" + 비용 큼 -> 3주차 실사 확보 후)
#   - 빈 안 단색 유지 = seg/depth 학습 신호 명확
#
# 실행 (백그라운드 밤샘):
#   nohup /data/jtm/blenderproc_venv/bin/blenderproc run \
#       /home/jtm/kaist_project/synth/gen_batch.py \
#       --blender-install-path /data/jtm/blender > /data/jtm/synth_out/batch.log 2>&1 &
import numpy as np
import glob
import os
import math

STL_DIR   = "/home/jtm/kaist_render/stl"
OUT_DIR   = "/data/jtm/synth_out/dataset_v1"   # /data/jtm 개인 폴더
NUM_SCENES = 2000                              # 목표 장수 (CPU ~10s/장 -> ~5.5h)
MIN_PARTS, MAX_PARTS = 5, 15                   # 장면당 부품 수 범위 (occlusion 다양성)
BIN_HALF, WALL_H, WALL_T = 0.15, 0.15, 0.01    # 빈 0.30m + 두꺼운 CUBE 벽 (관통 방지)
DROP_R = 0.05
SEED = 1234

bproc.init()
all_stl = sorted(glob.glob(os.path.join(STL_DIR, "*.stl")))
os.makedirs(OUT_DIR, exist_ok=True)

# 부품 STL은 한 번만 로드해두고 장면마다 재배치 (로드 비용 절약)
# -> BlenderProc는 장면 리셋이 까다로워, 안전하게 장면마다 로드/삭제 방식 사용
def build_bin():
    objs = []
    g = bproc.object.create_primitive("CUBE", scale=[BIN_HALF, BIN_HALF, WALL_T])
    g.set_location([0, 0, -WALL_T]); g.enable_rigidbody(active=False, collision_shape="BOX")
    objs.append(g)
    for x, y, span in [(BIN_HALF,0,"y"),(-BIN_HALF,0,"y"),(0,BIN_HALF,"x"),(0,-BIN_HALF,"x")]:
        sc = [WALL_T, BIN_HALF, WALL_H/2] if span == "y" else [BIN_HALF, WALL_T, WALL_H/2]
        w = bproc.object.create_primitive("CUBE", scale=sc)
        w.set_location([x, y, WALL_H/2]); w.enable_rigidbody(active=False, collision_shape="BOX")
        objs.append(w)
    return objs

# 렌더 출력 활성화는 루프 밖에서 1회만 (루프 안에서 두 번 호출 시 에러)
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"],
                                          default_values={"category_id": 0})

# 이미 생성된 hdf5 개수 -> 이어찍기(resume) 지원
existing = len(glob.glob(os.path.join(OUT_DIR, "*.hdf5")))
print(f"[BATCH] resume from {existing}, target {NUM_SCENES}", flush=True)

for scene_i in range(existing, NUM_SCENES):
    rng = np.random.default_rng(SEED + scene_i)   # 장면마다 다른 시드 (Math.random 회피)
    bproc.utility.reset_keyframes()
    # 이전 장면 부품/빈 제거
    bproc.object.delete_multiple(bproc.object.get_all_mesh_objects())

    # --- 빈 ---
    build_bin()

    # --- 부품: 5~15개 랜덤, 28종에서 랜덤 선택 ---
    n_parts = int(rng.integers(MIN_PARTS, MAX_PARTS + 1))
    parts = []
    for k in range(n_parts):
        cat_id = int(rng.integers(0, len(all_stl)))
        objs = bproc.loader.load_obj(all_stl[cat_id])
        for obj in objs:
            obj.set_scale([0.001, 0.001, 0.001])
            # ★ STL 원점이 형상 밖에 있는 부품 다수 -> 원점을 부피중심으로 재설정해야
            #   set_location이 실제 메시를 빈 안에 놓음 (안 하면 허공서 무한낙하 z=-510)
            obj.set_origin(mode="CENTER_OF_VOLUME")
            obj.set_location([float(rng.uniform(-DROP_R, DROP_R)),
                              float(rng.uniform(-DROP_R, DROP_R)),
                              float(0.06 + k * 0.025)])  # 시차 낙하 -> 적재 (낮게=관통 방지)
            obj.set_rotation_euler(rng.uniform(0, 2*np.pi, size=3).tolist())
            obj.enable_rigidbody(active=True, collision_shape="CONVEX_HULL")
            obj.set_cp("category_id", cat_id + 1)        # 0=배경, 1~28=부품
            parts.append(obj)

    # --- 중력 시뮬 ---
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=2, max_simulation_time=10, check_object_interval=1)

    # --- 조명 랜덤화 (위치·세기) ---
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([float(rng.uniform(-0.2, 0.2)),
                        float(rng.uniform(-0.2, 0.2)),
                        float(rng.uniform(0.3, 0.6))])
    light.set_energy(float(rng.uniform(40, 100)))

    # --- 카메라 랜덤화 (높이 + 약간 기울기) = eye-in-hand 변동 근사 ---
    cam_h = float(rng.uniform(0.35, 0.45))
    bproc.camera.set_resolution(512, 512)
    fov = 2 * math.atan((BIN_HALF * 1.4) / cam_h)
    bproc.camera.set_intrinsics_from_blender_params(lens=fov, lens_unit="FOV")
    # 수직 down-view([0,0,0] 회전)가 검증된 기준. tilt를 직접 주면 빈을 벗어나 빈 장면이 됨.
    # -> 위치(xy)만 약간 흔들고 회전은 아주 작게(±3도)만. 카메라가 빈 중심을 계속 보도록 유지
    rz = float(rng.uniform(0, 2*np.pi))       # 광축 중심 회전(roll)은 항상 안전 -> 시점 다양화
    small_tilt = rng.uniform(-0.05, 0.05, size=2)  # ~±3도만 (빈 안 유지)
    cam_pose = bproc.math.build_transformation_mat(
        [float(rng.uniform(-0.015, 0.015)), float(rng.uniform(-0.015, 0.015)), cam_h],
        [float(small_tilt[0]), float(small_tilt[1]), rz])
    bproc.camera.add_camera_pose(cam_pose)

    # --- 멀티모달 렌더 (출력 활성화는 루프 밖에서 이미 1회 호출됨) ---
    data = bproc.renderer.render()
    bproc.writer.write_hdf5(OUT_DIR, data, append_to_existing_output=True)

    if (scene_i + 1) % 10 == 0 or scene_i == existing:
        print(f"[BATCH] {scene_i + 1}/{NUM_SCENES} scenes done (parts={n_parts})", flush=True)

print(f"[BATCH] ALL DONE: {NUM_SCENES} scenes -> {OUT_DIR}", flush=True)
