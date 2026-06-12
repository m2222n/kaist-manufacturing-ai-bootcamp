import blenderproc as bproc
# KAIST 2주차 — 합성데이터 "1 scene" 생성 v3 (6/11 미팅 지시 반영)
#
# 6/11 KAIST 미팅 방향 지시:
#   - 부품 = "회색 단색" 고정. 알록달록하면 AI가 형태가 아니라 색으로 판별할 위험.
#     (v1은 사실상 회색이었고 그게 맞은 방향. v2의 색/CC텍스처 랜덤화는 폐기)
#   - 배경만 다양화하되, 추측성 텍스처가 아니라 "실제 가능한 두 환경"으로 한정:
#       (A) 투명 플라스틱 박스 안에 담긴 적재형   -> 4벽을 투명 재질로
#       (B) 흰색 책상 위에 흩어진 정렬형          -> 4벽 제거, 흰 바닥 평면만
#     scene마다 둘 중 하나를 랜덤 선택 (실제 빈피킹 2시나리오: 적재형/정렬형과 정합)
#
# 물리/카메라/원점/조명 다광원은 v1·v2에서 검증된 것 그대로 유지 (함정 3개 회피).
#
# 호출: blenderproc run gen_one_v3.py --blender-install-path /data/jtm/blender -- <scene_idx> <out_dir>
import numpy as np
import glob, os, sys, math

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
scene_idx = int(argv[0])
OUT_DIR   = argv[1] if len(argv) > 1 else "/data/jtm/synth_out/dataset_v3"

STL_DIR = "/home/jtm/kaist_render/stl"
# 부품 수/낙하 범위/높이는 배경(시나리오)별로 분기 — 적재형(박스) vs 정렬형(책상)
BIN_HALF, WALL_H, WALL_T = 0.15, 0.15, 0.03
DROP_R = 0.05
SEED = 1234

# 부품 회색 단색 — 모든 부품·모든 scene 동일. 무광 플라스틱 느낌.
PART_GRAY = [0.5, 0.5, 0.5, 1.0]
PART_ROUGHNESS = 0.5

bproc.init()
rng = np.random.default_rng(SEED + scene_idx)   # scene마다 다른 시드
all_stl = sorted(glob.glob(os.path.join(STL_DIR, "*.stl")))
os.makedirs(OUT_DIR, exist_ok=True)

# --- 배경 종류 선택: 0=투명 플라스틱 박스(적재형) / 1=흰색 책상(정렬형) ---
bg_kind = int(rng.integers(0, 2))

# 공통: 부품 회색 단색 재질 (한 번 만들어 모든 부품에 공유)
gray_mat = bproc.material.create("part_gray")
gray_mat.set_principled_shader_value("Base Color", PART_GRAY)
gray_mat.set_principled_shader_value("Roughness", PART_ROUGHNESS)
gray_mat.set_principled_shader_value("Metallic", 0.0)

if bg_kind == 0:
    # (A) 투명 플라스틱 박스 — 얇은 4벽 + 바닥, 부품은 박스 안에 적재(적재형)
    # 바닥: 박스 바닥(부품 안착 rigidbody)
    g = bproc.object.create_primitive("CUBE", scale=[BIN_HALF, BIN_HALF, WALL_T])
    g.set_location([0, 0, -WALL_T])
    g.enable_rigidbody(active=False, collision_shape="BOX")
    g.set_cp("category_id", 0)   # 배경(seg에서 부품만 라벨)
    # 투명 재질: 얇은 벽일수록 굴절 왜곡 적어 "유리/투명 플라스틱" 느낌이 잘 남.
    # 살짝 청록 틴트 + 낮은 roughness로 플라스틱 박스 질감.
    # 반투명 PP 박스 느낌: roughness를 약간 올려 거울 반사(은색 금속처럼 보이던 문제)를 죽이고
    # 뿌연 반투명으로. 청록 틴트 유지.
    clear_mat = bproc.material.create("clear_plastic")
    clear_mat.set_principled_shader_value("Base Color", [0.85, 0.92, 0.92, 1.0])
    clear_mat.set_principled_shader_value("Roughness", float(rng.uniform(0.15, 0.35)))
    clear_mat.set_principled_shader_value("Transmission Weight", 1.0)
    clear_mat.set_principled_shader_value("IOR", float(rng.uniform(1.40, 1.50)))
    clear_mat.set_principled_shader_value("Metallic", 0.0)
    g.replace_materials(clear_mat)
    thin_t = 0.006   # 6mm 얇은 벽 (기존 30mm는 너무 두꺼워 투명감 안 남)
    for x, y, span in [(BIN_HALF,0,"y"),(-BIN_HALF,0,"y"),(0,BIN_HALF,"x"),(0,-BIN_HALF,"x")]:
        sc = [thin_t, BIN_HALF, WALL_H/2] if span == "y" else [BIN_HALF, thin_t, WALL_H/2]
        w = bproc.object.create_primitive("CUBE", scale=sc)
        w.set_location([x, y, WALL_H/2])
        w.enable_rigidbody(active=False, collision_shape="BOX")
        w.set_cp("category_id", 0)   # 배경
        w.replace_materials(clear_mat)
    # 적재형: 부품 많이(8~15) + 좁게 떨어뜨려 박스 안에 쌓임 + z 간격 촘촘히
    min_p, max_p = 8, 15
    drop_r = DROP_R          # 좁게 모임
    z0, z_step = 0.05, 0.022 # 촘촘히 쌓임
else:
    # (B) 흰색 책상 — 벽 제거, 큰 흰 평면 위에 부품이 흩어짐(정렬형)
    # ⚠️ 바닥이 화각보다 작으면 평면 밖이 depth 무한대(배경)가 됨 -> 화각보다 충분히 크게.
    #    카메라가 보는 반경 ~= BIN_HALF*1.4 (최대 cam_h 0.44 기준), 여유 두고 BIN_HALF*5.
    desk_half = BIN_HALF * 5     # 0.75m 평면 -> 화각 전체가 책상 -> depth 항상 유효
    g = bproc.object.create_primitive("CUBE", scale=[desk_half, desk_half, WALL_T])
    g.set_location([0, 0, -WALL_T])
    g.enable_rigidbody(active=False, collision_shape="BOX")
    g.set_cp("category_id", 0)   # 배경
    white_mat = bproc.material.create("white_desk")
    v = float(rng.uniform(0.82, 0.95))   # 완전 1.0은 비현실 -> 약간의 밝기 변동
    white_mat.set_principled_shader_value("Base Color", [v, v, v, 1.0])
    white_mat.set_principled_shader_value("Roughness", float(rng.uniform(0.4, 0.8)))
    white_mat.set_principled_shader_value("Metallic", 0.0)
    g.replace_materials(white_mat)
    # 정렬형: 부품 4~8개 + 적당히 흩뿌림(너무 넓으면 화각 밖으로 빠져 휑함) + 낮은 높이에서
    # 거의 같은 z로 떨어뜨려 탑처럼 쌓이지 않고 책상 위에 드문드문 흩어지게
    min_p, max_p = 4, 8
    drop_r = DROP_R * 1.2    # 적당히 흩뿌림 (0.06m, 화각 안 유지 + 겹침↓). 0.075는 일부 화각밖 빠져 visible↓
    z0, z_step = 0.03, 0.005 # 낮고 거의 평평하게

# --- 부품 랜덤 적재 (회색 단색) — 부품 수/배치는 배경(시나리오)별로 위에서 결정 ---
n_parts = int(rng.integers(min_p, max_p + 1))
for k in range(n_parts):
    cid = int(rng.integers(0, len(all_stl)))
    for obj in bproc.loader.load_obj(all_stl[cid]):
        obj.set_scale([0.001, 0.001, 0.001])
        obj.set_origin(mode="CENTER_OF_VOLUME")   # STL 원점 offset 제거 (안하면 허공 무한낙하)
        obj.set_location([float(rng.uniform(-drop_r, drop_r)),
                          float(rng.uniform(-drop_r, drop_r)),
                          float(z0 + k * z_step)])
        obj.set_rotation_euler(rng.uniform(0, 2*np.pi, size=3).tolist())
        obj.enable_rigidbody(active=True, collision_shape="CONVEX_HULL")
        obj.set_cp("category_id", cid + 1)
        obj.replace_materials(gray_mat)            # 회색 단색 고정

bproc.object.simulate_physics_and_fix_final_poses(
    min_simulation_time=2, max_simulation_time=10, check_object_interval=1)

# --- 조명: 다광원 + 강도 랜덤 (6/11 "조명 중요") ---
# 흰 책상은 과노출 방지로 강도 약간 보수적
n_lights = int(rng.integers(1, 4))
e_lo, e_hi = (30, 90) if bg_kind == 0 else (25, 70)
for _ in range(n_lights):
    light = bproc.types.Light(); light.set_type("POINT")
    light.set_location([float(rng.uniform(-0.25,0.25)), float(rng.uniform(-0.25,0.25)), float(rng.uniform(0.3,0.6))])
    light.set_energy(float(rng.uniform(e_lo, e_hi)))

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
bg_name = "clear_box" if bg_kind == 0 else "white_desk"
print(f"[V3] scene {scene_idx}: bg={bg_name} parts={n_parts} "
      f"visible={len(_np.unique(seg))-1} lights={n_lights}", flush=True)
