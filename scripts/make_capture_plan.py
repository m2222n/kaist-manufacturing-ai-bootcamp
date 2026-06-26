#!/usr/bin/env python3
"""
실증 본촬영 100장 배치표 생성기 (least-used-first).

목적:
  - 27종 부품을 100장에 **최대한 균등**하게 분배하는 촬영 배치표를 만든다.
  - 장당 부품 수(K)는 시험촬영으로 확정한 값을 인자로 넣는다.
  - "지금까지 가장 적게 쓰인 종부터 채우기"(least-used-first)로 종당 등장 편차를 ±1로 최소화.

K값 결정 (시험촬영 후):
  - 시험촬영(8/10/12개)에서 valid% 와 가림 정도를 보고 K 범위를 정한다.
  - 단일 K로 고정하기보다 합성 분포(visible 8~14, 평균 8.12)와 맞춰 범위를 권장.
  - 기본값 = 드문 8 / 중간 10 / 빽빽 12 의 3구간 (합성과 정합 + occlusion 다양성).

실행 예:
  # 기본(8/10/12 혼합, 100장)
  python make_capture_plan.py

  # 단일 K=10으로 100장
  python make_capture_plan.py --k 10

  # 직접 구간 지정 (개수:장수 쌍)
  python make_capture_plan.py --mix 8:35 10:35 12:30

출력:
  - capture_plan.csv  (장번호, 밀도, 담을 부품 label 목록)
  - 종당 등장 횟수 요약 (균등성 확인)
"""
from __future__ import annotations
import argparse, csv, os

# label 1~27 (label_map_27parts.md 와 동일 순서. 번호만 쓰므로 이름은 참고용)
PART_NAMES = {
    1: "01_sol_block_a", 2: "02_sol_block_b", 3: "03_sol_block_front",
    4: "06_sol_block_back", 5: "07_guide_paper_l", 6: "08_r_guide_a",
    7: "09_guide_paper_r", 8: "11_sw_block", 9: "13_variant",
    10: "13_x2_bcf8ccb4", 11: "14_13", 12: "15_roller_bracket",
    13: "16_cam_f_bracket", 14: "17_mks_holder", 15: "18_button_function_niro",
    16: "bracket_case", 17: "bracket_sen_1", 18: "bracket_sensor1",
    19: "bracket_sensor2", 20: "brkt_switch", 21: "guide_paper_roll_cover_left",
    22: "guide_paper_roll_cover_right", 23: "main_body", 24: "plate_e",
    25: "r_guide_a_l", 26: "r_guide_a_r", 27: "top_inner_sheet",
}
N_PARTS = 27


def build_densities(args) -> list[int]:
    """장별 부품 수(K) 리스트 100개를 만든다."""
    if args.k:
        return [args.k] * args.scenes
    if args.mix:
        densities = []
        for token in args.mix:
            k, cnt = token.split(":")
            densities += [int(k)] * int(cnt)
        if len(densities) != args.scenes:
            raise SystemExit(f"--mix 합({len(densities)})이 --scenes({args.scenes})와 다릅니다.")
        return densities
    # 기본 = 8:35 / 10:35 / 12:30 = 100장
    return [8] * 35 + [10] * 35 + [12] * 30


def assign(densities: list[int]) -> list[list[int]]:
    """least-used-first: 매 장마다 지금까지 가장 적게 쓰인 label부터 K개 채운다.
    한 장 안에서 같은 종 중복은 허용(부품 실물이 여러 개일 수 있고, 골고루가 목적)."""
    used = {p: 0 for p in range(1, N_PARTS + 1)}
    scenes = []
    for k in densities:
        # 사용횟수 오름차순, 동률이면 label 번호 순
        order = sorted(range(1, N_PARTS + 1), key=lambda p: (used[p], p))
        picked = order[:k] if k <= N_PARTS else order + order[: k - N_PARTS]
        for p in picked:
            used[p] += 1
        scenes.append(sorted(picked))
    return scenes, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=100, help="총 장 수 (기본 100)")
    ap.add_argument("--k", type=int, default=None, help="단일 장당 부품 수로 고정")
    ap.add_argument("--mix", nargs="+", default=None,
                    help="구간 지정 'K:장수' (예: 8:35 10:35 12:30)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "capture_plan.csv"))
    args = ap.parse_args()

    densities = build_densities(args)
    if len(densities) != args.scenes:
        # --k 인 경우 길이 보정 불필요(이미 scenes 길이), 그 외 위에서 체크됨
        pass
    scenes, used = assign(densities)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "density_K", "labels"])
        for i, (k, labs) in enumerate(zip(densities, scenes), 1):
            w.writerow([i, k, " ".join(map(str, labs))])

    total_slots = sum(densities)
    counts = sorted(used.values())
    print(f"=== 배치표 생성 완료 → {out} ===")
    print(f"총 {len(scenes)}장 / 총 슬롯 {total_slots} / 종당 평균 {total_slots/N_PARTS:.1f}회")
    print(f"종당 등장: 최소 {counts[0]}회 ~ 최대 {counts[-1]}회 (편차 {counts[-1]-counts[0]})")
    print("\n[밀도 구성]")
    from collections import Counter
    for k, c in sorted(Counter(densities).items()):
        print(f"  {k}개 적재: {c}장")
    print("\n[종별 등장 횟수]")
    for p in range(1, N_PARTS + 1):
        print(f"  label {p:2d} ({PART_NAMES[p]:28s}): {used[p]}회")
    print("\n👉 촬영 시: 각 scene 행의 label 부품을 빈/트레이에 깔고 1장 촬영.")
    print("   파일명 권장: scene001_K8.npy 처럼 scene 번호로 (label은 이 CSV가 정답).")


if __name__ == "__main__":
    main()
