#!/usr/bin/env python3
"""
Blaze-112 실증 본촬영 100장 — 맥북에서 실행. 라이브 보면서 s로 모으기.

설계 의도 (6/26):
  - 부품 100번 교체는 비현실적 → 부품을 여러 종 깔아두고(적재형) **위치·자세·뭉침만
    바꿔가며 연사**로 모은다. 구성은 중간중간 몇 번 바꿔 27종이 골고루 들어가게.
  - 저장은 **raw(848×480 uint16 mm)** = 원본 보존. 합성 포맷(512/m/배경 NaN) 변환은
    조교 컨펌 후 별도 스크립트로(원본 raw가 있으면 재촬영 없이 변환만).
  - PART 지표(부품 거리/픽셀/채움 + OK) 화면 표시 = 매 장 품질 즉시 확인.

키:
  s     = 현재 프레임 저장 (shot_NNN.npy + .png), 번호 자동 증가
  스페이스 = 구성 메모 토글(파일명 접두 그룹 바꿈, 예: g1→g2) — 부품 구성 바꿀 때
  u     = 직전 저장 취소(번호 되돌림, 파일 삭제)
  q/ESC = 종료

실행:
  cd ~/Desktop && sudo python blaze_capture_100.py --ip <BLAZE_IP> --scale 2
  (이어찍기: 폴더에 있는 마지막 번호 다음부터 자동으로 이어감)
"""
from __future__ import annotations
import argparse, os, sys, time, glob, re
import numpy as np

try:
    from pypylon import pylon
    import cv2
except Exception as e:
    sys.exit("pypylon/opencv import 실패: %s\n맥북 binpick venv에서 실행하세요." % e)


def _try(label, fn):
    try:
        fn(); print(f"  ✓ {label}"); return True
    except Exception as e:
        print(f"  · {label} 건너뜀 ({type(e).__name__})"); return False


def open_blaze(ip: str, short_range=True):
    tl = pylon.TlFactory.GetInstance(); tl.CreateTl("BaslerGigE")
    di = pylon.CDeviceInfo(); di.SetIpAddress(ip); di.SetDeviceClass("BaslerGigE")
    cam = pylon.InstantCamera(tl.CreateDevice(di)); cam.Open()
    try:
        cam.ComponentSelector.SetValue("Intensity"); cam.ComponentEnable.SetValue(False)
    except Exception as e:
        print("Intensity off 경고:", e)
    try:
        cam.ComponentSelector.SetValue("Range"); cam.ComponentEnable.SetValue(True)
        try: cam.PixelFormat.SetValue("Coord3D_C16")
        except Exception: cam.PixelFormat.SetValue("Mono16")
    except Exception as e:
        print("Range 설정 경고:", e)
    print("연결:", cam.GetDeviceInfo().GetModelName(), cam.GetDeviceInfo().GetIpAddress())
    print("[depth 품질 파라미터] (없는 노드는 자동 건너뜀)")
    if short_range:
        _try("OperatingMode=ShortRange", lambda: cam.OperatingMode.SetValue("ShortRange"))
    for node in ("SpatialFilter", "Scan3dSpatialFilter"):
        if _try(f"{node}=On", lambda n=node: getattr(cam, n).SetValue(True)): break
    for node in ("TemporalFilter", "Scan3dTemporalFilter"):
        if _try(f"{node}=On", lambda n=node: getattr(cam, n).SetValue(True)): break
    for node in ("OutlierRemoval", "Scan3dOutlierRemoval", "FlyingPixelsRemoval"):
        if _try(f"{node}=On", lambda n=node: getattr(cam, n).SetValue(True)): break
    return cam


def part_stats(depth, band_mm=200):
    valid = depth[depth > 0]
    if valid.size < 50:
        return 0, 0, 0.0, 0, 0
    lo, hi = np.percentile(valid, [2, 98])
    bins = np.arange(int(lo), int(hi) + 20, 20)
    if bins.size < 2:
        peak = int(np.median(valid))
    else:
        h, edges = np.histogram(valid, bins=bins)
        peak = int((edges[h.argmax()] + edges[h.argmax() + 1]) / 2)
    near = max(0, peak - band_mm // 2); far = peak + band_mm // 2
    part = (depth >= near) & (depth <= far)
    part_px = int(part.sum())
    if part_px == 0:
        return 0, 0, 0.0, near, far
    part_med = int(np.median(depth[part]))
    ys, xs = np.where(part)
    bbox = max((ys.max()-ys.min()+1) * (xs.max()-xs.min()+1), 1)
    return part_med, part_px, 100.0*part_px/bbox, near, far


def colorize(depth):
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    norm = np.clip((depth.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    vis = (255 * (1 - norm)).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    color[depth == 0] = (0, 0, 0)
    return color


def next_index(out: str) -> int:
    """폴더의 마지막 shot 번호 다음(이어찍기)."""
    nums = [int(m.group(1)) for f in glob.glob(os.path.join(out, "shot_*.npy"))
            if (m := re.search(r"shot_(\d+)", os.path.basename(f)))]
    return (max(nums) + 1) if nums else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=os.environ.get("BASLER_BLAZE_IP", "<BLAZE_IP>"))
    ap.add_argument("--out", default=os.path.expanduser("~/Desktop/blaze_capture100"))
    ap.add_argument("--target", type=int, default=100, help="목표 장수")
    ap.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cam = open_blaze(args.ip)
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    idx = next_index(args.out); group = 1
    last_saved = None
    print(f"본촬영 시작 — 저장 {args.out} / {idx}번부터 / 목표 {args.target}장")
    print("[s]저장 [스페이스]구성그룹변경 [u]직전취소 [q]종료")
    win = "Blaze capture100"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(848*args.scale), int(480*args.scale))
    last = time.time(); fps = 0.0
    try:
        while True:
            res = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
            if not res.GrabSucceeded():
                res.Release(); continue
            depth = res.Array.copy(); res.Release()
            now = time.time(); fps = 0.9*fps + 0.1*(1.0/max(now-last, 1e-3)); last = now
            pct = 100*(depth > 0).sum()/depth.size
            p_med, p_px, fill, near, far = part_stats(depth)
            ok = "OK" if (pct < 35 and p_px > 10000 and fill >= 35) else ".."
            vis = colorize(depth)
            if args.scale != 1.0:
                vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale,
                                 interpolation=cv2.INTER_NEAREST)
            saved = idx - 1
            cv2.putText(vis, f"saved={saved}/{args.target}  next=shot_{idx:03d}  group=g{group}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(vis, f"PART med={p_med}mm px={p_px//1000}k fill={fill:.0f}% [{ok}]",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0) if ok=="OK" else (0,200,255), 2)
            cv2.putText(vis, f"(all_valid={pct:.0f}% fps={fps:.1f})  [s]save [space]group [u]undo [q]quit",
                        (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.imshow(win, vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('s'):
                base = os.path.join(args.out, f"shot_{idx:03d}_g{group}")
                np.save(base + ".npy", depth)           # raw uint16 mm (원본)
                cv2.imwrite(base + ".png", vis)         # 미리보기
                print(f"[{idx}/{args.target}] saved {os.path.basename(base)}  "
                      f"med={p_med}mm px={p_px} fill={fill:.0f}% [{ok}]")
                last_saved = base; idx += 1
                if idx - 1 >= args.target:
                    print(f"\n🎉 목표 {args.target}장 달성! (계속 찍어도 됨)")
            elif k == ord(' '):
                group += 1
                print(f"== 구성 그룹 변경 → g{group} (부품 구성 바꾼 뒤 이어 촬영) ==")
            elif k == ord('u') and last_saved:
                for ext in (".npy", ".png"):
                    p = last_saved + ext
                    if os.path.exists(p): os.remove(p)
                idx -= 1; print(f"↩ 직전 취소: {os.path.basename(last_saved)} 삭제 → next=shot_{idx:03d}")
                last_saved = None
    finally:
        cam.StopGrabbing(); cam.Close(); cv2.destroyAllWindows()
        print(f"\n종료. 총 {idx-1}장 저장 →", args.out)


if __name__ == "__main__":
    main()
