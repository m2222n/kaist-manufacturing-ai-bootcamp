#!/usr/bin/env python3
"""
Blaze-112 라이브 뷰어 — 맥북에서 실행. 실시간 depth 화면 + 키로 캡처.

GigE TL + IP 직접 (6/23 검증) + Range만(Intensity off).
화면: depth를 컬러맵으로 표시(가까울수록 따뜻한색), 좌상단에 거리/유효픽셀 표시.
키:
  s  = 현재 프레임 저장 (~/Desktop/blaze_step0/live_HHMMSS.npy + .png)
  q 또는 ESC = 종료

실행:
  sudo .venv/binpick/bin/python ~/Desktop/blaze_live_view.py --ip <BLAZE_IP>
  sudo .venv/binpick/bin/python ~/Desktop/blaze_live_view.py --ip <BLAZE_IP> --scale 2

화질 개선(6/26, 안전판):
  - 표시 로직은 이전(잘 되던) 그대로 = valid 픽셀 전부 표시(작업영역 게이팅 없음).
  - --scale: 창만 확대(데이터 무관, 보기용).
  - 카메라 depth 파라미터(노출/필터)는 노드가 있으면 적용, 없으면 조용히 건너뜀
    → 표시를 건드리지 않으므로 화면이 비는 일 없음.
"""
from __future__ import annotations
import argparse, os, sys, time
from datetime import datetime
import numpy as np

try:
    from pypylon import pylon
    import cv2
except Exception as e:
    sys.exit("pypylon/opencv import 실패: %s\n맥북 binpick venv에서 실행하세요." % e)


def _try(label, fn):
    """GenICam 노드는 모델·펌웨어별로 없을 수 있음 → 있으면 적용, 없으면 조용히 패스."""
    try:
        fn(); print(f"  ✓ {label}"); return True
    except Exception as e:
        print(f"  · {label} 건너뜀 ({type(e).__name__})"); return False


def tune_quality(cam, exposure_us, short_range):
    """depth 데이터 품질 향상 — 있는 노드만. 표시는 절대 안 건드림(화면 비는 일 없음)."""
    print("[depth 품질 파라미터] (없는 노드는 자동 건너뜀)")
    if short_range:
        _try("OperatingMode=ShortRange", lambda: cam.OperatingMode.SetValue("ShortRange"))
    if exposure_us is not None:
        _try("ExposureAuto=Off", lambda: cam.ExposureAuto.SetValue("Off"))
        _try(f"ExposureTime={exposure_us}us", lambda: cam.ExposureTime.SetValue(float(exposure_us)))
    for node in ("SpatialFilter", "Scan3dSpatialFilter"):
        if _try(f"{node}=On", lambda n=node: getattr(cam, n).SetValue(True)): break
    for node in ("TemporalFilter", "Scan3dTemporalFilter"):
        if _try(f"{node}=On", lambda n=node: getattr(cam, n).SetValue(True)): break
    for node in ("OutlierRemoval", "Scan3dOutlierRemoval", "FlyingPixelsRemoval"):
        if _try(f"{node}=On", lambda n=node: getattr(cam, n).SetValue(True)): break


def open_blaze(ip: str, exposure_us=None, short_range=False):
    tl = pylon.TlFactory.GetInstance()
    tl.CreateTl("BaslerGigE")
    di = pylon.CDeviceInfo(); di.SetIpAddress(ip); di.SetDeviceClass("BaslerGigE")
    cam = pylon.InstantCamera(tl.CreateDevice(di))
    cam.Open()
    # Intensity off, Range만
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
    tune_quality(cam, exposure_us, short_range)
    return cam


def colorize(depth: np.ndarray) -> np.ndarray:
    """depth(uint16 mm) → 컬러맵 BGR. 가까울수록 밝고 따뜻하게, 배경(0)=검정. (이전 로직 유지)"""
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    norm = np.clip((depth.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    vis = (255 * (1 - norm)).astype(np.uint8)        # 가까울수록 밝게
    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)  # JET→TURBO (단계 구분 약간 우위)
    color[depth == 0] = (0, 0, 0)                    # 무효 = 검정
    return color


def part_stats(depth: np.ndarray, band_mm: int = 200):
    """부품 중심 지표 — 화면은 안 건드리고 '부품이 잘 잡혔나'만 숫자로.

    가까이(30~40cm) 찍으면 전체 valid%는 낮게 나오지만(배경 NaN=정상, 합성과 일치),
    정작 봐야 할 건 '부품 픽셀이 두껍게·또렷하게 잡혔나'다. 그래서:
      - 부품 거리대 = valid depth의 하위(가까운) 클러스터. 가장 가까운 픽셀(p2)부터
        band_mm(기본 12cm) 안쪽을 부품으로 본다(박스 바닥보다 부품이 카메라에 가깝다는 가정).
      - part_med = 부품 median 거리(mm) — 0.3~0.4m 들어오는지 확인용
      - part_px  = 부품 픽셀 수(천 단위) — 화각을 충분히 채우는지
      - fill     = 부품 영역 내부가 안 비고 채워진 비율(%) — 구멍·flying pixel 적을수록 ↑
    반환: (part_med_mm, part_px, fill_pct, near_mm, far_mm)
    """
    valid = depth[depth > 0]
    if valid.size < 50:
        return 0, 0, 0.0, 0, 0
    near = int(np.percentile(valid, 2))              # 가장 가까운 표면(노이즈 컷)
    far = near + band_mm                             # 부품 거리대 상한
    part = (depth >= near) & (depth <= far)
    part_px = int(part.sum())
    if part_px == 0:
        return 0, 0, 0.0, near, far
    part_med = int(np.median(depth[part]))
    # fill = 부품 bbox 안에서 실제 부품 픽셀 비율(구멍 적을수록 1에 가까움)
    ys, xs = np.where(part)
    bbox_area = max((ys.max()-ys.min()+1) * (xs.max()-xs.min()+1), 1)
    fill = 100.0 * part_px / bbox_area
    return part_med, part_px, fill, near, far


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=os.environ.get("BASLER_BLAZE_IP", "<BLAZE_IP>"))
    ap.add_argument("--out", default=os.path.expanduser("~/Desktop/blaze_step0"))
    ap.add_argument("--scale", type=float, default=1.5, help="창 확대 배율(데이터 무관, 보기용)")
    ap.add_argument("--exposure", type=float, default=None, help="ToF 노출 us(지정 시 수동). 미지정=기본")
    ap.add_argument("--short-range", action="store_true", help="OperatingMode=ShortRange(근거리 선명)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cam = open_blaze(args.ip, args.exposure, args.short_range)
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    print("라이브 시작 — [s] 저장 / [q]·ESC 종료")
    win = "Blaze depth (close=warm)"
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
            valid = depth[depth > 0]
            pct = 100*valid.size/depth.size                       # 전체 valid(참고용)
            p_med, p_px, fill, near, far = part_stats(depth)      # 부품 중심 지표(핵심)
            vis = colorize(depth)
            if args.scale != 1.0:
                vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale,
                                 interpolation=cv2.INTER_NEAREST)
            # 부품 거리(Blaze FOV 넓어 0.8~1.0m가 현실 최적: 부품만 크게+박스벽 화각밖)
            # ·부품 픽셀(클수록 화각 채움)·채움률(구멍 적을수록 ↑)
            ok = "OK" if (700 <= p_med <= 1100 and p_px > 4000 and fill >= 25) else ".."
            txt1 = f"PART med={p_med}mm px={p_px//1000}k fill={fill:.0f}% [{ok}]"
            txt2 = f"(all valid={pct:.0f}% band={near}-{far}mm fps={fps:.1f})"
            cv2.putText(vis, txt1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(vis, txt2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.putText(vis, "[s]save [q]quit", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
            cv2.imshow(win, vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            if k == ord('s'):
                stamp = datetime.now().strftime("%H%M%S")
                base = os.path.join(args.out, f"live_{stamp}")
                np.save(base + ".npy", depth)
                cv2.imwrite(base + ".png", vis)
                print(f"saved {base}.npy/.png  PART med={p_med}mm px={p_px}({p_px//1000}k) "
                      f"fill={fill:.0f}%  band={near}-{far}mm  all_valid={pct:.0f}%")
    finally:
        cam.StopGrabbing(); cam.Close(); cv2.destroyAllWindows()
        print("종료. 저장 →", args.out)


if __name__ == "__main__":
    main()
