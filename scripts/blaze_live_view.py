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
            med = int(np.median(valid)) if valid.size else 0
            pct = 100*valid.size/depth.size
            vis = colorize(depth)
            if args.scale != 1.0:
                vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale,
                                 interpolation=cv2.INTER_NEAREST)
            txt = f"med={med}mm valid={pct:.0f}% fps={fps:.1f}"
            cv2.putText(vis, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(vis, "[s]save [q]quit", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.imshow(win, vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            if k == ord('s'):
                stamp = datetime.now().strftime("%H%M%S")
                base = os.path.join(args.out, f"live_{stamp}")
                np.save(base + ".npy", depth)
                cv2.imwrite(base + ".png", vis)
                print(f"saved {base}.npy/.png  med={med}mm valid={pct:.0f}%")
    finally:
        cam.StopGrabbing(); cam.Close(); cv2.destroyAllWindows()
        print("종료. 저장 →", args.out)


if __name__ == "__main__":
    main()
