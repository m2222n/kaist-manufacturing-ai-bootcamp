#!/usr/bin/env python3
"""
Blaze-112 라이브 뷰어 — 맥북에서 실행. 실시간 depth 화면 + 키로 캡처.

GigE TL + IP 직접 (6/23 검증) + Range만(Intensity off).
화면: depth를 컬러맵으로 표시(가까울수록 따뜻한색), 좌상단에 거리/유효픽셀 표시.
키:
  s  = 현재 프레임 저장 (~/Desktop/blaze_step0/live_HHMMSS.npy + .png)
  q 또는 ESC = 종료

실행:
  sudo .venv/binpick/bin/python ~/Desktop/blaze_live_view.py
  sudo .venv/binpick/bin/python ~/Desktop/blaze_live_view.py --ip <BLAZE_IP>
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


def open_blaze(ip: str):
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
    return cam


def colorize(depth: np.ndarray) -> np.ndarray:
    """depth(uint16 mm) → 컬러맵 BGR. 가까울수록 밝고 따뜻하게, 배경(0)=검정."""
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    norm = np.clip((depth.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    vis = (255 * (1 - norm)).astype(np.uint8)        # 가까울수록 밝게
    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    color[depth == 0] = (0, 0, 0)                    # 무효 = 검정
    return color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=os.environ.get("BASLER_BLAZE_IP", "<BLAZE_IP>"))
    ap.add_argument("--out", default=os.path.expanduser("~/Desktop/blaze_step0"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cam = open_blaze(args.ip)
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    print("라이브 시작 — [s] 저장 / [q]·ESC 종료")
    win = "Blaze depth (close=warm)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
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
