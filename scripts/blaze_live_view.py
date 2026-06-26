#!/usr/bin/env python3
"""
Blaze-112 라이브 뷰어 — 맥북에서 실행. 실시간 depth 화면 + 키로 캡처.

GigE TL + IP 직접 (6/23 검증) + Range만(Intensity off).
화면: depth를 컬러맵으로 표시(가까울수록 따뜻한색), 좌상단에 거리/유효픽셀 표시.

화질 개선 (6/26):
  - 데이터 품질: ToF 노출/누적(Exposure·OperatingMode HDR)·온칩 필터(Spatial/Temporal/Outlier)
    GenICam 노드가 있으면 켬 → valid% / 선명도 상승. 없으면 조용히 건너뜀(모델차 안전).
  - 표시 품질: 창 업스케일(--scale), confidence/거리 게이팅으로 배경 구멍 정리,
    ROI 윈도우(가까운 작업영역만 정규화)로 부품 대비 강화, 무효 픽셀 인페인트(선택).
키:
  s  = 현재 프레임 저장 (~/Desktop/blaze_step0/live_HHMMSS.npy + .png)
  r  = 거리 정규화 범위 리셋(자동)
  [ / ] = 정규화 근/원거리 클립 조절(대비)
  i  = 인페인트 토글(배경 구멍 메우기, 표시 전용)
  q 또는 ESC = 종료

실행:
  sudo .venv/binpick/bin/python ~/Desktop/blaze_live_view.py
  sudo .venv/binpick/bin/python ~/Desktop/blaze_live_view.py --ip <BLAZE_IP> --scale 2
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
    """GenICam 노드 set은 모델·펌웨어별로 없을 수 있음 → 있으면 적용, 없으면 조용히 패스."""
    try:
        fn()
        print(f"  ✓ {label}")
        return True
    except Exception as e:
        print(f"  · {label} 건너뜀 ({type(e).__name__})")
        return False


def _set_enum(cam, node, value):
    getattr(cam, node).SetValue(value)


def tune_quality(cam, exposure_us: float | None, hdr: bool):
    """ToF depth 품질 향상 파라미터 — 있는 노드만 적용.

    Blaze는 모델/펌웨어에 따라 노드명이 다름. 가장 흔한 후보들을 시도하고
    실패하면 넘어간다(기본값으로 동작). 핵심:
      - OperatingMode: ShortRange/LongRange/HDR — 작업거리 0.5~0.8m면 ShortRange가 선명
      - ExposureTime: 길수록 SNR↑(=valid%↑)이나 너무 길면 모션블러/과포화
      - Spatial/Temporal/Outlier 필터: flying pixel·노이즈 제거 → 가장자리 선명
    """
    print("[품질 파라미터 적용] (없는 노드는 자동 건너뜀)")
    # 작업거리 짧음 → ShortRange / HDR
    if hdr:
        _try("OperatingMode=HDR", lambda: _set_enum(cam, "OperatingMode", "Hdr"))
    else:
        _try("OperatingMode=ShortRange", lambda: _set_enum(cam, "OperatingMode", "ShortRange"))
    # 노출
    if exposure_us is not None:
        _try(f"ExposureTime={exposure_us}us", lambda: cam.ExposureTime.SetValue(float(exposure_us)))
        # 일부 펌웨어는 ExposureAuto가 자동을 강제 → 수동으로
        _try("ExposureAuto=Off", lambda: _set_enum(cam, "ExposureAuto", "Off"))
    # 온칩 depth 필터 (노드명 후보 여러 개 시도)
    for node in ("SpatialFilter", "Scan3dSpatialFilter"):
        if _try(f"{node}=On", lambda n=node: getattr(cam, n).SetValue(True)):
            break
    for node in ("TemporalFilter", "Scan3dTemporalFilter"):
        if _try(f"{node}=On", lambda n=node: getattr(cam, n).SetValue(True)):
            break
    for node in ("OutlierRemoval", "Scan3dOutlierRemoval", "FlyingPixelsRemoval"):
        if _try(f"{node}=On", lambda n=node: getattr(cam, n).SetValue(True)):
            break
    # confidence 임계 (저신뢰 픽셀 제거) — 노드 있으면
    _try("ConfidenceThreshold up", lambda: cam.ConfidenceThreshold.SetValue(
        max(cam.ConfidenceThreshold.GetMin(), cam.ConfidenceThreshold.GetValue())))


def open_blaze(ip: str, exposure_us, hdr):
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
    tune_quality(cam, exposure_us, hdr)
    return cam


def colorize(depth: np.ndarray, lo: float, hi: float, near_mm: int, far_mm: int,
             inpaint: bool) -> np.ndarray:
    """depth(uint16 mm) → 컬러맵 BGR.

    개선점:
      - 작업영역(near_mm~far_mm) 밖은 무효 처리 → 빈 너머 벽/바닥이 화면을 안 깎음
      - 그 영역 안에서만 lo~hi 정규화 → 부품 표면 대비 강화(전역 정규화보다 또렷)
      - inpaint=True면 작은 구멍(0)을 주변값으로 메워 깔끔하게 표시(표시 전용, 저장 raw는 그대로)
    """
    d = depth.astype(np.float32)
    work = (depth > near_mm) & (depth < far_mm)        # 작업거리 게이팅
    if work.sum() == 0:
        return np.zeros((*depth.shape, 3), np.uint8)
    norm = np.clip((d - lo) / max(hi - lo, 1), 0, 1)
    vis = (255 * (1 - norm)).astype(np.uint8)          # 가까울수록 밝게
    vis[~work] = 0
    if inpaint:
        holes = ((depth == 0) | (~work)).astype(np.uint8)
        # 작업영역 내부의 작은 구멍만 메움(가장자리 전부 메우면 왜곡)
        vis = cv2.inpaint(vis, holes, 2, cv2.INPAINT_TELEA)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)  # JET보다 단계 구분 좋음
    color[~work] = (30, 30, 30)                         # 작업영역 밖 = 진회색(검정보다 덜 지저분)
    return color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=os.environ.get("BASLER_BLAZE_IP", "<BLAZE_IP>"))
    ap.add_argument("--out", default=os.path.expanduser("~/Desktop/blaze_step0"))
    ap.add_argument("--scale", type=float, default=2.0, help="표시 확대 배율(데이터 무관, 보기용)")
    ap.add_argument("--near", type=int, default=300, help="작업영역 최소거리 mm (이보다 가까우면 무효)")
    ap.add_argument("--far", type=int, default=1200, help="작업영역 최대거리 mm (이보다 멀면 무효=배경 컷)")
    ap.add_argument("--exposure", type=float, default=None, help="ToF 노출 us (지정 시 수동). 미지정=카메라 기본")
    ap.add_argument("--hdr", action="store_true", help="OperatingMode=HDR (반사 심한 장면)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cam = open_blaze(args.ip, args.exposure, args.hdr)
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    print("라이브 시작 — [s]저장 [r]범위리셋 [/[] 대비 [i]인페인트 [q]종료")
    win = "Blaze depth (close=warm)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(848*args.scale), int(480*args.scale))
    last = time.time(); fps = 0.0
    auto_range = True          # True면 작업영역 픽셀로 lo/hi 자동
    lo = hi = None
    inpaint = False
    near, far = args.near, args.far
    try:
        while True:
            res = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
            if not res.GrabSucceeded():
                res.Release(); continue
            depth = res.Array.copy(); res.Release()
            now = time.time(); fps = 0.9*fps + 0.1*(1.0/max(now-last, 1e-3)); last = now

            work = depth[(depth > near) & (depth < far)]
            if auto_range and work.size:
                lo, hi = np.percentile(work, [2, 98])   # 작업영역만으로 대비 잡기
            if lo is None:
                lo, hi = near, far
            med = int(np.median(work)) if work.size else 0
            pct = 100*work.size/depth.size              # 작업영역 valid%
            vis = colorize(depth, lo, hi, near, far, inpaint)
            if args.scale != 1.0:
                vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale,
                                 interpolation=cv2.INTER_NEAREST)
            txt = f"med={med}mm valid={pct:.0f}% fps={fps:.1f} range={int(lo)}-{int(hi)}mm work={near}-{far}"
            cv2.putText(vis, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(vis, "[s]save [r]reset [/[]contrast [i]inpaint [q]quit",
                        (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210,210,210), 1)
            cv2.imshow(win, vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('r'):
                auto_range = True
            elif k == ord(']'):
                auto_range = False; hi = (hi or far) + 50
            elif k == ord('['):
                auto_range = False; hi = max((lo or near) + 50, (hi or far) - 50)
            elif k == ord('i'):
                inpaint = not inpaint; print("inpaint:", inpaint)
            elif k == ord('s'):
                stamp = datetime.now().strftime("%H%M%S")
                base = os.path.join(args.out, f"live_{stamp}")
                np.save(base + ".npy", depth)            # raw uint16 mm (가공 X)
                cv2.imwrite(base + ".png", vis)          # 표시 그대로(확대·컬러맵 포함)
                print(f"saved {base}.npy/.png  med={med}mm valid={pct:.0f}%")
    finally:
        cam.StopGrabbing(); cam.Close(); cv2.destroyAllWindows()
        print("종료. 저장 →", args.out)


if __name__ == "__main__":
    main()
