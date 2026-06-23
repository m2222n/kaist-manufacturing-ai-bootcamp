#!/usr/bin/env python3
"""
Blaze-112 시험촬영 스크립트 (KAIST 실물 데이터셋 Step0) — 맥북에서 실행.

목적 (6/19 미팅 ⑤, Step0):
  - 빈에 부품을 A=4 / B=8 / C=12개 깔고 depth로 부품이 구분되는 한계점(K) 측정.
  - 결과(depth .npy + 미리보기 PNG)를 조교께 공유해 "장당 적정 부품 수" + 포맷 OK 먼저 받기.
  - ⚠️ K(장당 개수)는 책상에서 정하지 말고 Blaze 인식력으로 결정 (사용자 지적).

검증된 워크플로우 (메모리 basler_setup_history, 5/12 Mac 풀작동):
  pypylon + BaslerGigE TL + ComponentSelector="Range" + ComponentEnable=True
  + PixelFormat Mono16(raw) → 848x480 uint16, mm 단위 depth.
  (pylon Viewer는 macOS Blaze 미지원 → pypylon 단독)

설치: pip install pypylon numpy opencv-python
실행 예:
  python blaze_test_capture.py --tag A_4parts        # 4개 깔고 한 장
  python blaze_test_capture.py --tag B_8parts --n 3  # 8개 깔고 3장 연사
  python blaze_test_capture.py --preview             # 저장 없이 라이브 한 프레임 확인

⚠️ 합성 데이터(조교 코드)는 float32 meter + 배경 NaN, 512x512.
   본 스크립트는 raw(848x480 uint16 mm)를 그대로 저장 = Step0는 포맷·밀도 확인이 목적.
   본촬영(100장) 포맷 변환(m 단위/배경 마스킹/512 리사이즈)은 K값·조교 컨펌 후 별도 스크립트로.
"""
from __future__ import annotations
import argparse, os, sys, time
from datetime import datetime
import numpy as np

try:
    from pypylon import pylon
except Exception as e:
    sys.exit("pypylon import 실패. 'pip install pypylon' 후 맥북에서 실행하세요. (%s)" % e)


def open_blaze(ip: str | None = None):
    """Blaze-112(ToF) 카메라 열기.

    ⚠️ macOS에서 pylon 자동 탐색(EnumerateDevices)이 GigE 브로드캐스트를 못 잡아
       Found 0이 나옴 → **GigE TL 명시 + IP 직접 지정**이 정답 (6/23 검증).
       기본 IP = <BLAZE_IP> (Blaze static, Mac en8=<MAC_IP>).
    """
    ip = ip or "<BLAZE_IP>"
    tl = pylon.TlFactory.GetInstance()
    tl.CreateTl("BaslerGigE")  # GigE TL 명시 (자동탐색 우회)
    di = pylon.CDeviceInfo()
    di.SetIpAddress(ip)
    di.SetDeviceClass("BaslerGigE")
    try:
        cam = pylon.InstantCamera(tl.CreateDevice(di))
        cam.Open()
    except Exception as e:
        sys.exit("Blaze 연결 실패(IP %s): %s\n"
                 "확인: ping -c3 %s / Mac en8=<MAC_IP> / 24V 전원(12V금지)" % (ip, e, ip))
    print("연결:", cam.GetDeviceInfo().GetModelName(), cam.GetDeviceInfo().GetIpAddress())
    return cam


def configure_range(cam):
    """Range(depth) 컴포넌트만 활성, Intensity는 끔 (6/23 검증).

    ⚠️ blaze-112-GEV는 기본적으로 Intensity+Range 둘 다 enable=True →
       두 이미지가 세로로 붙어 (960,848)으로 나옴. Intensity를 꺼야 Range(480,848)만 나옴.
       Range PixelFormat = Coord3D_C16 (mm 단위 depth). 안 되면 Mono16 fallback.
    """
    # 1) Intensity 끄기
    try:
        cam.ComponentSelector.SetValue("Intensity")
        cam.ComponentEnable.SetValue(False)
    except Exception as e:
        print("⚠️ Intensity off 경고:", e)
    # 2) Range만 켜기 + depth 포맷
    try:
        cam.ComponentSelector.SetValue("Range")
        cam.ComponentEnable.SetValue(True)
        try:
            cam.PixelFormat.SetValue("Coord3D_C16")  # mm 단위 depth (정석)
        except Exception:
            cam.PixelFormat.SetValue("Mono16")        # fallback (동일 정보)
    except Exception as e:
        print("⚠️ Range 설정 경고(모델별 노드명 차이 가능):", e)


def grab_depth(cam, timeout_ms: int = 2000) -> np.ndarray:
    """한 프레임 depth(uint16, mm) 반환."""
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    res = cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
    if not res.GrabSucceeded():
        res.Release(); cam.StopGrabbing()
        raise RuntimeError("Grab 실패: %s" % res.ErrorDescription)
    depth = res.Array.copy()
    res.Release(); cam.StopGrabbing()
    return depth


def summarize(depth: np.ndarray) -> str:
    valid = depth[depth > 0]
    if valid.size == 0:
        return "유효 depth 0 (전부 0) — 거리/노출 확인 필요"
    return ("shape=%s dtype=%s  유효픽셀=%.1f%%  depth(mm) min=%d med=%d max=%d"
            % (depth.shape, depth.dtype, 100*valid.size/depth.size,
               int(valid.min()), int(np.median(valid)), int(valid.max())))


def save_preview(depth: np.ndarray, path_png: str):
    """가까울수록 밝게 정규화한 미리보기 PNG (배경=검정)."""
    try:
        import cv2
    except Exception:
        print("opencv 없음 → PNG 미리보기 건너뜀 (npy만 저장)"); return
    valid = depth[depth > 0]
    if valid.size == 0:
        cv2.imwrite(path_png, np.zeros_like(depth, np.uint8)); return
    lo, hi = np.percentile(valid, [2, 98])
    vis = np.clip((depth.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    vis = (255 * (1 - vis)).astype(np.uint8)        # 가까울수록 밝게
    vis[depth == 0] = 0                              # 배경 검정
    cv2.imwrite(path_png, vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="test", help="장면 태그 (예: A_4parts/B_8parts/C_12parts)")
    ap.add_argument("--n", type=int, default=1, help="연속 촬영 장수")
    ap.add_argument("--out", default=os.path.expanduser("~/Desktop/blaze_step0"), help="저장 폴더(맥북 바탕화면)")
    ap.add_argument("--ip", default=os.environ.get("BASLER_BLAZE_IP"), help="Blaze IP (옵션, 환경변수 BASLER_BLAZE_IP)")
    ap.add_argument("--preview", action="store_true", help="저장 없이 한 프레임 요약만")
    args = ap.parse_args()

    cam = open_blaze(args.ip)
    configure_range(cam)

    if args.preview:
        d = grab_depth(cam)
        print("[PREVIEW]", summarize(d))
        cam.Close(); return

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S")
    for i in range(args.n):
        d = grab_depth(cam)
        base = os.path.join(args.out, f"{args.tag}_{stamp}_{i:02d}")
        np.save(base + ".npy", d)                    # raw uint16 mm
        save_preview(d, base + ".png")
        print(f"[{i+1}/{args.n}] saved {base}.npy/.png  |  {summarize(d)}")
        if args.n > 1:
            time.sleep(0.3)
    cam.Close()
    print("\n완료 →", args.out)
    print("👉 A/B/C 태그별로 찍은 뒤 PNG를 조교께 공유 → '장당 적정 부품 수(K)' 확인받기.")


if __name__ == "__main__":
    main()
