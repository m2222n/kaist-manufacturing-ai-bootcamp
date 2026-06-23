# Blaze 시험촬영(Step0) 사용법 — 맥북에서

> 목적: 빈에 부품을 몇 개 깔았을 때 depth로 잘 구분되는지(=장당 적정 부품 수 K) 측정 → 조교 공유.
> ⚠️ 맥북에서만 동작(6000엔 pylon SDK 없음). Blaze는 **24V 전원 전용**(12V 금지).

## 0. 준비
```bash
pip install pypylon numpy opencv-python
```
카메라: 24V 전원 + 이더넷 연결. (5/12에 Mac에서 풀작동 검증된 워크플로우)

## 1. 연결 확인 (저장 없이 한 프레임)
```bash
python blaze_test_capture.py --preview
```
→ `유효픽셀 %`, `depth min/med/max(mm)`가 정상으로 찍히면 OK. (med 700~1000mm 근처면 정상)

## 2. 시험촬영 — A/B/C 밀도별
빈에 부품을 4개 / 8개 / 12개 깔고 각각 찍습니다(위치·자세 다양하게):
```bash
python blaze_test_capture.py --tag A_4parts  --n 3
python blaze_test_capture.py --tag B_8parts  --n 3
python blaze_test_capture.py --tag C_12parts --n 3
```
저장 위치: `~/Desktop/blaze_step0/` (npy = raw depth, png = 미리보기).

## 3. 조교 공유
`~/Desktop/blaze_step0/`의 PNG들을 조교께 보내고 물어볼 것:
- "이 밀도(4/8/12개)에서 depth로 부품 구분이 되나요? 장당 몇 개가 적당할까요?"
- "이 raw 포맷(848×480 uint16 mm)으로 본촬영해도 되나요, 아니면 합성처럼 m 단위·512로 맞출까요?"

## 4. 다음 (K값 확정 후)
조교가 장당 개수 K를 정해주면 → Claude가 **least-used-first 100장 배치표**(27종 ±1 균등) 생성
→ 표대로 본촬영(label은 파일명/폴더로, RGB X·마스크 X).
label 표 = `~/kaist_project/docs/label_map_27parts.md` (합성 1000장과 동일 번호, 새로 정하지 말 것).

## 참고
- 출력 = uint16 mm depth (848×480). 합성 데이터(조교 코드)는 float32 meter + 배경 NaN, 512×512.
  → Step0는 밀도·포맷 확인이 목적이라 raw 저장. 본촬영 포맷 변환은 K값·조교 컨펌 후.
- 검증 워크플로우 상세: memory `project_basler_setup_history.md`
