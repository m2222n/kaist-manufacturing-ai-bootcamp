# 27종 부품 Label 매핑표 (확정)

> label = 합성 데이터 생성기(`gen_one_2denc.py`)의 `sorted(glob("*.stl"))` 순서 + 1.
> 합성 1000장(`dataset_2denc`)이 이미 이 번호(1~27)로 생성됨. **실물 100장도 반드시 이 표와 동일하게 label 부여.**

| label | 부품 (STL 파일명) |
|-------|-------------------|
| 1 | 01_sol_block_a |
| 2 | 02_sol_block_b |
| 3 | 03_sol_block_front |
| 4 | 06_sol_block_back |
| 5 | 07_guide_paper_l |
| 6 | 08_r_guide_a |
| 7 | 09_guide_paper_r |
| 8 | 11_sw_block |
| 9 | 13_variant |
| 10 | 13_x2_bcf8ccb4 |
| 11 | 14_13 |
| 12 | 15_roller_bracket |
| 13 | 16_cam_f_bracket |
| 14 | 17_mks_holder |
| 15 | 18_button_function_niro |
| 16 | bracket_case |
| 17 | bracket_sen_1 |
| 18 | bracket_sensor1 |
| 19 | bracket_sensor2 |
| 20 | brkt_switch |
| 21 | guide_paper_roll_cover_left |
| 22 | guide_paper_roll_cover_right |
| 23 | main_body |
| 24 | plate_e |
| 25 | r_guide_a_l |
| 26 | r_guide_a_r |
| 27 | top_inner_sheet |

총 27종. (원본 28종 중 `10_guide_paper_roll_l`은 Form4 빌드볼륨 초과로 제외)
