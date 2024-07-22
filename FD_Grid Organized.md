**FD_Grid Organized**

1. INPUT_VIDEO_01
(900,1700), (200,1080)
contrast=1, brightness=-50
# of failed FD frame => 0
# of frame precision < 0.95 => 4
# of low precision frames: [28, 29, 30, 31]

@정확도 저하의 원인 분석: 고개를 숙였을 시 정확도가 떨어짐. FD의 기준인 턱이 아예 관측되지 않은 것을 원인으로 추정

2. INPUT_VIDEO_02
(900,1700), (200,1080)
contrast=1, brightness=-50
# of failed FD frame => 0
# of frame precision < 0.99 => 1
# of low precision frames: []

3. INPUT_VIDEO_03
(700,1100), (0,650)
contrast=1, brightness=-42
# of failed FD frame => 0
# of frame precision < 0.97 => 0 
#                    < 0.98 => 9
# of low precision frames: [78, 83, 84, 85, 87, 88, 89, 125, 127]

4. INPUT_VIDEO_04
(700,1100), (0,650)
contrast=1, brightness=-42
# of failed FD frame => 0
# of frame precision < 0.95 => 30
# of low precision frames: [150, 153, 155, 164, 167, 168, 169, 170, 171, 181, 378, 379, 381, 383, 385, 386, 387, 388, 393, 394, 395, 397, 398, 399, 403, 404, 405, 406, 407, 410]

@ 정확도 저하의 원인 분석: 고개를 오른쪽으로 돌렸을 시(운전자 기준), 좌측 창가에서 들어오는 강한 햇빛에 의해 턱선이 탐지 되지 않음. 하지만 input03과 거의 같은 상황일 때에도 03의 얼굴은 탐지되는 반면, 04는 탐지가 되지 않음. 따라서 단순 턱선 감지 유무가 원인은 아닌 것으로 고려됨.