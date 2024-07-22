**FD_Grid Organized**

[480:640 = 3:4]

1. INPUT_VIDEO_01
(1080,1560), (440,1080)
contrast=1, brightness=-50
# of failed FD frame => 0
# of frame precision < 0.95 => 6
# of low precision frames: [13, 16, 17, 19, 26, 27]

@정확도 저하의 원인 분석: 고개를 숙였을 시 정확도가 떨어짐. FD의 기준인 턱이 아예 관측되지 않은 것을 원인으로 추정

2. INPUT_VIDEO_02
(1080,1560), (200,840)
contrast=1, brightness=-50
# of failed FD frame => 0
# of frame precision < 0.95 => 0
# of low precision frames: []

3. INPUT_VIDEO_03
(660,1140), (0,640)
contrast=1, brightness=-42
# of failed FD frame => 0
# of frame precision < 0.95 => 5
# of low precision frames: [77, 89, 128, 129, 439]

4. INPUT_VIDEO_04
(700,1100), (0,650)
contrast=1, brightness=-42
# of failed FD frame => 2
# of frame precision < 0.95 => 37
# of low precision frames: [148, 154, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 173, 174, 175, 176, 177, 219, 377, 379, 380, 382, 383, 384, 385, 386, 387, 393, 399, 400, 401, 402, 403, 407, 410, 413]

@ 정확도 저하의 원인 분석: 고개를 오른쪽으로 돌렸을 시(운전자 기준), 좌측 창가에서 들어오는 강한 햇빛에 의해 턱선이 탐지 되지 않음. 하지만 input03과 거의 같은 상황일 때에도 03의 얼굴은 탐지되는 반면, 04는 탐지가 되지 않음. 따라서 단순 턱선 감지 유무가 원인은 아닌 것으로 고려됨.