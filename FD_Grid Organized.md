**FD_Grid Organized**

1. INPUT_VIDEO_01
(900,1700), (200,1080)
contrast=1, brightness=-50
# of failed FD frame => 0
# of frame precision < 0.95 => 4

@정확도 저하의 원인 분석: 고개를 숙였을 시 정확도가 떨어짐. FD의 기준인 턱이 관측되지 않은 것을 원인으로 추정

2. INPUT_VIDEO_02
(900,1700), (200,1080)
contrast=1, brightness=-50
# of failed FD frame => 0
# of frame precision < 0.99 => 1

3. INPUT_VIDEO_03
(700,1100), (0,650)
contrast=1, brightness=-42
# of failed FD frame => 0
# of frame precision < 0.97 => 0 
#                    < 0.98 => 9

4. INPUT_VIDEO_04
(700,1100), (0,650)
contrast=1, brightness=-42
# of failed FD frame => 1
# of frame precision < 0.95 => 30

@ 정확도 저하의 원인 분석: 고개를 오른쪽으로 돌렸을 시(운전자 기준), 좌측 창가에서 들어오는 강한 햇빛에 의해 턱선이 탐지 되지 않음. 
                         Grayscaling을 진행 후 edge detection을 해본 결과, 턱선이 탐지되지 않음. 
