# FD_Grid

### FD_Grid 프로젝트
- 기존 원래의 크기의 영상 프레임에서 얼굴이 안잡히는 문제가 발생시 FD_Grid를 통해 해결 방안으로 사용할 수 있는가를 확인하는 것이 목표
- 임의로 설정한 크기의 구역에서 모델을 돌렸을 때 얼마나 개선이 되는가 확인이 필요
- 모델을 통해 결과를 얻는 방식만 제공했기에 자세한 로직은 프로젝트 진행자들이 구성해야함

### 필수 설치 패키지 목록
- install_package.sh 파일을 ./install_packages.sh 명령어로 실행하여 설치를 진행
- onnx 1.16.1
- onnxruntime 1.18.1
- opencv-python 4.7.0.68
- numpy 1.26.4
- torch 1.8.1
- torchvision 0.9.1

### Example.py
- 예제용 파일을 참고하여 프로젝트를 진행할 것
- 예제용 코드에는 이미지로 사용하는 예제만 적용되어있음

### onnx_layer_check.py
- onnx 모델파일인 mdfd.onnx의 input & output 레이어의 shape를 볼 수 있는 코드

### FaceBoxesV2
- 해당 경로에 있는 파일들은 FD를 사용하기 위한 코드
- 프로젝트를 진행함에 있어서 예제파일에 사용할만한 함수는 모두 사용했기에 참고용으로만 사용

