AI 모델들 레포지토리
=====
폴더구조
-----
1.MobileNet_V4  
이미지 추론 AI 모델  
2.PaddleOCR  
이미지 텍스트 추론 모델    
  
MobileNet_V4
-----
1.PlusData.py    
dataset/origin 안의 원본파일들과 dataset/background 안의 배경사진을 사용해 직접 찍은사진처럼 데이터증강을 진행함  
2.Train.py  
1번에서 미리 만들어둔 데이터셋 폴더를 활용해 All-Fine-Tuning (Freezing 사용 X)  
3.Inference.py  
추론파일 맨아래 원하는 img url을 넣으면 해당파일로 기존 학습된 가중치를 사용해 추론함    
4.export_onnx.py  
위에서 사용한 모델을 onnx로 추출 가중치 파일과 모델의 뼈대를 서로 합쳐 하나의 onnx파일로 만들어줌

PaddleOCR
-----
1.Inference.py  
main함수에 원하는 이미지 url을 넣으면 해당 이미지에서 텍스트박스를 검출해 해당 박스를 각각 추론한 뒤 이미지의 어느부분에서 추출한지와 추론결과를 말해주는 하나의 이미지를 생성함  
2.export_onnx.txt  
위의 MobileNet_V4와 다르게 그냥 터미널에서 변환하는게 더 간결하고 확실해서 터미널 명령어를 여기에 적어둠 중간에 --save_file 옵션에 저장하고싶은 경로를 적으면됨
