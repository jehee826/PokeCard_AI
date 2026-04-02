import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import os

# 1. 모델 설정
MODEL_NAME = 'mobilenetv4_conv_small'
# ⚠️ 중요: 학습할 때 사용한 클래스 개수(폴더 개수)
NUM_CLASSES = 3 
# ⚠️ 중요: 학습할 때 폴더 순서대로 이름을 적으세요
MY_CATEGORIES = ["charizard","emolga", "pikachu"] 

def load_my_model():
    # 뼈대 만들기
    model = timm.create_model(MODEL_NAME, pretrained=False)
    
    # 학습 때 수정했던 것과 똑같이 마지막 층 교체
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    
    # 저장된 가중치(.pth) 경로
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    weights_path = os.path.join(current_dir, "pokecard_model.pth")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {weights_path}")
        
    # 가중치 로드
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval() # 추론 모드 확정
    return model

# 2. 이미지 전처리 (학습 때와 동일하게 설정)
transform = transforms.Compose([
    transforms.Resize((224, 224)), # MobileNet V4 표준 크기
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 추론 실행 함수
def predict(image_path):
    print(f"🔄 모델 로딩 중...")
    model = load_my_model()
    
    # 이미지 로드 및 전처리
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0) # (1, 3, 224, 224)

    print(f"📸 이미지 분석 중: {os.path.basename(image_path)}")
    with torch.no_grad():
        output = model(img_tensor)
        
    # 확률 변환 (클래스가 1개일 때는 Sigmoid, 2개 이상일 때는 Softmax 사용)
    if NUM_CLASSES == 1:
        probability = torch.sigmoid(output)[0]
        print(f"\n--- 분석 결과 ---")
        print(f"결과: {MY_CATEGORIES[0]}일 확률 ({probability.item()*100:.2f}%)")
    else:
        probabilities = F.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, len(MY_CATEGORIES))
        print(f"\n--- 분석 결과 ---")
        for i in range(top_prob.size(0)):
            print(f"{i+1}: {MY_CATEGORIES[top_catid[i]]} ({top_prob[i].item()*100:.2f}%)")

# 4. 메인 실행
if __name__ == "__main__":
    test_img = r"C:\Users\USER\Downloads\에몽가추론용.webp" 
    
    if os.path.exists(test_img):
        predict(test_img)
    else:
        print(f"❌ 테스트 이미지를 찾을 수 없습니다: {test_img}")