import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import os

# 1. 경로 및 카테고리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
# 가중치 파일 이름 확인
WEIGHTS_NAME = "pokecard_model.pth"
MODEL_NAME = 'mobilenetv4_conv_small'

# 카테고리 자동 로드
pokemonName_path = os.path.abspath(os.path.join(current_dir, "..", "dataset", "origin"))
MY_CATEGORIES = sorted([f for f in os.listdir(pokemonName_path) 
                       if os.path.isdir(os.path.join(pokemonName_path, f))])
NUM_CLASSES = len(MY_CATEGORIES)
print(MY_CATEGORIES)

# 장치 설정 (GPU 있으면 GPU 사용)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_my_model():
    print(f"🔄 모델 구조 생성 중: {MODEL_NAME} (Classes: {NUM_CLASSES})")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    
    weights_path = os.path.join(current_dir, WEIGHTS_NAME)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {weights_path}")
        
    # 가중치 로드 (장치에 맞게)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ 모델 로드 완료 ({device})")
    return model

# 전처리 (학습 때와 100% 일치해야 함)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 전역 변수로 모델 한 번만 로드
loaded_model = load_my_model()

def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    print(f"📸 이미지 분석 중: {os.path.basename(image_path)}")
    with torch.no_grad():
        output = loaded_model(img_tensor)
        
    probabilities = F.softmax(output[0], dim=0)
    
    # 상위 5개만 출력하도록 설정 (k=min(5, 클래스수))
    top_k = min(5, len(MY_CATEGORIES))
    top_prob, top_catid = torch.topk(probabilities, top_k)
    
    print(f"\n--- 분석 결과 (Top {top_k}) ---")
    for i in range(top_prob.size(0)):
        prob = top_prob[i].item() * 100
        category = MY_CATEGORIES[top_catid[i]]
        print(f"{i+1}위: {category.ljust(15)} {prob:.2f}%")

if __name__ == "__main__":
    # r""을 쓰거나 /를 써서 경로 에러 방지
    test_img = os.path.abspath(os.path.join(current_dir, "..", "dataset", "test", "mimikyu.jpg"))
    
    if os.path.exists(test_img):
        predict(test_img)
    else:
        print(f"❌ 테스트 이미지를 찾을 수 없습니다: {test_img}")