from PIL import Image
from torchvision import transforms
import os

# 1. 원본 webp 이미지 로드
image_path = r'C:\Users\jehee\vscode-workspace\Ai\dataset\train\emolga\Emolga.jpg'  # 확장자를 webp로 수정
sava_path = r'C:\Users\jehee\vscode-workspace\Ai\dataset\train\emolga' #저장할 폴더
image_name = "emolga" #저장할 이름명 image_name_(001)식으로 생성됨

if not os.path.exists(image_path):
    print(f"❌ 파일을 찾을 수 없습니다: {image_path}")
    exit()

# webp를 열고 RGB 모드로 변환 (RGBA인 경우 투명도 제거를 위해)
original_img = Image.open(image_path).convert('RGB')

# 2. 증강 파이프라인 (기존과 동일)
augmentation_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 약간의 위치 이동 추가
])

# 3. 저장 폴더 생성
os.makedirs(sava_path, exist_ok=True)

# 4. 100장 생성
for i in range(100):
    augmented_img = augmentation_pipeline(original_img)
    # 학습 효율을 위해 jpg로 저장하는 것을 추천합니다
    save_path = os.path.join(sava_path, f'{image_name}_{i:03d}.jpg')
    augmented_img.save(save_path, 'JPEG')

print(f"✅ webp 원본으로부터 100장의 jpg 생성 완료!")