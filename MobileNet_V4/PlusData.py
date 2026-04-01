from PIL import Image
from torchvision import transforms
import os

#기본옵션설정
folder_name = "pikachu" #폴더명, 저장할이미지이름 저장 

current_dir = os.path.dirname(os.path.abspath(__file__)) 
# 윈도우 경로 구분자(\) 문제를 방지하기 위해 콤마(,)로 연결하는 것이 안전합니다.
train_path = os.path.join(current_dir, "..", "dataset", "train", folder_name)

# 1. 원본 이미지 리스트 (파일명만 리스트로 관리하면 훨씬 깔끔합니다)
original_files = ["pikachu.webp", "pikachu2.webp", "pikachu3.webp", "pikachu4.jpg", "pikachu5.jpg"] #train_path아래에있는 파일들
save_path_dir = train_path # 저장할 폴더

# 이미지 로드 및 변환
original_imgs = []
for file_name in original_files:
    full_path = os.path.join(train_path, file_name)
    if os.path.exists(full_path):
        img = Image.open(full_path).convert('RGB')
        original_imgs.append(img)
    else:
        print(f"⚠️ 파일을 찾을 수 없습니다: {full_path}")

# 2. 증강 파이프라인
augmentation_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(p=0.5)
])

# 3. 저장 폴더 생성
os.makedirs(save_path_dir, exist_ok=True)

# 4. 이미지 생성 (총 500장)
count = 0
for i, img in enumerate(original_imgs):
    for j in range(100):
        augmented_img = augmentation_pipeline(img)
        
        # 파일명에 원본 번호(i)와 증강 번호(j)를 조합하거나 전체 카운트를 사용해 중복 방지
        file_name = f'{folder_name}_orig{i}_{j:03d}.jpg'
        final_save_path = os.path.join(save_path_dir, file_name)
        
        augmented_img.save(final_save_path, 'JPEG')
        count += 1

print(f"✅ 총 {count}장의 jpg 생성 완료! (경로: {save_path_dir})")